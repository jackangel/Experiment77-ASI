import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from transformers import GPT2TokenizerFast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ==========================================
# 1. DATASET PREPARATION WITH BPE
# ==========================================
file_path = 'input.txt'

if not os.path.exists(file_path):
    print("input.txt not found. Generating a dummy logic-puzzle dataset...")
    with open(file_path, 'w') as f:
        text = ("Move Up, Move Left -> End at (-1, 1). "
                "Move Down, Move Right -> End at (1, -1). "
                "A connects to B, B connects to C -> A to C. ") * 500
        f.write(text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)
vocab_size = tokenizer.vocab_size

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# --- HYPERPARAMETERS ---
batch_size     = 2
block_size     = 2048     
d_model        = 256
num_layers     = 3
max_iters      = 200000
eval_interval  = 500
learning_rate  = 3e-4
warmup_iters   = 400
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
debug_mode     = False  # Disable by default

# --- NEW VARIABLE LENGTH CONTROLS ---
max_tokens_per_batch = batch_size * block_size  # e.g., 8 * 1024 = 8192 tokens per batch
short_seq_prob       = 0.25  # 5% chance to force a short sequence (<= 64 tokens)
min_seq_len          = 8    # Minimum length for a short proposition

def get_batch(split, variable_length=False):
    d = train_data if split == 'train' else val_data
    
    if variable_length:
        # Control how often we randomly reduce the length size
        if torch.rand(1).item() < short_seq_prob:
            # Short sequence (e.g., single proposition, hits local expert only)
            seq_len = torch.randint(min_seq_len, 65, (1,)).item()
        else:
            # Normal/Long sequence
            seq_len = torch.randint(65, block_size + 1, (1,)).item()
    else:
        seq_len = block_size

    seq_len = min(seq_len, len(d) - 1)
    
    # Dynamic batch sizing: keep total tokens per forward pass constant for GPU efficiency
    current_batch_size = max(1, max_tokens_per_batch // seq_len)
    
    ix = torch.randint(len(d) - seq_len, (current_batch_size,))
    x  = torch.stack([d[i:i+seq_len]   for i in ix])
    y  = torch.stack([d[i+1:i+seq_len+1] for i in ix])

    return x.to(device), y.to(device)

def get_lr(iter_num: int) -> float:
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters
    progress  = (iter_num - warmup_iters) / max(1, max_iters - warmup_iters)
    cos_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr    = learning_rate * 0.1
    return min_lr + (learning_rate - min_lr) * cos_coeff

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

# ==========================================
# 2. LOCAL EXPERT
# ==========================================
class ImprovedHybridReasoningLayer(nn.Module):
    def __init__(self, d_model, window_size=8, summary_decay=0.9):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

        self.ema_kernel_size = min(64, 256)
        self.ema_conv = nn.Conv1d(d_model, d_model, kernel_size=self.ema_kernel_size, padding=self.ema_kernel_size - 1, groups=d_model, bias=False)
        with torch.no_grad():
            weights = torch.zeros(d_model, 1, self.ema_kernel_size)
            for i in range(self.ema_kernel_size):
                weights[:, 0, i] = (1 - summary_decay) * (summary_decay ** i)
            self.ema_conv.weight.copy_(weights.flip(-1))
            self.ema_conv.weight.requires_grad = False

        self.local_conv_size = 4
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=self.local_conv_size, padding=self.local_conv_size - 1, groups=d_model, bias=True)

        self.delta_proj = nn.Linear(d_model * 3, d_model)
        self.gate_proj  = nn.Linear(d_model * 3, d_model)
        self.delta_norm = nn.LayerNorm(d_model)
        self.step_size = 0.5 / math.sqrt(64 / 32)

        self.seq_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)

    def forward(self, parallel_repr, sequential_state, padding_mask=None):
        B, T, D = parallel_repr.size()
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(parallel_repr.device)

        attn_out, attn_weights = self.self_attn(
            parallel_repr, parallel_repr, parallel_repr,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        parallel_repr = self.ln1(parallel_repr + attn_out)

        local_history = self.local_conv(parallel_repr.transpose(1, 2))[:, :, :T].transpose(1, 2)
        global_summary = self.ema_conv(parallel_repr.transpose(1, 2))[:, :, :T].transpose(1, 2)

        combined = torch.cat([parallel_repr, local_history, global_summary], dim=-1)
        deltas = self.delta_norm(torch.sigmoid(self.gate_proj(combined)) * torch.tanh(self.delta_proj(combined)))
        
        accumulated_states = torch.cumsum(deltas, dim=1) * self.step_size
        states = self.ln3(sequential_state.unsqueeze(1) + accumulated_states)

        seq_guidance = self.seq_proj(states)
        parallel_repr = self.ln2(parallel_repr + self.ffn_parallel(parallel_repr + 0.3 * seq_guidance))
        final_sequential_state = self.ln4(states[:, -1, :])

        return parallel_repr, final_sequential_state, attn_weights

class LocalExpertModel(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([ImprovedHybridReasoningLayer(d_model) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sequential_state=None):
        B, T = x.size()
        padding_mask = (x == pad_token_id)
        out = self.pos_encoder(self.embedding(x))
        
        if sequential_state is None:
            sequential_state = torch.zeros(B, d_model, device=x.device)

        attn_maps = []
        for layer in self.layers:
            out, sequential_state, attn = layer(out, sequential_state, padding_mask=padding_mask)
            attn_maps.append(attn)
            
        return self.layer_norm(out), sequential_state, attn_maps

# ==========================================
# 3. GLOBAL EXPERT (Strengthened with FFN)
# ==========================================
class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.scale = d_model ** 0.5
        
        # NEW: Feed-Forward Network (FFN) to increase expressivity
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        B, T, D = x.size()
        
        # Work entirely in FP32 for numerical stability
        x_f32 = x.float()
        
        q, k, v = self.qkv(x_f32).chunk(3, dim=-1)
        
        # Scale and normalize
        q = q / self.scale
        k = k / self.scale
        
        # L2 normalize to unit vectors
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Use softplus for non-negative values + epsilon for stability
        q = F.softplus(q) + 1e-6
        k = F.softplus(k) + 1e-6

        # All operations in FP32
        kv = torch.einsum('btd,bte->btde', k, v)
        kv_sum = torch.cumsum(kv, dim=1)
        k_sum = torch.cumsum(k, dim=1)

        num = torch.einsum('btd,btde->bte', q, kv_sum)
        den = torch.einsum('btd,btd->bt', q, k_sum).unsqueeze(-1).clamp(min=1e-6)
        
        # Clamp the numerator to prevent overflow
        num = torch.clamp(num, min=-1e4, max=1e4)
        
        attn_out = num / den
        
        # Residual connection + LayerNorm 1
        x_attn = self.ln1(x_f32 + self.proj(attn_out))
        
        # NEW: FFN Block with Residual connection + LayerNorm 2
        out = self.ln2(x_attn + self.ffn(x_attn))
        
        return out.to(x.dtype)

class GlobalExpertModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.layers = nn.ModuleList([LinearAttentionLayer(d_model) for _ in range(num_layers)])
        
    def forward(self, x):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        return out

# ==========================================
# 4. ATTENTION IMAGE CNN
# ==========================================
class AttentionCNN(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.conv1 = nn.Conv2d(num_layers, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4)) 
        self.fc = nn.Linear(64 * 4 * 4, d_model)

    def forward(self, attn_maps):
        x = torch.stack(attn_maps, dim=1)
        B, L, T, _ = x.size()

        if T < 64:
            pad = 64 - T
            x = F.pad(x, (0, pad, 0, pad))
        elif T > 64:
            x = x[:, :, -64:, -64:]

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(B, -1)
        return self.fc(x)

# ==========================================
# 5. TWO-TIER FUSION MODEL (Fixed & Gated with Anti-Collapse)
# ==========================================
class TwoTierFusionModel(nn.Module):
    def __init__(self, local_layers=3, global_layers=4, min_gate=0.1, gate_reg_weight=0.5):
        super().__init__()
        self.local_expert = LocalExpertModel(num_layers=local_layers)
        self.global_expert = GlobalExpertModel(num_layers=global_layers)
        self.attn_cnn = AttentionCNN(num_layers=local_layers, d_model=d_model)
        
        # Hyperparameters for anti-collapse
        self.min_gate = min_gate
        self.gate_range = 1.0 - (2 * min_gate) # e.g., 0.8 if min_gate is 0.1
        self.gate_reg_weight = gate_reg_weight
        
        # Gating mechanism for competitive routing
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Adjusted fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # Shared decoder for both global-only and fused representations
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.size()
        
        # 1. Global Expert processes the FULL sequence
        global_out = self.global_expert(x)
        
        if T <= 64:
            x_local = x
            global_context = global_out
        else:
            x_local = x[:, -64:]
            # Token-aligned global context instead of collapsing to one vector
            global_context = global_out[:, -64:, :]

        # 2. Local Expert & CNN on the local window
        local_out, _, attn_maps = self.local_expert(x_local)
        attn_img_feature = self.attn_cnn(attn_maps)
        attn_img_expanded = attn_img_feature.unsqueeze(1).expand(-1, x_local.size(1), -1)
        
        # 3. Competitive Gating with Anti-Collapse Limits
        gate_input = torch.cat([local_out, global_context], dim=-1)
        raw_gate = self.gate(gate_input)
        
        # HARD LIMIT: Scale from [0, 1] to [min_gate, 1 - min_gate] (e.g., [0.1, 0.9])
        gate_weights = self.min_gate + (self.gate_range * raw_gate)
        
        # Calculate the percentage of global context being used
        global_usage_pct = (1.0 - gate_weights).mean()
        
        # The model dynamically chooses between local and global representations
        gated_repr = gate_weights * local_out + (1 - gate_weights) * global_context
        
        # Final fusion with the Attention CNN features
        fused = torch.cat([gated_repr, attn_img_expanded], dim=-1)
        final_repr = self.fusion_proj(fused)
        
        # 4. Generate Logits
        logits_fused = self.decoder(final_repr)
        
        if T > 64:
            # Decode the global-only part (everything before the last 64 tokens)
            logits_global = self.decoder(global_out[:, :-64, :])
            # Concatenate to form full sequence logits [B, T, vocab_size]
            logits = torch.cat([logits_global, logits_fused], dim=1)
        else:
            logits = logits_fused

        # 5. Compute Dual Loss + Gate Regularization
        loss = None
        if targets is not None:
            if T <= 64:
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size), 
                    targets.reshape(-1), 
                    ignore_index=pad_token_id
                )
            else:
                # Loss for the local window (fused representation)
                loss_fused = F.cross_entropy(
                    logits_fused.view(-1, vocab_size), 
                    targets[:, -64:].reshape(-1), 
                    ignore_index=pad_token_id
                )
                # Loss for the global window (global expert representation)
                loss_global = F.cross_entropy(
                    logits_global.view(-1, vocab_size), 
                    targets[:, :-64].reshape(-1), 
                    ignore_index=pad_token_id
                )
                # Combine losses so both networks get strong gradient signals
                loss = loss_global + loss_fused
                
            # SOFT LIMIT: Penalty for deviating from a 50/50 split
            # This pushes the mean of the gate towards 0.5 to prevent collapse
            gate_penalty = torch.mean((gate_weights - 0.5) ** 2)
            loss = loss + (self.gate_reg_weight * gate_penalty)

        return logits, loss, global_usage_pct

# ==========================================
# 6. TRAINING SCRIPT
# ==========================================
# Increased global_layers to 4 for better global expressivity
model = TwoTierFusionModel(local_layers=num_layers, global_layers=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scaler = torch.amp.GradScaler(device) if device == 'cuda' else None
checkpoint_path = 'two_tier_fusion_model.pt'

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split, variable_length=True)
            if device == 'cuda':
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    _, loss, _ = model(X, Y)
            else:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate_prediction(model, prompt="Move Up", max_new_tokens=200, temperature=0.3, top_k=50, top_p=0.9, repetition_penalty=1.2):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        logits, _, _ = model(x)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= repetition_penalty
                else:
                    next_token_logits[0, token_id] /= repetition_penalty
                    
        # Top-K filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
            
        # Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
        # Apply softmax and sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, next_token), dim=1)
        
    model.train()
    return tokenizer.decode(x[0].tolist())

# Check for existing checkpoint
if os.path.exists(checkpoint_path):
    user_choice = input(f"Checkpoint '{checkpoint_path}' found. Enter 'c' to continue training, or 'chat' to enter chat mode: ").strip().lower()
    if user_choice == 'chat':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("\n--- Entering Chat Mode ---")
        print("Type 'quit' or 'exit' to stop.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                exit()
            response = generate_prediction(model, prompt=user_input, max_new_tokens=500)
            print(f"Model: {response}\n")
    elif user_choice == 'c':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Checkpoint loaded. Resuming training...")
    else:
        print("Invalid choice. Starting training from scratch...")

print(f"\n{'='*80}")
print(f"TWO-TIER FUSION MODEL (Local + Global + Attn CNN)")
print(f"{'='*80}")
print(f"Parameters:      {sum(p.numel() for p in model.parameters()):,}")
print(f"Device:          {device}")
print(f"{'='*80}\n")

train_start = time.time()

# Trackers for global usage reporting
running_global_usage = 0.0
usage_count = 0

for iter_num in range(max_iters + 1):
    current_lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    if iter_num > 0 and iter_num % eval_interval == 0:
        eval_start = time.time()
        losses     = estimate_loss(model)
        eval_time  = time.time() - eval_start
        total_time = time.time() - train_start
        
        avg_global_usage = (running_global_usage / max(1, usage_count)) * 100
        avg_local_usage = 100.0 - avg_global_usage

        print(
            f"[Iter {iter_num:5d}] LR: {current_lr:.2e} | "
            f"Loss: Train {losses['train']:.4f}, Val {losses['val']:.4f} | "
            f"Eval: {eval_time:.2f}s | Total: {total_time:.2f}s\n"
            f"   -> Fusion Stats: The gating mechanism is currently utilizing {avg_global_usage:.2f}% Global Context "
            f"and {avg_local_usage:.2f}% Local Context for the final 64 tokens."
        )
        
        # Reset trackers
        running_global_usage = 0.0
        usage_count = 0

    # Every 1000 iterations, print a prediction
    if iter_num % 1000 == 0 and iter_num > 0:
        sample_pred = generate_prediction(model, prompt="BELARIUS. So sure as you your father")
        print(f"--- Prediction Sample @ Iter {iter_num} ---")
        print(f"{sample_pred}")
        print("-------------------------------------------")

    X, Y = get_batch('train', variable_length=True)
    optimizer.zero_grad(set_to_none=True)

    if device == 'cuda':
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            _, loss, global_usage = model(X, Y)
    else:
        _, loss, global_usage = model(X, Y)

    if global_usage is not None:
        running_global_usage += global_usage.item()
        usage_count += 1

    if torch.isnan(loss):
        print(f"\n[Iter {iter_num}] NaN loss detected! Skipping batch...")
        continue

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

print("\nTRAINING COMPLETE")
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved to {checkpoint_path}")