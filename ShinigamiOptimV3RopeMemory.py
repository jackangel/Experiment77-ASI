import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import time
from transformers import GPT2TokenizerFast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable CUDNN Benchmarking for optimized convolution algorithms
torch.backends.cudnn.benchmark = True

# ==========================================
# 1. HYPERPARAMETERS & CONFIG
# ==========================================
file_path      = 'input.txt'
batch_size     = 16
block_size     = 128       # The hardware/architecture limit
mem_size       = 64        # Tokens reserved for Long-Term Memory (RMT)
text_size      = block_size - mem_size # 112 tokens of actual text per step
bptt_steps     = 37         # Fast unrolling steps for training (3 * 112 = 336 token gradient window)

d_model        = 512
num_layers     = 3
max_iters      = 3000
eval_interval  = 1000
learning_rate  = 1e-4     
warmup_iters   = 400      
device         = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. DATASET PREPARATION (GPU Optimized)
# ==========================================
if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        text = ("Logic: A implies B. B implies C. Therefore A implies C. ") * 2000
        f.write(text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)
vocab_size = tokenizer.vocab_size

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def create_streams(d, bsz):
    stream_len = len(d) // bsz
    d = d[:bsz * stream_len]
    return d.view(bsz, stream_len)

# Move entire streams to GPU immediately to avoid PCIe transfer overhead
train_streams = create_streams(train_data, batch_size).to(device)
val_streams   = create_streams(val_data, batch_size).to(device)

train_ptr = 0
val_ptr   = 0

def get_batch(split):
    global train_ptr, val_ptr
    streams = train_streams if split == 'train' else val_streams
    ptr     = train_ptr if split == 'train' else val_ptr
    
    seq_len = min(text_size, streams.size(1) - 1)

    reset_state = False
    if ptr + seq_len >= streams.size(1):
        ptr = 0
        reset_state = True
        
    x = streams[:, ptr : ptr + seq_len]
    y = streams[:, ptr + 1 : ptr + seq_len + 1]
    
    if split == 'train':
        train_ptr = ptr + seq_len
    else:
        val_ptr = ptr + seq_len

    if x.size(1) < text_size:
        pad_len = text_size - x.size(1)
        x = F.pad(x, (pad_len, 0), value=pad_token_id)
        y = F.pad(y, (pad_len, 0), value=pad_token_id)

    return x, y, reset_state

def get_lr(iter_num: int) -> float:
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters
    progress  = (iter_num - warmup_iters) / max(1, max_iters - warmup_iters)
    cos_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr    = learning_rate * 0.1
    return min_lr + (learning_rate - min_lr) * cos_coeff

# ==========================================
# 3. ROTARY POSITIONAL EMBEDDINGS (Cached)
# ==========================================
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(block_size, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cache', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cache', emb.sin()[None, None, :, :])
        
        causal_mask = torch.ones(block_size, block_size, dtype=torch.bool).tril()
        self.register_buffer('causal_mask', causal_mask.view(1, 1, block_size, block_size))

    def forward(self, x, padding_mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos = self.cos_cache[:, :, :T, :]
        sin = self.sin_cache[:, :, :T, :]

        def rotate_half(tensor):
            x1, x2 = tensor.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        if padding_mask is not None:
            valid_mask = ~padding_mask.view(B, 1, 1, T)
            attn_mask = self.causal_mask[:, :, :T, :T] & valid_mask
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)

# ==========================================
# 4. THE REASONING LAYER
# ==========================================
class ImprovedHybridReasoningLayer(nn.Module):
    def __init__(self, d_model, window_size=8, summary_decay=0.9):
        super().__init__()
        self.d_model = d_model
        self.self_attn = RoPEMultiheadAttention(d_model, num_heads=4)
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

        self.ema_kernel_size = min(block_size, 256)
        self.ema_conv = nn.Conv1d(d_model, d_model, self.ema_kernel_size, padding=self.ema_kernel_size - 1, groups=d_model, bias=False)
        with torch.no_grad():
            weights = torch.zeros(d_model, 1, self.ema_kernel_size)
            for i in range(self.ema_kernel_size):
                weights[:, 0, i] = (1 - summary_decay) * (summary_decay ** i)
            self.ema_conv.weight.copy_(weights.flip(-1))
            self.ema_conv.weight.requires_grad = False

        self.local_conv = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model, bias=True)
        self.delta_proj = nn.Linear(d_model * 3, d_model)
        self.gate_proj  = nn.Linear(d_model * 3, d_model)
        self.delta_norm = nn.LayerNorm(d_model)
        self.step_size  = 0.5 / math.sqrt(block_size / 32)
        self.seq_proj   = nn.Linear(d_model, d_model)
        self.ln1, self.ln2, self.ln3, self.ln4 =[nn.LayerNorm(d_model) for _ in range(4)]

    def forward(self, parallel_repr, sequential_state, padding_mask=None):
        B, T, D = parallel_repr.size()

        attn_out = self.self_attn(parallel_repr, padding_mask=padding_mask)
        parallel_repr = self.ln1(parallel_repr + attn_out)

        parallel_repr_t = parallel_repr.transpose(1, 2)
        local_history  = self.local_conv(parallel_repr_t)[:, :, :T].transpose(1, 2)
        global_summary = self.ema_conv(parallel_repr_t)[:, :, :T].transpose(1, 2)
        
        combined = torch.cat([parallel_repr, local_history, global_summary], dim=-1)

        deltas = torch.sigmoid(self.gate_proj(combined)) * torch.tanh(self.delta_proj(combined))
        deltas = self.delta_norm(deltas)

        accumulated_states = torch.cumsum(deltas, dim=1) * self.step_size
        states = self.ln3(sequential_state.unsqueeze(1) + accumulated_states)

        seq_guidance  = self.seq_proj(states)
        ffn_out       = self.ffn_parallel(parallel_repr + 0.3 * seq_guidance)
        parallel_repr = self.ln2(parallel_repr + ffn_out)

        return parallel_repr, self.ln4(states[:, -1, :])

# ==========================================
# 5. RMT WRAPPER MODEL
# ==========================================
class ImprovedHybridReasoningModel(nn.Module):
    def __init__(self, num_layers=3, mem_size=16):
        super().__init__()
        self.mem_size = mem_size
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_drop = nn.Dropout(0.1)

        self.initial_memory = nn.Parameter(torch.randn(1, mem_size, d_model) * 0.02)

        self.layers = nn.ModuleList([
            ImprovedHybridReasoningLayer(d_model) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder    = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None, sequential_state=None, memory_state=None):
        B, T = x.size()
        
        if memory_state is None:
            memory_state = self.initial_memory.expand(B, -1, -1)
        
        x_embed = self.embedding(x)
        full_seq = torch.cat([memory_state, x_embed], dim=1)
        full_seq = self.embed_drop(full_seq)
        
        mem_mask = torch.zeros(B, self.mem_size, device=x.device, dtype=torch.bool)
        text_mask = (x == pad_token_id)
        full_padding_mask = torch.cat([mem_mask, text_mask], dim=1)

        if sequential_state is None:
            sequential_state = torch.zeros(B, d_model, device=x.device)

        curr_repr = full_seq
        for layer in self.layers:
            curr_repr, sequential_state = layer(curr_repr, sequential_state, padding_mask=full_padding_mask)

        new_memory_state = curr_repr[:, :self.mem_size, :]
        logits_repr = curr_repr[:, self.mem_size:, :]
        logits = self.decoder(self.layer_norm(logits_repr))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_token_id)

        return logits, loss, sequential_state, new_memory_state

# ==========================================
# 6. EVALUATION & GENERATION UTILITIES
# ==========================================
@torch.no_grad()
def generate_text(model, start_text, max_new_tokens=200):
    model.eval()
    encoded = tokenizer.encode(start_text)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    s_s, m_s = None, None
    
    for _ in range(max_new_tokens):
        idx_cond = x[:, -text_size:]
        logits, _, s_s, m_s = model(idx_cond, sequential_state=s_s, memory_state=m_s)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        
    model.train()
    return tokenizer.decode(x[0].tolist())

@torch.no_grad()
def estimate_loss(model):
    global train_ptr, val_ptr
    out = {}
    model.eval()
    saved_train_ptr = train_ptr
    saved_val_ptr = val_ptr
    
    for split in ['train', 'val']:
        losses =[]
        s_s, m_s = None, None
        
        if split == 'train': train_ptr = 0 
        else: val_ptr = 0
            
        for _ in range(20):
            X, Y, reset = get_batch(split)
            if reset: s_s, m_s = None, None
            _, loss, s_s, m_s = model(X, targets=Y, sequential_state=s_s, memory_state=m_s)
            losses.append(loss.item())
            
        out[split] = sum(losses) / len(losses)
        
    train_ptr = saved_train_ptr
    val_ptr = saved_val_ptr
    model.train()
    return out

import os
import sys
import time
import torch

# ==========================================
# 7. FAST TBPTT TRAINING LOOP & CHECKPOINTING
# ==========================================
model = ImprovedHybridReasoningModel(num_layers=num_layers, mem_size=mem_size).to(device)
checkpoint_path = 'optimized_rmt_fast_model.pt'

# Checkpoint handling and User Prompt
start_iter = 0
if os.path.exists(checkpoint_path):
    print(f"\nFound existing checkpoint at '{checkpoint_path}'.")
    user_choice = input("Enter 't' to continue Training, or 'c' to enter Chat mode: ").strip().lower()
    
    # Load the model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Checkpoint loaded successfully.")
    
    if user_choice == 'c':
        print("\n" + "="*40)
        print("ENTERING CHAT MODE (Type 'quit' or 'exit' to stop)")
        print("="*40)
        model.eval()
        while True:
            try:
                user_prompt = input("\nYou: ")
                if user_prompt.lower() in ['quit', 'exit']:
                    print("Exiting chat mode.")
                    sys.exit(0)
                
                # Assuming generate_text takes the model, prompt, and max_new_tokens
                response = generate_text(model, user_prompt, max_new_tokens=200)
                print(f"Model: {response}")
            except KeyboardInterrupt:
                print("\nExiting chat mode.")
                sys.exit(0)
    else:
        print("Continuing training from checkpoint...\n")
        # Note: If you want to resume exactly where you left off, you would also 
        # need to save/load the optimizer state and the current iteration number.

if hasattr(torch, 'compile') and sys.platform != 'win32':
    print("PyTorch 2.0+ detected. Compiling model for faster training...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Compilation skipped due to: {e}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scaler = torch.amp.GradScaler(device) if device == 'cuda' else None

print(f"RMT Model initialized. Physical Block: {block_size} (Mem: {mem_size}, Text: {text_size})")
print(f"Training Speed: Fast unrolling {bptt_steps} chunks per gradient update.")
print(f"Effective Context: Infinite (Memory state never resets until end of document).")

seq_state, mem_state = None, None
train_start = time.time()

model.train() # Ensure model is in training mode
for iter_num in range(start_iter, max_iters + 1):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ------------------------------------------------------------------
    # EVALUATION & TEXT GENERATION BLOCK
    # ------------------------------------------------------------------
    if iter_num % eval_interval == 0:
        eval_start = time.time()
        model.eval() # Switch to eval mode for loss estimation and generation
        losses = estimate_loss(model)
        print(f"\n[Iter {iter_num:5d}] LR: {lr:.2e} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        # See what the model has learned so far!
        sample = generate_text(model, 'Logic:', max_new_tokens=50)
        print(f"Generated: {sample}")
        print(f"Eval Time: {time.time() - eval_start:.2f}s | Total Run: {time.time() - train_start:.2f}s\n" + "-"*60)
        model.train() # Switch back to train mode

    # ------------------------------------------------------------------
    # FAST TRAINING LOOP (3 Steps Backprop)
    # ------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0

    for step in range(bptt_steps):
        X, Y, reset_state = get_batch('train')
        
        if reset_state: 
            seq_state, mem_state = None, None

        if device == 'cuda':
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                logits, loss, seq_state, mem_state = model(X, targets=Y, sequential_state=seq_state, memory_state=mem_state)
                scaled_loss = loss / bptt_steps 
        else:
            logits, loss, seq_state, mem_state = model(X, targets=Y, sequential_state=seq_state, memory_state=mem_state)
            scaled_loss = loss / bptt_steps

        # Accumulate gradients over the small TBPTT window
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        accumulated_loss += loss.item()

        # Detach states so backprop stops here, BUT the vectors 
        # keep moving forward carrying your 4096+ token context!
        if seq_state is not None: seq_state = seq_state.detach()
        if mem_state is not None: mem_state = mem_state.detach()

    # Step the optimizer
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

torch.save(model.state_dict(), checkpoint_path)
print("\nTraining Complete.")