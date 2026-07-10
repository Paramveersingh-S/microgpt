import torch
import torch.nn as nn
import math

def precompute_rope_freqs(head_dim: int, seq_len: int, base: float = 10000.0, device='cpu'):
    # theta_i = base ** (-2i/d) for i in [0, d/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq) # (seq_len, head_dim/2)
    # repeat each frequency twice to match the interleaved layout: [x0, x1, x2, x3, ...]
    freqs = torch.repeat_interleave(freqs, 2, dim=-1)
    
    # Store cos and sin for the specified sequence length
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def rotate_half(x):
    # Rotates half the hidden dims of the input.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # Interleave to get [-x2, x1, -x4, x3, ...]
    res = torch.empty_like(x)
    res[..., 0::2] = -x2
    res[..., 1::2] = x1
    return res

def apply_rope(xq, xk, cos, sin):
    # xq, xk shape: (B, num_heads, T, head_dim)
    # cos, sin shape: (T, head_dim)
    
    # Reshape cos and sin for broadcasting over Batch and Heads
    # They come in as (T, head_dim), we need (1, 1, T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Cast cos/sin to the type of xq/xk (e.g. bfloat16) to avoid type mismatch
    cos = cos.to(xq.dtype)
    sin = sin.to(xq.dtype)
    
    # Apply rotation
    xq_rot = (xq * cos) + (rotate_half(xq) * sin)
    xk_rot = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_rot, xk_rot

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.pe = nn.Embedding(block_size, n_embd)
        
    def forward(self, seq_len: int, device: torch.device):
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        return self.pe(positions)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as a buffer so it's not trainable
        
    def forward(self, seq_len: int, device: torch.device):
        return self.pe[:seq_len, :].to(device)
