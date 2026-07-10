import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig
from .positional import apply_rope

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.pos_encoding = config.pos_encoding
        
        # QKV projection: 
        # For GQA, Q needs n_head * head_dim
        # K and V each need n_kv_head * head_dim
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout
        
        # Causal mask for the naive attention path
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )
        
        # KV Cache for inference
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, cos=None, sin=None, use_cache=False, slow_attn=False):
        B, T, C = x.size()
        
        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if specified
        if self.pos_encoding == "rope" and cos is not None and sin is not None:
            # During inference with cache, cos and sin must be sliced externally 
            # to match the sequence position being computed.
            q, k = apply_rope(q, k, cos, sin)
            
        # Update KV cache during inference
        if use_cache:
            if self.cache_k is not None and self.cache_v is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k = k
            self.cache_v = v
            
        # Grouped Query Attention: repeat KV heads to match Q heads
        if self.n_kv_head < self.n_head:
            num_repeat = self.n_head // self.n_kv_head
            # repeat_interleave is standard for GQA
            k = torch.repeat_interleave(k, repeats=num_repeat, dim=1)
            v = torch.repeat_interleave(v, repeats=num_repeat, dim=1)
            
        # Attention
        if slow_attn:
            # Naive/Manual Attention Path
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply mask only for training or if T > 1. 
            # If using cache generating token by token, T=1 and we attend to all past tokens without masking.
            if T > 1 and not use_cache:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout > 0.0:
                att = F.dropout(att, p=self.dropout)
            y = att @ v
        else:
            # Flash Attention path (SDPA)
            is_causal = (T > 1) and not use_cache
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
            
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        if self.training and self.dropout > 0.0:
            y = F.dropout(y, p=self.dropout)
            
        return y
