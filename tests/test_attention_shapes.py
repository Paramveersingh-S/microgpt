import torch
from config import GPTConfig
from model.attention import CausalSelfAttention
from model.positional import precompute_rope_freqs

def test_attention_shapes():
    # Test standard MHA
    config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4, block_size=32)
    attn = CausalSelfAttention(config)
    
    B, T, C = 2, 16, 128
    x = torch.randn(B, T, C)
    
    # Needs RoPE cos/sin if rope is used
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rope_freqs(head_dim, config.block_size)
    cos = cos[:T]
    sin = sin[:T]
    
    out = attn(x, cos, sin)
    assert out.shape == (B, T, C), f"Expected {(B, T, C)}, got {out.shape}"
    
    # Test GQA
    config_gqa = GPTConfig(n_embd=128, n_head=4, n_kv_head=2, block_size=32)
    attn_gqa = CausalSelfAttention(config_gqa)
    out_gqa = attn_gqa(x, cos, sin)
    assert out_gqa.shape == (B, T, C), f"Expected {(B, T, C)}, got {out_gqa.shape}"
    
    print("test_attention_shapes passed!")
