import torch
from config import GPTConfig
from model.gpt import GPT

def test_causal_mask():
    config = GPTConfig(vocab_size=100, n_embd=64, n_head=2, n_layer=1, block_size=16)
    model = GPT(config)
    model.eval()
    
    idx1 = torch.randint(0, 100, (1, 8))
    idx2 = idx1.clone()
    
    # Change a future token
    idx2[0, 7] = (idx2[0, 7] + 1) % 100
    
    with torch.no_grad():
        logits1, _ = model(idx1, slow_attn=True)  # Use slow attn to explicitly check our mask
        logits2, _ = model(idx2, slow_attn=True)
        
    # The output at position 6 should be identical regardless of position 7
    diff = (logits1[0, 6] - logits2[0, 6]).abs().max()
    assert diff < 1e-5, f"Causality violated: diff {diff}"
    print("test_causal_mask passed!")
