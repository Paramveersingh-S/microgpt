import torch
from torch.optim import AdamW
from config import GPTConfig
from model.gpt import GPT

def test_overfit_single_batch():
    config = GPTConfig(vocab_size=100, n_embd=64, n_head=2, n_layer=2, block_size=16, dropout=0.0)
    model = GPT(config)
    
    idx = torch.randint(0, 100, (4, 16))
    target = torch.randint(0, 100, (4, 16))
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Train for a few steps on CPU
    for step in range(200):
        optimizer.zero_grad()
        logits, loss = model(idx, target)
        loss.backward()
        optimizer.step()
        
    assert loss.item() < 0.1, f"Model didn't overfit, final loss: {loss.item()}"
    print(f"test_overfit_single_batch passed! Final loss: {loss.item()}")
