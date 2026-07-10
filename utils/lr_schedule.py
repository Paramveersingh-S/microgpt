import math

def get_lr(it: int, warmup_iters: int, max_iters: int, max_lr: float, min_lr: float):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
        
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
        
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)
