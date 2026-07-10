import os
import time
import math
import argparse
import torch
from config import GPTConfig, TrainConfig
from model.gpt import GPT
from utils.data_loader import DataLoader
from utils.lr_schedule import get_lr
from utils.logging_utils import Logger

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # 2D+ params will be weight decayed, 1D params (biases, layernorms/rmsnorms) will not
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Use Fused AdamW if on CUDA
    use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    
    return optimizer

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, ctx):
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tinystories')
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--eval_interval', type=int, default=250)
    parser.add_argument('--pos_encoding', type=str, default='rope')
    parser.add_argument('--norm_type', type=str, default='rmsnorm')
    parser.add_argument('--mlp_type', type=str, default='swiglu')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    
    data_dir = os.path.join('data', args.dataset)
    import json
    meta_path = os.path.join(data_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        vocab_size = meta['vocab_size']
    else:
        # Fallback to GPT2 default
        vocab_size = 50304
    
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_head, # Defaulting to standard MHA
        n_embd=args.n_embd,
        pos_encoding=args.pos_encoding,
        norm_type=args.norm_type,
        mlp_type=args.mlp_type,
    )
    
    t_config = TrainConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        compile=args.compile
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.amp.autocast(device_type=device_type, enabled=False)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16)) if device_type == 'cuda' else None

    try:
        train_loader = DataLoader(data_dir, 'train', config.block_size, t_config.batch_size, device)
        val_loader = DataLoader(data_dir, 'val', config.block_size, t_config.batch_size, device)
    except Exception as e:
        print(f"Error loading data (Did you run prepare_data.py?): {e}")
        return
    
    model = GPT(config)
    model.to(device)
    
    if t_config.compile:
        print("Compiling model (this takes a minute)...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed, continuing uncompiled. Error: {e}")
            
    optimizer = configure_optimizers(model, t_config.weight_decay, t_config.learning_rate, (t_config.beta1, t_config.beta2), device_type)
    
    out_dir = 'checkpoints'
    logger = Logger(out_dir)
    
    if args.wandb:
        import wandb
        wandb.init(project="microgpt", config=vars(args))
        
    X, Y = train_loader.get_batch()
    best_val_loss = float('inf')
    t0 = time.time()
    
    for iter_num in range(t_config.max_iters):
        
        lr = get_lr(iter_num, t_config.warmup_iters, t_config.max_iters, t_config.learning_rate, t_config.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if iter_num % t_config.eval_interval == 0 or iter_num == t_config.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, t_config.eval_iters, ctx)
            val_loss = losses['val']
            metrics = {
                'train_loss': losses['train'],
                'val_loss': val_loss,
                'perplexity': math.exp(val_loss) if val_loss < 20 else float('inf'),
                'lr': lr
            }
            logger.log(metrics, step=iter_num)
            if args.wandb:
                wandb.log(metrics, step=iter_num)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                
        for micro_step in range(t_config.grad_accum_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / t_config.grad_accum_steps
                
            X, Y = train_loader.get_batch()
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_config.grad_clip)
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)
        
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % 10 == 0:
            tokens_per_sec = (t_config.batch_size * t_config.grad_accum_steps * config.block_size) / dt
            print(f"iter {iter_num} | loss {loss.item() * t_config.grad_accum_steps:.4f} | dt {dt*1000:.2f}ms | tok/s {tokens_per_sec:.2f}")

if __name__ == '__main__':
    main()
