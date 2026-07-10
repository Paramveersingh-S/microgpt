import os
import argparse
import torch
import tiktoken
from model.gpt import GPT
from config import GPTConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ckpt.pt')
    parser.add_argument('--prompt', type=str, default='\n')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = args.device
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found. Please train the model first.")
        return
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    model = GPT(config)
    
    # Remove _orig_mod. prefix if model was compiled
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"Loaded model from {args.checkpoint} (iter {checkpoint.get('iter_num', 'unknown')})")
    
    enc = tiktoken.get_encoding("gpt2")
    
    if args.prompt == '':
        args.prompt = '\n'
        
    start_ids = enc.encode_ordinary(args.prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    
    print("Generating...")
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, use_cache=True)
            
    print("\n----- RESULT -----\n")
    print(enc.decode(y[0].tolist()))
    print("\n------------------")

if __name__ == '__main__':
    main()
