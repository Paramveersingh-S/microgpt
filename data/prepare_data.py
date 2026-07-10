import os
import argparse
import requests
import json
import numpy as np
import tiktoken
from datasets import load_dataset

def prepare_shakespeare(data_dir, enc):
    input_file_path = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading TinyShakespeare from {data_url}...")
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
            
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Length of dataset in characters: {len(data):,}")
    
    print("Tokenizing...")
    ids = enc.encode_ordinary(data)
    print(f"Has {len(ids):,} tokens")
    
    # Split 90/10
    n = len(ids)
    train_data = ids[:int(n*0.9)]
    val_data = ids[int(n*0.9):]
    
    return train_data, val_data, len(ids)

def prepare_tinystories(data_dir, enc):
    print("Downloading/Loading TinyStories dataset...")
    # Load a portion for training
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    print("Tokenizing...")
    # Take a subset (e.g. 100k stories) to keep it manageable in this demo
    num_samples = len(dataset)
    subset = dataset.select(range(min(100000, num_samples))) 
    
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}
    
    tokenized = subset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing stories",
        num_proc=os.cpu_count() or 1
    )
    
    all_ids = []
    for row in tokenized:
        all_ids.extend(row['ids'])
        
    n = len(all_ids)
    print(f"Has {n:,} tokens in the subset")
    
    # Split 99/1
    train_data = all_ids[:int(n*0.99)]
    val_data = all_ids[int(n*0.99):]
    
    return train_data, val_data, n

def main():
    parser = argparse.ArgumentParser(description="Prepare data for microGPT")
    parser.add_argument('--dataset', type=str, default='tinystories', choices=['shakespeare', 'tinystories'])
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    
    # We use tiktoken gpt2 for tokenization
    enc = tiktoken.get_encoding("gpt2")
    
    if args.dataset == 'shakespeare':
        train_data, val_data, total_tokens = prepare_shakespeare(data_dir, enc)
    else:
        train_data, val_data, total_tokens = prepare_tinystories(data_dir, enc)
        
    print(f"Train has {len(train_data):,} tokens")
    print(f"Val has {len(val_data):,} tokens")
    
    # Export to bin files
    train_ids = np.array(train_data, dtype=np.uint16)
    val_ids = np.array(val_data, dtype=np.uint16)
    
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    
    meta = {
        'vocab_size': enc.n_vocab,
        'dataset': args.dataset,
        'total_tokens': total_tokens,
        'tokenizer': 'tiktoken/gpt2'
    }
    with open(os.path.join(data_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"Data preparation for {args.dataset} complete!")

if __name__ == '__main__':
    main()
