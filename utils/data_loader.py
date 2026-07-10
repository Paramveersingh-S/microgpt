import os
import torch
import numpy as np

class DataLoader:
    def __init__(self, data_dir, split, block_size, batch_size, device='cpu'):
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # We memmap the array for fast, memory efficient loading
        bin_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.len = len(self.data)
        
    def get_batch(self):
        # randomly sample starting indices
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        
        # extract sequences
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        # move to device asynchronously
        if 'cuda' in self.device:
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y
