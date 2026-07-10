import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig
from .positional import precompute_rope_freqs, LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
from .layers import get_norm, get_mlp
from .attention import CausalSelfAttention

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Pre-Norm residual structure
        self.norm_1 = get_norm(config)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = get_norm(config)
        self.mlp = get_mlp(config)

    def forward(self, x, cos=None, sin=None, use_cache=False, slow_attn=False):
        x = x + self.attn(self.norm_1(x), cos, sin, use_cache, slow_attn)
        x = x + self.mlp(self.norm_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        if config.pos_encoding == "learned":
            self.pos_embedding = LearnedPositionalEmbedding(config.block_size, config.n_embd)
        elif config.pos_encoding == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEmbedding(config.block_size, config.n_embd)
        else:
            self.pos_embedding = None # RoPE is applied inside attention
            
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = get_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_weights:
            self.token_embedding.weight = self.lm_head.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal init
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        # Scaled init for residual projections to keep variance stable at depth
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
                
    def reset_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()
            
    def forward(self, idx, targets=None, use_cache=False, slow_attn=False):
        device = idx.device
        B, T = idx.size()
        
        x = self.token_embedding(idx)
        
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(T, device)
            
        # Precompute/fetch RoPE frequencies
        cos, sin = None, None
        if self.config.pos_encoding == "rope":
            head_dim = self.config.n_embd // self.config.n_head
            cos, sin = precompute_rope_freqs(head_dim, self.config.block_size, device=device)
            
            if use_cache:
                cache_len = self.blocks[0].attn.cache_k.size(2) if self.blocks[0].attn.cache_k is not None else 0
                cos = cos[cache_len:cache_len+T]
                sin = sin[cache_len:cache_len+T]
            else:
                cos = cos[:T]
                sin = sin[:T]
                
        # Forward pass through Transformer blocks
        for block in self.blocks:
            x = block(x, cos=cos, sin=sin, use_cache=use_cache, slow_attn=slow_attn)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True):
        self.eval()
        if use_cache:
            self.reset_cache()
            
        for _ in range(max_new_tokens):
            if use_cache:
                # Use only the most recent token if we have a cache
                idx_cond = idx[:, -1:]
            else:
                # Crop context if it exceeds block size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                
            logits, _ = self(idx_cond, use_cache=use_cache)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
