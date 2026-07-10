import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # x: (..., dim)
        # RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * weight
        # No mean subtraction, no bias, used in Llama/Gemma
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm

class LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        # Native PyTorch LayerNorm
        self.ln = nn.LayerNorm(dim, bias=bias, eps=eps)
        
    def forward(self, x):
        return self.ln(x)

class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # SwiGLU has 3 weight matrices. 
        # Hidden dim is typically ~ (2/3) * 4 * n_embd, to keep param count equal to GELU MLP
        hidden_dim = int(8 * config.n_embd / 3)
        # Sometimes rounded up to nearest multiple of 256 for optimal hardware usage, 
        # but exact ratio is fine for this implementation
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias) # Gate
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias) # Up
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias) # Down
        
    def forward(self, x):
        # SwiGLU(x) = (Swish(x W_gate)) * (x W_up) W_down
        # Swish(x) = x * sigmoid(x) = F.silu(x)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class GELU_MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x) 
        x = self.c_proj(x)
        return x

def get_norm(config: GPTConfig):
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    else:
        return LayerNorm(config.n_embd, bias=config.bias)

def get_mlp(config: GPTConfig):
    if config.mlp_type == "swiglu":
        return SwiGLU(config)
    else:
        return GELU_MLP(config)
