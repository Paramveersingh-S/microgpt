from dataclasses import dataclass
from typing import Literal

@dataclass
class GPTConfig:
    vocab_size: int = 50304  # Default for GPT-2 vocab, often padded to multiple of 64
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6  # For GQA, default to n_head for standard MHA
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pos_encoding: Literal["rope", "learned", "sinusoidal"] = "rope"
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    mlp_type: Literal["swiglu", "gelu"] = "swiglu"
    tie_weights: bool = True

@dataclass
class TrainConfig:
    dataset: str = "tinystories"
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 20
    learning_rate: float = 5e-4
    warmup_iters: int = 100
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    compile: bool = False
    device: str = "cuda" # cuda, cpu, mps
