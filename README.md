# microGPT 

<div align="center">
  <img src="logo.png" alt="microGPT Logo" width="200" />
</div>

<p align="center">
  <b>A modern, from-scratch, decoder-only GPT implementation in PyTorch.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white" />
</p>

## Overview
microGPT is a compact, high-performance autoregressive language model built from first principles. Unlike vanilla 2017 Transformer clones, microGPT implements the modern (2025/2026-era) architectural decisions found in state-of-the-art open-weight models like Llama 3, Mistral, and Gemma. 

## Modern Architecture Highlights

| Feature | Implementation Detail | Why it matters |
|---------|------------------------|----------------|
| **Positional Encoding** | RoPE (Rotary Positional Embeddings) | Encodes relative position directly into the attention dot product, generalizing better to longer sequences. |
| **Normalization** | RMSNorm (Pre-Norm) | Computationally cheaper than LayerNorm while maintaining training stability at depth. |
| **Feedforward** | SwiGLU | Replaces standard GELU for better representation capacity (used in PaLM, Llama). |
| **Attention** | Grouped Query Attention (GQA) | Reduces KV cache memory footprint drastically during inference with negligible quality loss. |
| **Kernel Dispatch** | `F.scaled_dot_product_attention` | Automatically routes to Flash Attention v2 on supported hardware for optimal memory and throughput. |
| **Inference** | KV Caching | Prevents recomputation of past tokens during autoregressive generation. |

## Architecture Diagram

```mermaid
graph TD
    classDef input fill:#2563eb,stroke:#1e40af,stroke-width:2px,color:#fff,rx:8px,ry:8px;
    classDef norm fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff,rx:5px,ry:5px;
    classDef attn fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff,rx:10px,ry:10px;
    classDef mlp fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff,rx:10px,ry:10px;
    classDef merge fill:#475569,stroke:#334155,stroke-width:2px,color:#fff,shape:circle;
    classDef output fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff,rx:8px,ry:8px;

    A([Input Tokens]):::input --> B[Token Embedding]:::input
    B --> C{N x Transformer Blocks}
    
    subgraph Transformer Block [Transformer Block Architecture]
        direction TB
        C1[RMSNorm]:::norm --> C2[Causal Self-Attention]:::attn
        C2 --> C3[Grouped Query Attention GQA]:::attn
        C3 --> C4[RoPE Injection]:::attn
        C4 --> C5((+)):::merge
        
        C5 --> C6[RMSNorm]:::norm
        C6 --> C7[SwiGLU MLP]:::mlp
        C7 --> C8((+)):::merge
    end
    
    C --> C1
    
    %% Residual Connections (Using invisible nodes/direct paths)
    C -- "Residual Stream" ---> C5
    C5 -- "Residual Stream" ---> C8

    C8 --> D[Final RMSNorm]:::norm
    D --> E[Linear LM Head]:::output
    E --> F([Next Token Probabilities]):::output
    
    style Transformer Block fill:#f8fafc,stroke:#94a3b8,stroke-width:2px,stroke-dasharray: 5 5,rx:10px,ry:10px;
```

## Results

### Training Loss Curve
<div align="center">
  <img src="loss_curve.png" alt="Training Loss Curve" width="600">
</div>

### Sample Generations

```text
[Prompt]: Once upon a time
[Completion]: ...
```

## Quickstart (Colab / Kaggle)

You can clone and train microGPT directly in a single cell on a free-tier GPU (T4). 
*Note for Kaggle users: Ensure your notebook settings have GPU T4 x2 enabled and clone into a relative working path.*

```python
!git clone https://github.com/Paramveersingh-S/microgpt.git
%cd microgpt
!pip install -q -r requirements.txt

# Prepare dataset (TinyStories subset)
!python data/prepare_data.py --dataset tinystories

# Train the model (compilation is optional but recommended)
!python train.py \
    --dataset tinystories \
    --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 \
    --batch_size 64 --grad_accum_steps 1 \
    --max_iters 3000 --eval_interval 250 \
    --pos_encoding rope --norm_type rmsnorm --mlp_type swiglu \
    --compile

# Sample from the trained model
!python sample.py \
    --checkpoint checkpoints/ckpt.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 50
```

## Scaling Up
To scale microGPT for production or large-scale training:
1. **Multi-GPU (DDP/FSDP)**: Transition the training loop from single-device to DistributedDataParallel or FullyShardedDataParallel to handle larger batch sizes and parameter counts.
2. **Dataset Size**: Swap the TinyStories subset for a larger corpus like FineWeb or RedPajama.
3. **Advanced GQA/RoPE**: At larger scales, GQA becomes critical for serving memory constraints, and RoPE base frequencies should be increased (e.g., to 500,000) for extremely long context windows.
