from __future__ import annotations
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Ensure embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0
        
        # Initialize key, query, value projections and output projection
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # Split heads and reshape
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention layer
        self.attention = MultiHeadAttention(config)
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.attention(self.ln1(x))
        # Apply MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embedding layer
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        self.config = config

    def forward(self, idx):
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        
        # Combine embeddings
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply final layer norm and language modeling head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
