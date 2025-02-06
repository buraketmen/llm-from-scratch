from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tokenizer import Tokenizer

class TextDataset(Dataset):
    def __init__(self, text:str, tokenizer: Tokenizer, block_size: int):
        # Tokenize the entire text
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self):
        # Return the number of possible sequences
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of tokens of length block_size
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # Input sequence
        y = chunk[1:]   # Target sequence
        return x, y
