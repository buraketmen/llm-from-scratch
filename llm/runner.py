from __future__ import annotations
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from typing import TYPE_CHECKING
from tokenizer import Tokenizer
if TYPE_CHECKING:
    from .model import LLM
    from .dataset import TextDataset
    from .config import ModelConfig

class Runner:
    def __init__(self, model: LLM, dataset: TextDataset, config: ModelConfig):
        self.model: LLM = model
        self.dataset: TextDataset = dataset
        self.config: ModelConfig = config
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
    def train(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        self.model.train()
        total_batches = len(dataloader)
        for epoch in range(self.config.max_epochs):
            with tqdm(total=total_batches, desc=f"Epoch {epoch}") as pbar:
                for (x, y) in dataloader:
                    # Move batch to device
                    x, y = x.to(self.config.device), y.to(self.config.device)
                    
                    # Forward pass
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
    def generate(self, prompt, max_tokens=128):
        self.model.eval()
        # No need to self.dataset?
        tokenizer = Tokenizer()
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get predictions
                logits = self.model(tokens[:, -self.config.block_size:])
                logits = logits[:, -1, :] # Get predictions for last token
                probs = F.softmax(logits, dim=-1) # Calculate probabilities
                
                next_token = torch.multinomial(probs, num_samples=1) # Sample from distribution
                tokens = torch.cat([tokens, next_token], dim=1) # Add to sequence
                
                # If end of text token is generated, stop
                if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
                    break
        
        return tokenizer.decode(tokens.squeeze(0).tolist())
