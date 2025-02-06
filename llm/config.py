import torch

class ModelConfig:
    # Model architecture parameters
    vocab_size = 50257  # GPT-2 vocabulary size
    n_embd = 768       # Embedding dimension
    n_head = 12        # Number of attention heads
    n_layer = 6        # Number of transformer layers
    dropout = 0     # Dropout rate
    block_size = 128   # Maximum sequence length
    
    # Training parameters
    batch_size = 32
    learning_rate = 3e-4
    max_epochs = 10
    eval_interval = 500
    eval_iters = 200
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
