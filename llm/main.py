import torch
import os, argparse
from config import ModelConfig
from tokenizer import Tokenizer
from model import LLM
from dataset import TextDataset
from runner import Runner


DATA_PATH = "data/game_of_thrones.txt"
MODEL_PATH = "models/game_of_thrones.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text using the model")
    parser.add_argument("--prompt", type=str, help="Prompt for text generation")
    return parser.parse_args()

def train():
    # Initialize components
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please download the dataset and place it in the data directory.")
        return
    
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}. Please use another name.")
        return
    
    config = ModelConfig()
    tokenizer = Tokenizer()
    print(f"Initialized tokenizer with {tokenizer.vocab_size} tokens")
    print(f'Initialized with device: {config.device}')
    # Load your text data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text)} characters from the dataset")
    print(f'First 32 characters: {text[:32]}')
    # Create dataset
    dataset = TextDataset(text=text, tokenizer=tokenizer, block_size=config.block_size)
    if len(dataset) == 0:
        print("Dataset is empty. Please check your dataset.")
        return
    print(f"Created dataset of {len(dataset)} sequences")
    
    # Initialize model
    model = LLM(config=config).to(device=config.device)
    print("Model initialized")
    
    # Create runner and train
    runner = Runner(model=model, dataset=dataset, config=config)
    print("Training started...")
    runner.train()

    print("Training completed. Saving the model...")
    torch.save(model.state_dict(), MODEL_PATH)
    # We dont need to save gpt2 tokenizer, it is not changed
    print("Generating new text by using 'Khaleesi saw jon snow and'...")
    prompt = "Khaleesi saw jon snow and"
    generated_text = runner.generate(prompt)
    print(generated_text)

def evaluate(text:str = "Khaleesi saw jon snow and"):
    # Load the model
    
    config = ModelConfig()
    model = LLM(config=config).to(device=config.device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    runner = Runner(model=model, dataset=None, config=config)
    result = runner.generate(text)
    print('Generated text:\n')
    print(result)

if __name__ == "__main__":
    args = parse_args()
    if args.train or not args.generate:
        train()
    elif args.generate:
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Please train the model first.")
        else: 
            if args.prompt:
                evaluate(args.prompt)
            else:
                print("Please provide a prompt for text generation. We will use 'Khaleesi saw jon snow and' as the default prompt.")
                evaluate()
