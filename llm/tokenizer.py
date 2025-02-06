import tiktoken

class Tokenizer:
    def __init__(self, encoding_name:str="gpt2"):
        # Initialize the tokenizer with GPT-2 encoding
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def encode(self, text):
        # Convert text to token ids
        return self.encoding.encode(text, allowed_special={'<|endoftext|>'})
    
    def decode(self, token_ids):
        # Convert token ids back to text
        return self.encoding.decode(token_ids)
    
    @property
    def vocab_size(self):
        # Return the vocabulary size
        return self.encoding.n_vocab
