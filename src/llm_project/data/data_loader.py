import torch
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Tokenize the entire text
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Dataset has {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of tokens
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # input
        y = chunk[1:]  # targets (shifted by 1)
        return x, y


class GPTDataLoader:
    def __init__(self, text_file_path, block_size, batch_size, device):
        self.device = device
        self.block_size = block_size
        self.batch_size = batch_size

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Load and tokenize text
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.dataset = TextDataset(text, self.tokenizer, block_size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )

    def get_vocab_size(self):
        return self.tokenizer.n_vocab

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
