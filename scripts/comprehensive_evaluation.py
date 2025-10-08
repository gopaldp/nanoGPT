import torch
import numpy as np
from collections import Counter
import sys, os

sys.path.append('src')
from llm_project.models.gpt import GPT
from llm_project.data.data_loader import GPTDataLoader
import tiktoken
import math


def calculate_metrics(model, data_loader, device, num_samples=100):
    """Calculate comprehensive evaluation metrics"""
    model.eval()

    # 1. Perplexity
    total_loss = 0
    total_tokens = 0

    for i, (x, y) in enumerate(data_loader.dataloader):
        if i >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    perplexity = math.exp(total_loss / total_tokens)

    # 2. Token diversity (measure repetition)
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "ROMEO:"
    start_ids = tokenizer.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    generated = model.generate(x, max_new_tokens=200, temperature=0.8)
    tokens = generated[0].tolist()

    unique_tokens = len(set(tokens))
    total_tokens_gen = len(tokens)
    diversity_ratio = unique_tokens / total_tokens_gen

    # 3. Repetition score
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    repetition_score = 1 - (len(set(bigrams)) / len(bigrams))

    return {
        'perplexity': perplexity,
        'diversity_ratio': diversity_ratio,
        'repetition_score': repetition_score
    }

# Use this in your evaluation
