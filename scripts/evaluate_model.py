#!/usr/bin/env python3
import os
import sys
import torch
import math
import time
from contextlib import nullcontext

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_project.models.gpt import GPT
from llm_project.data.data_loader import GPTDataLoader
import tiktoken


def load_model(checkpoint_path, device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']

    model = GPT(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, model_args


@torch.no_grad()
def calculate_perplexity(model, data_loader, device, num_batches=100):
    """Calculate perplexity - a key LLM evaluation metric"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    print("Calculating perplexity...")
    data_iter = iter(data_loader.dataloader)

    for i in range(min(num_batches, len(data_loader.dataloader))):
        try:
            x, y = next(data_iter)
        except StopIteration:
            break

        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, loss = model(x, y)

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        if i % 20 == 0:
            current_ppl = math.exp(total_loss / total_tokens)
            print(f"Batch {i}/{num_batches}: Current perplexity = {current_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def generation_quality_test(model, tokenizer, device):
    """Test generation quality with various prompts"""
    test_prompts = [
        "ROMEO:",
        "JULIET:",
        "HAMLET:",
        "To be or not to be,",
        "Once upon a time",
        "The king said"
    ]

    print("\n" + "=" * 60)
    print("GENERATION QUALITY TEST")
    print("=" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)

        # Generate with different temperatures
        for temp in [0.3, 0.8, 1.2]:
            start_ids = tokenizer.encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

            generated = model.generate(x, max_new_tokens=50, temperature=temp, top_k=100)
            generated_text = tokenizer.decode(generated[0].tolist())

            print(f"Temperature {temp}: {generated_text}")


def benchmark_speed(model, tokenizer, device):
    """Benchmark generation speed"""
    print("\n" + "=" * 60)
    print("SPEED BENCHMARK")
    print("=" * 60)

    prompt = "HAMLET: To be or not to be,"
    start_ids = tokenizer.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Warmup
    for _ in range(3):
        model.generate(x, max_new_tokens=10, temperature=0.8)

    # Benchmark different sequence lengths
    for seq_len in [50, 100, 200]:
        start_time = time.time()
        generated = model.generate(x, max_new_tokens=seq_len, temperature=0.8)
        end_time = time.time()

        tokens_per_second = seq_len / (end_time - start_time)
        print(f"Generated {seq_len} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load model
    checkpoint_path = 'models/checkpoints/ckpt.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model, model_args = load_model(checkpoint_path, device)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load data for perplexity calculation
    data_file = 'data/raw/shakespeare.txt'
    if os.path.exists(data_file):
        data_loader = GPTDataLoader(data_file, model_args.block_size, 8, device)

        # Calculate perplexity
        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)

        perplexity, avg_loss = calculate_perplexity(model, data_loader, device, num_batches=50)
        print(f"\nFinal Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")

        # Interpretation
        if perplexity < 20:
            print("🟢 Excellent! Very low perplexity - model learned well")
        elif perplexity < 50:
            print("🟡 Good perplexity - decent learning")
        else:
            print("🔴 High perplexity - model needs more training")

    # Generation quality test
    generation_quality_test(model, tokenizer, device)

    # Speed benchmark
    benchmark_speed(model, tokenizer, device)

    print(
        f"\n🎉 Evaluation complete! Your model with {sum(p.numel() for p in model.parameters()):,} parameters is ready to use.")


if __name__ == '__main__':
    main()
