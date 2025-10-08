#!/usr/bin/env python3
import os
import sys
import torch
import argparse
from contextlib import nullcontext

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_project.models.gpt import GPT, GPTConfig
import tiktoken


def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']

    # Create model
    model = GPT(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, model_args


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=200, device='cpu'):
    """Generate text from the model"""
    model.eval()

    # Encode the prompt
    start_ids = tokenizer.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate tokens
    print(f"Generating text with prompt: '{prompt}'")
    print("=" * 50)

    with torch.no_grad():
        generated = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    # Decode back to text
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text


def interactive_mode(model, tokenizer, device):
    """Interactive text generation"""
    print("\n🤖 Interactive Mode - Your GPT is Ready!")
    print("Type your prompts below. Commands:")
    print("  /quit - Exit")
    print("  /temp <value> - Set temperature (0.1-2.0)")
    print("  /length <value> - Set max tokens (10-500)")
    print("  /help - Show this help")
    print("=" * 50)

    temperature = 0.8
    max_length = 100

    while True:
        try:
            prompt = input("\n💭 You: ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.startswith('/'):
                if prompt == '/quit':
                    print("Goodbye! 👋")
                    break
                elif prompt.startswith('/temp'):
                    try:
                        temperature = float(prompt.split()[1])
                        temperature = max(0.1, min(2.0, temperature))
                        print(f"Temperature set to {temperature}")
                    except:
                        print("Usage: /temp <value> (e.g., /temp 0.8)")
                elif prompt.startswith('/length'):
                    try:
                        max_length = int(prompt.split()[1])
                        max_length = max(10, min(500, max_length))
                        print(f"Max length set to {max_length}")
                    except:
                        print("Usage: /length <value> (e.g., /length 150)")
                elif prompt == '/help':
                    print("Commands:")
                    print("  /quit - Exit")
                    print("  /temp <value> - Set temperature (0.1-2.0)")
                    print("  /length <value> - Set max tokens (10-500)")
                continue

            # Generate response
            generated_text = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                device=device
            )

            print(f"🤖 GPT: {generated_text}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate text with your trained GPT model')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/ckpt.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k sampling parameter')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return

    model, model_args = load_model(args.checkpoint, device)

    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, device)
    else:
        # Single generation mode
        if args.prompt is None:
            args.prompt = "HAMLET:"  # Default Shakespeare prompt

        generated_text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        print(f"\nGenerated text:\n{generated_text}")


if __name__ == '__main__':
    main()
