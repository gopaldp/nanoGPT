#!/usr/bin/env python3
import os
import sys
import time
import math
import torch
import platform
from contextlib import nullcontext

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_project.models.gpt import GPT, GPTConfig
from llm_project.data.data_loader import GPTDataLoader


def get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Learning rate scheduler with warmup and cosine decay"""
    # 1) Linear warmup for warmup_iters steps
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # 2) If iter_num > lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def should_compile_model():
    """Determine if we should compile the model based on system and availability"""
    # Disable compilation on Windows or if compiler not available
    if platform.system() == "Windows":
        return False

    # Check if we can compile
    try:
        # Try a simple compilation test
        def simple_fn(x):
            return x * 2

        torch.compile(simple_fn)
        return True
    except Exception as e:
        print(f"Compilation not available: {e}")
        return False


def train():
    # Training configuration
    out_dir = 'models/checkpoints'
    eval_interval = 250
    log_interval = 10  # Increased for less frequent logging
    eval_iters = 50  # Reduced for faster evaluation
    always_save_checkpoint = True

    # Data configuration
    dataset = 'shakespeare'
    gradient_accumulation_steps = 4  # Reduced for CPU
    batch_size = 8  # Reduced batch size for CPU
    block_size = 256  # Reduced sequence length for faster training

    # Model configuration (smaller model for CPU training)
    n_layer = 6  # Reduced layers
    n_head = 6  # Reduced heads
    n_embd = 384  # Reduced embedding dimension
    dropout = 0.1  # Added some dropout
    bias = False

    # Optimizer configuration
    learning_rate = 1e-3  # Slightly higher LR for smaller model
    max_iters = 2000  # Reduced iterations for demo
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

    # Learning rate decay settings
    decay_lr = True
    warmup_iters = 100  # Reduced warmup
    lr_decay_iters = max_iters
    min_lr = 1e-4

    # System configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    # Disable compilation on Windows or if not available
    compile_model = should_compile_model()

    print(f"Training on {device} with dtype {dtype}")
    print(f"Model compilation: {'enabled' if compile_model else 'disabled'}")

    # Create directories
    os.makedirs(out_dir, exist_ok=True)

    # Download Shakespeare data if it doesn't exist
    data_dir = 'data/raw'
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, 'shakespeare.txt')

    if not os.path.exists(data_file):
        print("Downloading Shakespeare dataset...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_file)
        print(f"Dataset saved to {data_file}")

    # Initialize data loader
    data_loader = GPTDataLoader(data_file, block_size, batch_size, torch.device(device))

    # Model configuration
    config = GPTConfig(
        block_size=block_size,
        vocab_size=data_loader.get_vocab_size(),
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )

    # Initialize model
    model = GPT(config)
    model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")

    # Compile model only if available and not on Windows
    if compile_model:
        print("Compiling model... (takes ~1 minute)")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Compilation failed, continuing without: {e}")

    # Initialize optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

    # Training context
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=torch.float16)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16' and device != 'cpu'))

    # Training loop
    model.train()
    iter_num = 0
    best_val_loss = 1e9
    running_mfu = -1.0

    print("Starting training...")
    t0 = time.time()

    # Create iterator once to avoid recreating DataLoader
    data_iter = iter(data_loader.dataloader)

    while True:
        # Determine learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate the loss on train/val sets
        if iter_num % eval_interval == 0 or iter_num == 0:
            print(f"Evaluating at step {iter_num}...")
            losses = estimate_loss(model, data_loader, eval_iters, ctx, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                    print(f"Checkpoint saved at step {iter_num}")

        # Forward backward update with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                # Reset iterator when we reach the end
                data_iter = iter(data_loader.dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            with ctx:
                logits, loss = model(x, y)
                loss = loss / gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

        # Clip gradients
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % log_interval == 0 and iter_num > 0:
            lossf = loss.item() * gradient_accumulation_steps
            if device == 'cuda':
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt) if hasattr(model,
                                                                                                  'estimate_mfu') else -1
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            else:
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}")

        iter_num += 1

        # Termination conditions
        if iter_num > max_iters:
            break

    print("Training completed!")

    # Generate a sample to test the model
    print("\nGenerating sample text...")
    generate_sample(model, data_loader.tokenizer, device, config.block_size)


@torch.no_grad()
def estimate_loss(model, data_loader, eval_iters, ctx, device):
    """Estimate loss on train and validation splits"""
    out = {}
    model.eval()

    # Since we don't have a separate validation set, we'll use the same data
    # In a real scenario, you'd have separate train/val datasets
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data_iter = iter(data_loader.dataloader)

        for k in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader.dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            with ctx:
                logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


@torch.no_grad()
def generate_sample(model, tokenizer, device, block_size, max_new_tokens=100):
    """Generate a sample text to test the model"""
    model.eval()

    # Start with a simple prompt
    prompt = "ROMEO:"
    start_ids = tokenizer.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    generated = model.generate(x, max_new_tokens, temperature=0.8, top_k=200)
    generated_text = tokenizer.decode(generated[0].tolist())

    print(f"Generated text:\n{'-' * 50}")
    print(generated_text)
    print('-' * 50)

    model.train()


if __name__ == '__main__':
    train()
