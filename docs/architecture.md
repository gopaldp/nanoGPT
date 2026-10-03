# Architecture Documentation

This document describes the modular architecture of `nanoGPT`, detailing its core components, modules, and how data and control flow through the system.

## System Overview

`nanoGPT` is structured as a modular Python package (`src/llm_project/`) adhering to standard deep learning and transformer design patterns. The architecture is split into data loading, transformer model components, training configurations, evaluation scripts, and user interfaces.

```text
src/llm_project/
├── data/
│   └── data_loader.py         # Dataset chunking, tokenization, and PyTorch dataloaders
├── models/
│   ├── attention.py           # Multi-Head Causal Self-Attention
│   ├── transformer_blocks.py  # LayerNorm, Residual Connections, and MLP Decoder Blocks
│   └── gpt.py                 # Core GPT model definition, weight initialization, & generation
└── utils/                     # Utility helpers
```

---

## Core Modules

### 1. Model Components (`src/llm_project/models/`)

- **`attention.py` (`CausalSelfAttention`)**:
  - Implements multi-head masked self-attention.
  - Applies causal masking to prevent tokens from attending to future tokens.
  - Supports scaling, dropout, and efficient matrix operations.

- **`transformer_blocks.py` (`Block`)**:
  - Combines layer normalization (`LayerNorm`), causal self-attention, and a feed-forward multilayer perceptron (MLP) with GELU activation.
  - Implements residual connections around both attention and MLP sub-layers.

- **`gpt.py` (`GPT`, `GPTConfig`)**:
  - **`GPTConfig`**: Data class holding architecture hyperparameters (`vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`).
  - **`GPT`**: The complete language model. It embeds token indices and positional encodings, stacks transformer `Block` layers, applies final layer normalization, and projects logits onto the vocabulary space.
  - **Methods**:
    - `forward(idx, targets=None)`: Computes logits and cross-entropy loss if targets are provided.
    - `generate(idx, max_new_tokens, temperature=1.0, top_k=None)`: Autoregressive token generation using top-k filtering and temperature scaling.
    - `configure_optimizers(weight_decay, learning_rate, betas, device)`: Separates weight decay parameters for AdamW optimization.

### 2. Data Pipeline (`src/llm_project/data/data_loader.py`)

- **`GPTDataLoader`**:
  - Reads raw text files (e.g., Tiny Shakespeare corpus).
  - Encodes text using `tiktoken` (specifically the `gpt2` encoding).
  - Creates sliding-window input sequences (`x`) and target sequences (`y`) offset by 1 token.
  - Wraps data in PyTorch `DataLoader` objects for batched training and evaluation.

### 3. Training & Evaluation (`scripts/`)

- **`scripts/train_model.py`**:
  - Orchestrates the training lifecycle.
  - Implements cosine learning rate decay with linear warmup.
  - Supports mixed-precision training (`bfloat16`/`float16`) and `torch.compile` optimization.
  - Automatically fetches the Tiny Shakespeare dataset (`tinyshakespeare/input.txt`) if missing.
  - Saves model checkpoints (`models/checkpoints/ckpt.pt`).

- **`scripts/evaluate_model.py` & `scripts/comprehensive_evaluation.py`**:
  - Computes model perplexity and cross-entropy loss over validation/evaluation batches.
  - Tests generation quality across various temperatures (`0.3`, `0.8`, `1.2`).
  - Benchmarks generation speed (tokens per second).
  - Computes token diversity and bigram repetition scores.

### 4. Interfaces (`app/` & `scripts/generate_text.py`)

- **`app/streamlit_app.py`**:
  - Interactive Streamlit web application ("Shakespeare GPT").
  - Caches model loading (`@st.cache_resource`).
  - Provides sidebar controls for temperature, max tokens, top-k sampling, and preset prompt buttons.
- **`scripts/generate_text.py`**:
  - CLI text generation tool supporting single-prompt generation and an interactive terminal mode (`/quit`, `/temp`, `/length`, `/help`).

---

## Data and Control Flow

1. **Training Flow**:
   - `GPTDataLoader` loads text $\rightarrow$ encodes via `tiktoken` $\rightarrow$ batches input-target pairs $(x, y)$.
   - `GPT` forward pass $\rightarrow$ Causal Self-Attention & Transformer Blocks $\rightarrow$ Logits & Loss.
   - AdamW optimizer updates weights based on gradient clipping and cosine learning rate schedule.
   - Checkpoints saved to `models/checkpoints/ckpt.pt`.

2. **Inference / Generation Flow**:
   - Prompt string $\rightarrow$ `tiktoken` encoding $\rightarrow$ Tensor $x$.
   - `model.generate()` autoregressively predicts next-token logits $\rightarrow$ Applies temperature & top-k filtering $\rightarrow$ Appends token to sequence.
   - `tiktoken` decoding $\rightarrow$ Human-readable generated text output (CLI or Streamlit app).
