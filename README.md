# nanoGPT 🎭

A small-scale, educational Large Language Model (LLM) implementation inspired by Andrej Karpathy's `nanoGPT`. This repository provides a clean, readable PyTorch implementation of a GPT-style transformer model, complete with training scripts, evaluation utilities, and a Streamlit interactive demo web application.

## Key Features

- **Transformer Architecture**: Clean modular implementation of multi-head self-attention (`src/llm_project/models/attention.py`), transformer blocks (`src/llm_project/models/transformer_blocks.py`), and the full GPT language model (`src/llm_project/models/gpt.py`).
- **Data Pipeline**: Efficient dataset loading and batching via `GPTDataLoader` using `tiktoken` (`src/llm_project/data/data_loader.py`).
- **Training Script**: Automated dataset downloading (Tiny Shakespeare corpus), mixed-precision support (`bfloat16`/`float16`), cosine learning rate scheduler with warmup, gradient clipping, and optional `torch.compile` support (`scripts/train_model.py`).
- **Evaluation Utilities**: Perplexity computation, generation quality testing across multiple temperatures, speed benchmarking (`scripts/evaluate_model.py`), and comprehensive metrics tracking (`scripts/comprehensive_evaluation.py`).
- **Interactive Generation**: CLI interactive mode with commands (`scripts/generate_text.py`) and a fully-featured Streamlit web app (`app/streamlit_app.py`).

## Tech Stack

- **Python** (>=3.8)
- **PyTorch** (>=2.0.0)
- **tiktoken** (Tokenization)
- **Streamlit** (Web UI)
- **Transformers**, **Datasets**, **Matplotlib**, **Pandas**, **NumPy**

## Prerequisites

- Python 3.8 or higher
- PyTorch compatible environment (CPU or CUDA GPU)

## Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/gopaldp/nanoGPT.git
   cd nanoGPT
   ```

2. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

   > **Note on Streamlit**: `streamlit` is required to run the web application (`app/streamlit_app.py`) but is currently not listed in `requirements.txt`. Install it manually if needed:
   > ```bash
   > pip install streamlit
   > ```
   > *TODO: confirm if `streamlit` should be added to `requirements.txt` or `setup.py`.*

## Usage

### 1. Training the Model
Train the GPT model on the Tiny Shakespeare dataset using default settings:
```bash
python scripts/train_model.py
```
This script automatically downloads `tinyshakespeare/input.txt` if not present, initializes a small GPT configuration, saves checkpoints into `models/checkpoints/ckpt.pt`, and trains using mixed precision where supported.

### 2. Generating Text (CLI)
Generate text from a trained checkpoint with custom prompts:
```bash
python scripts/generate_text.py --checkpoint models/checkpoints/ckpt.pt --prompt "ROMEO:" --max_tokens 100 --temperature 0.8
```
To run in interactive mode:
```bash
python scripts/generate_text.py --checkpoint models/checkpoints/ckpt.pt --interactive
```
In interactive mode, available commands include `/quit`, `/temp <value>`, `/length <value>`, and `/help`.

### 3. Evaluating the Model
Evaluate model perplexity, generation quality across various temperatures, and speed benchmarking:
```bash
python scripts/evaluate_model.py
```
For comprehensive metrics (including token diversity and repetition scores):
```bash
python scripts/comprehensive_evaluation.py
```

### 4. Running the Streamlit App
Launch the interactive web UI ("Shakespeare GPT"):
```bash
streamlit run app/streamlit_app.py
```
Features sidebar controls for temperature, max tokens, top-k sampling, and quick prompt buttons (`ROMEO:`, `JULIET:`, `HAMLET:`).

## Configuration

Configuration parameters and environment variables are documented in `.env.example`. You can copy it to `.env`:
```bash
cp .env.example .env
```
Key configuration settings include:
- **Training Configuration** (`config/training_config.py`): Learning rate (`3e-4`), batch size (`16`), weight decay, warmup steps, gradient clipping, etc.
- **Environment variables**: `MODEL_NAME`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `WANDB_PROJECT`, etc.

## Project Structure

```text
nanoGPT/
├── app/
│   └── streamlit_app.py          # Streamlit web application UI
├── config/
│   └── training_config.py        # Training hyperparameters and settings
├── scripts/
│   ├── comprehensive_evaluation.py # Comprehensive metrics (perplexity, diversity, repetition)
│   ├── evaluate_model.py         # Model evaluation (perplexity, generation quality, speed)
│   ├── generate_text.py          # CLI text generation & interactive mode
│   └── train_model.py            # Model training script
├── src/
│   └── llm_project/
│       ├── data/
│       │   └── data_loader.py    # GPT dataset loading & batching
│       ├── models/
│       │   ├── attention.py      # Multi-head self-attention module
│       │   ├── gpt.py            # GPT model architecture & config
│       │   └── transformer_blocks.py # Transformer decoder blocks
│       └── utils/                # Utility modules
├── tests/                        # Test suite
├── requirements.txt              # Project dependencies
├── setup.py                      # Package installation script
└── README.md
```

## Testing

Run tests using `pytest`:
```bash
pytest
```

## Documentation

For further details, see:
- [Architecture Documentation](docs/architecture.md)
- [Setup Guide](docs/setup.md)
