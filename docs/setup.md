# Setup Guide

This guide provides detailed instructions for setting up the local development environment for `nanoGPT`, running the project, and troubleshooting common issues.

## Prerequisites

- **Python**: Version 3.8 or higher (`python3 --version`)
- **Pip**: Python package installer (`pip --version`)
- **PyTorch**: Version 2.0.0 or higher (CPU or CUDA-enabled GPU)

## Local Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gopaldp/nanoGPT.git
   cd nanoGPT
   ```

2. **Create and Activate a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install the Package in Editable Mode**:
   ```bash
   pip install -e .
   ```

5. **Optional UI Dependency**:
   To run the Streamlit web application (`app/streamlit_app.py`), install `streamlit`:
   ```bash
   pip install streamlit
   ```
   > *TODO: confirm if `streamlit` should be added to `requirements.txt`.*

## Environment Configuration

Copy the example environment file to `.env` to configure training variables and paths:
```bash
cp .env.example .env
```
Edit `.env` to configure model name, batch size, learning rate, and checkpoints paths as needed.

## Verification & Testing

Verify your installation by running the test suite with `pytest`:
```bash
pytest
```

## Troubleshooting Common Issues

### 1. `ModuleNotFoundError: No module named 'llm_project'`
- **Cause**: The package is not installed in the python environment.
- **Solution**: Run `pip install -e .` from the repository root, or ensure `src/` is added to `PYTHONPATH` (scripts automatically append `src/` to `sys.path`).

### 2. CUDA Out of Memory (OOM) / GPU Errors
- **Cause**: Batch size or model size too large for available GPU/CPU memory.
- **Solution**: Reduce `batch_size`, `block_size`, or model dimensions (`n_layer`, `n_embd`) in `scripts/train_model.py` or `config/training_config.py`.

### 3. Missing Dataset
- **Cause**: The Tiny Shakespeare dataset has not been downloaded.
- **Solution**: Running `python scripts/train_model.py` automatically downloads the dataset from GitHub (`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`) into `data/raw/shakespeare.txt`.

### 4. Streamlit Command Not Found
- **Cause**: `streamlit` package is not installed in the active environment.
- **Solution**: Install streamlit via `pip install streamlit`.
