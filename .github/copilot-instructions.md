# Copilot instructions

## Project

A small GPT-style language model built for learning. It trains on a
Shakespeare text dataset, generates text, evaluates the model, and offers a
Streamlit demo ("Shakespeare GPT").

Stack: Python, PyTorch, `tiktoken`. The package lives in
`src/llm_project/` (`models/gpt.py`, `models/attention.py`,
`models/transformer_blocks.py`, `data/data_loader.py`) and is installed with
`pip install -e .` (`setup.py`). Dependencies are in `requirements.txt`.

How it runs:

1. `pip install -r requirements.txt` and `pip install -e .`
2. Train: `python scripts/train_model.py`. Settings are variables at the top of
   the script (for example `dataset = 'shakespeare'`, `batch_size`,
   `block_size`, `out_dir = 'models/checkpoints'`). Check the script for the
   exact names and where the text file is read from.
3. Generate: `python scripts/generate_text.py --checkpoint models/checkpoints/ckpt.pt --prompt "…" --max_tokens 100 --temperature 0.8`.
4. Evaluate: `python scripts/evaluate_model.py` and
   `python scripts/comprehensive_evaluation.py`.
5. Demo: `streamlit run app/streamlit_app.py`.

Configuration: `config/training_config.py` (`TrainingConfig`) and
`.env.example`.

## Documentation notes

- The current `README.md` is broken. After a title and one-line intro it
  contains leftover shell commands (`touch …/.gitkeep`, `cat > config/model_config.py`)
  instead of documentation. Rewrite it into a proper README, keeping the
  title and intro.
- `config/model_config.py` is referenced by the old README but **does not
  exist** in the repo. Don't document it as existing.
- `app/streamlit_app.py` imports `streamlit`, but `streamlit` is **not** in
  `requirements.txt`. Tell users to install it separately, and add a
  `TODO: confirm …` suggesting it be added to the requirements.
- Check each script's arguments in its code before documenting them.
- The default branch is `master`.

## Ignore

- `src/llm_project.egg-info/` (generated), and data, checkpoint and model
  files.

## Conventions

- Keep changes small and focused. One concern per pull request.
- Base documentation on the actual code. Never invent features, metrics or
  commands. Mark anything uncertain with `TODO: confirm …`.
- Use UTF-8 for all text files.
