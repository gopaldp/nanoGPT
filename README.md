# My GPT Project 🚀

A small-scale Large Language Model (LLM) implementation for learning purposes.

## Project Structure


python -m venv venv
# Create .gitkeep files to preserve empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/interim/.gitkeep
touch data/external/.gitkeep
touch models/checkpoints/.gitkeep
touch models/final/.gitkeep
touch models/experiments/.gitkeep
touch results/figures/.gitkeep
touch results/metrics/.gitkeep
touch results/generated_text/.gitkeep

# Create initial configuration files
cat > config/model_config.py << 'EOF'
"""Model configuration parameters"""

class ModelConfig:
    # Model architecture
    vocab_size = 50257
    block_size = 1024
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.1
    
    # Model variants
    SMALL = {
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'block_size': 512,
    }
    
    MEDIUM = {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
    }
