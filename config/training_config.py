"""Training configuration parameters"""

class TrainingConfig:
    # Training hyperparameters
    learning_rate = 3e-4
    batch_size = 16
    num_iterations = 5000
    eval_interval = 500
    save_interval = 1000
    
    # Optimization
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # Learning rate scheduler
    warmup_steps = 1000
    lr_decay_steps = 5000
    min_lr = 1e-5
