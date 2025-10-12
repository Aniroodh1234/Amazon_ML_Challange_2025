import sys
sys.path.append('/content/Amazon_ML_Challange_2025')
from update_config import update_config

CONFIGS = {
    'sub01': {
        'BATCH_SIZE': 4,
        'NUM_EPOCHS': 2,
        'LEARNING_RATE': 2e-5,
        'HIDDEN_DIM': 256,
        'IMAGE_SIZE': 196,
        'MAX_TEXT_LENGTH': 128,
        'GRADIENT_ACCUMULATION_STEPS': 8,
        'NUM_WORKERS': 2
    },
    'sub02': {  # Text-only
        'BATCH_SIZE': 8,
        'NUM_EPOCHS': 3,
        'LEARNING_RATE': 3e-5,
        'NUM_WORKERS': 2
    },
    'sub03': {
        'BATCH_SIZE': 4,
        'NUM_EPOCHS': 3,
        'LEARNING_RATE': 4e-5,
        'HIDDEN_DIM': 256,
        'IMAGE_SIZE': 196,
        'MAX_TEXT_LENGTH': 128,
        'GRADIENT_ACCUMULATION_STEPS': 8
    },
    'sub04': {
        'BATCH_SIZE': 4,
        'NUM_EPOCHS': 4,
        'LEARNING_RATE': 1e-5,
        'HIDDEN_DIM': 256,
        'IMAGE_SIZE': 196,
        'GRADIENT_ACCUMULATION_STEPS': 8
    },
    'sub06': {
        'BATCH_SIZE': 4,
        'NUM_EPOCHS': 6,
        'LEARNING_RATE': 1e-5,
        'HIDDEN_DIM': 256,
        'IMAGE_SIZE': 196
    },
    'sub07': {
        'BATCH_SIZE': 8,
        'NUM_EPOCHS': 3,
        'LEARNING_RATE': 3e-5,
        'HIDDEN_DIM': 256,
        'GRADIENT_ACCUMULATION_STEPS': 4
    },
    'sub08': {
        'BATCH_SIZE': 2,
        'NUM_EPOCHS': 3,
        'LEARNING_RATE': 2e-5,
        'HIDDEN_DIM': 384,
        'GRADIENT_ACCUMULATION_STEPS': 16
    },
    'sub09': {  # Image-only
        'BATCH_SIZE': 8,
        'NUM_EPOCHS': 4,
        'LEARNING_RATE': 2e-5,
        'NUM_WORKERS': 2
    },
    'sub11': {  # Ultimate
        'BATCH_SIZE': 4,
        'NUM_EPOCHS': 8,
        'LEARNING_RATE': 1.5e-5,
        'HIDDEN_DIM': 320,
        'DROPOUT_RATE': 0.15,
        'GRADIENT_ACCUMULATION_STEPS': 8,
        'IMAGE_SIZE': 196
    },
}

def apply_config(sub_name):
    """Apply configuration for a specific submission"""
    if sub_name not in CONFIGS:
        print(f"Config '{sub_name}' not found!")
        print(f"Available: {list(CONFIGS.keys())}")
        return False
    
    print(f"\nApplying config for {sub_name}...")
    update_config(CONFIGS[sub_name])
    print(f"Config applied for {sub_name}")
    return True

def show_all_configs():
    """Display all configurations"""
    print("\n" + "="*60)
    print("ALL SUBMISSION CONFIGURATIONS")
    print("="*60)
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        apply_config(sys.argv[1])
    else:
        show_all_configs()

