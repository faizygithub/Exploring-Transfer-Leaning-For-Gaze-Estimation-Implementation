"""
Configuration file for Gaze Estimation Transfer Learning Research

Paper: "Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability"

Supports three gaze estimation regions: LE (Left Eye), Face, and RE (Right Eye).
Each experiment uses a multi-participant transfer learning approach:
  - Baseline: Trained on all other participants' baseline data (no augmentation)
  - Transfer Learning: Fine-tuned on target participant with varying dataset sizes (500, 400, 300, 200, 100)
  - Fair Comparison: Consistent test set across all dataset sizes for unbiased evaluation

Customize these settings to match your environment and research needs.
"""

# ============================================================================
# Experiment-Specific Dataset Configuration
# ============================================================================

# LE (Left Eye) - Gaze estimation from left eye region
LE_BASELINE_DATASET = r'E:\Data\FinalData\AllData\FirstExperimentOla\LEWithoutOla-All'  # All participants baseline data
LE_TEST_DATASET = r'E:\Data\FinalData\AllData\FirstExperimentOla\LEWithOlaTest-All'  # Target participant test set
LE_TRAINING_DATASETS = {  # Target participant augmented data with varying sizes for transfer learning
    '100': r'E:\Data\FinalData\AllData\FirstExperimentOla\LEWithOla\folder_100',
    '200': r'E:\Data\FinalData\AllData\FirstExperimentOla\LEWithOla\folder_200',
    '300': r'E:\Data\FinalData\AllData\FirstExperimentOla\LEWithOla\folder_300',
    '400': r'E:\Data\FinalData\AllData\FirstExperimentOla\LEWithOla\folder_400',
    '500': r'E:\Data\FinalData\AllData\FirstExperimentOla\LEWithOla\folder_500',
}

# Face - Gaze features and head pose context  
FACE_BASELINE_DATASET = r'E:\Data\FinalData\AllData\FirstExperimentOla\FaceWithoutOla-All'  # All participants baseline data
FACE_TEST_DATASET = r'E:\Data\FinalData\AllData\FirstExperimentOla\FaceWithOlaTest-All'  # Target participant test set
FACE_TRAINING_DATASETS = {  # Target participant augmented data with varying sizes for transfer learning
    '100': r'E:\Data\FinalData\AllData\FirstExperimentOla\FaceWithOla\folder_100',
    '200': r'E:\Data\FinalData\AllData\FirstExperimentOla\FaceWithOla\folder_200',
    '300': r'E:\Data\FinalData\AllData\FirstExperimentOla\FaceWithOla\folder_300',
    '400': r'E:\Data\FinalData\AllData\FirstExperimentOla\FaceWithOla\folder_400',
    '500': r'E:\Data\FinalData\AllData\FirstExperimentOla\FaceWithOla\folder_500',
}

# RE (Right Eye) - Gaze estimation from right eye region
RE_BASELINE_DATASET = r'E:\Data\FinalData\AllData\FirstExperimentOla\REWithoutOla-All'  # All participants baseline data
RE_TEST_DATASET = r'E:\Data\FinalData\AllData\FirstExperimentOla\REWithOlaTest-All'  # Target participant test set
RE_TRAINING_DATASETS = {  # Target participant augmented data with varying sizes for transfer learning
    '100': r'E:\Data\FinalData\AllData\FirstExperimentOla\REWithOla\folder_100',
    '200': r'E:\Data\FinalData\AllData\FirstExperimentOla\REWithOla\folder_200',
    '300': r'E:\Data\FinalData\AllData\FirstExperimentOla\REWithOla\folder_300',
    '400': r'E:\Data\FinalData\AllData\FirstExperimentOla\REWithOla\folder_400',
    '500': r'E:\Data\FinalData\AllData\FirstExperimentOla\REWithOla\folder_500',
}

# Experiment mapping (for dynamic access)
EXPERIMENTS = {
    'LE': {
        'name': 'Left Eye (LE) Gaze Estimation',
        'baseline': LE_BASELINE_DATASET,  # Multi-participant baseline
        'test': LE_TEST_DATASET,  # Target participant test (constant)
        'training': LE_TRAINING_DATASETS,  # Target participant train (variable sizes)
    },
    'Face': {
        'name': 'Face Gaze Features',
        'baseline': FACE_BASELINE_DATASET,  # Multi-participant baseline
        'test': FACE_TEST_DATASET,  # Target participant test (constant)
        'training': FACE_TRAINING_DATASETS,  # Target participant train (variable sizes)
    },
    'RE': {
        'name': 'Right Eye (RE) Gaze Estimation',
        'baseline': RE_BASELINE_DATASET,  # Multi-participant baseline
        'test': RE_TEST_DATASET,  # Target participant test (constant)
        'training': RE_TRAINING_DATASETS,  # Target participant train (variable sizes)
    },
}

# Default dataset configuration (backward compatibility)
BASELINE_DATASET = LE_BASELINE_DATASET
TEST_DATASET = LE_TEST_DATASET
TRAINING_DATASETS = LE_TRAINING_DATASETS

# ============================================================================
# Model Training Configuration
# ============================================================================

# Image preprocessing
IMAGE_SIZE = 64  # Square image dimension (64x64)

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.2

# Model architecture
DENSE_UNITS = 96  # Units in the dense hidden layer

# ============================================================================
# Optimization and Callbacks
# ============================================================================

# Adam optimizer settings
OPTIMIZER_BETA_1 = 0.9
OPTIMIZER_BETA_2 = 0.999
OPTIMIZER_EPSILON = 1e-7

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_MODE = 'min'

# Learning rate reduction on plateau
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 5
MIN_LEARNING_RATE = 1e-5
REDUCE_LR_MONITOR = 'val_loss'
REDUCE_LR_MODE = 'min'

# ============================================================================
# Data Augmentation Settings
# ============================================================================

AUGMENTATION_CONFIG = {
    'rotation_range': 10,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
}

# ============================================================================
# Output and Logging
# ============================================================================

# Directory for saving models and history
OUTPUT_DIRECTORY = '.'

# Save formats
SAVE_MODEL_FORMAT = 'keras'  # 'keras' or 'h5'
SAVE_HISTORY_FORMAT = 'json'  # Always JSON
SAVE_PLOTS = True

# Logging verbosity
VERBOSE_LEVEL = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch

# ============================================================================
# GPU Configuration
# ============================================================================

# GPU device ID (0 = first GPU, -1 = CPU only)
GPU_DEVICE = 0

# Enable dynamic memory growth
GPU_DYNAMIC_MEMORY = True

# Allow memory growth
GPU_ALLOW_GROWTH = True

# ============================================================================
# Experiment Configuration
# ============================================================================

# Which experiments to run
RUN_BASELINE = True
RUN_TRANSFER_LEARNING = True
RUN_FROM_SCRATCH = True

# Dataset sizes for transfer learning and from-scratch experiments
DATASET_SIZES = ['500', '400', '300', '200', '100']

# ============================================================================
# Research Paper Settings
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Paper/project name
PROJECT_NAME = "Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability"

# Author information
AUTHOR_NAME = "[Your Name]"
AUTHOR_EMAIL = "[your-email@example.com]"

# Institution/Affiliation
INSTITUTION = "[Your Institution]"

# Research date
RESEARCH_DATE = "2026"


# ============================================================================
# Helper Functions
# ============================================================================

def get_config(experiment: str = 'LE') -> dict:
    """
    Get configuration for a specific experiment.
    
    Research Strategy:
      - Baseline: Trained on all participants' baseline data (multi-participant)
      - Test: Evaluation set for the target participant (constant across all sizes)
      - Training: Target participant's augmented data with varying sizes (100-500)
    
    Args:
        experiment: Experiment name ('LE', 'Face', or 'RE')
        
    Returns:
        Dictionary with experiment-specific configuration and hyperparameters
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}. Must be one of {list(EXPERIMENTS.keys())}")
    
    exp_info = EXPERIMENTS[experiment]
    
    return {
        'experiment': experiment,
        'experiment_info': exp_info,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE,
        'dense_units': DENSE_UNITS,
        'patience_early_stopping': EARLY_STOPPING_PATIENCE,
        'patience_reduce_lr': REDUCE_LR_PATIENCE,
        'factor_reduce_lr': REDUCE_LR_FACTOR,
        'min_learning_rate': MIN_LEARNING_RATE,
    }
