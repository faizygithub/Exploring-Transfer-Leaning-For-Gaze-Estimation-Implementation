# Model Training Suite for LE, Face, and Right Eye Experiments

## Project Overview

This is a clean, modular Python project for training deep learning models for three computer vision experiments:
- **LE (Left Eye)** - Luminance Estimation
- **Face** - Face Feature Detection
- **RE (Right Eye)** - Right Eye Feature Detection

All experiments follow the same architecture and training pipeline, making it easy to compare results across different regions.

## Project Structure

```
├── config.py                      # Configuration for all experiments
├── data_loader.py                 # Data loading and preprocessing modules
├── model_architecture.py           # CNN model definition
├── training_utils.py              # Training, callbacks, and utilities
├── system_monitoring.py           # GPU/RAM monitoring
├── experiments.py                 # Main experiment orchestrator
│
├── LE_Training_Clean.ipynb        # Notebook for LE experiment
├── Face_Training_Clean.ipynb      # Notebook for Face experiment
├── RE_Training_Clean.ipynb        # Notebook for Right Eye experiment
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Clean version control configuration
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide
└── CHANGELOG.md                   # Version history
```

## Module Descriptions

### config.py
**Purpose**: Centralized configuration management

**Key Features**:
- Experiment definitions (LE, Face, RE)
- Dataset paths for each experiment
- Training hyperparameters (learning rate, batch size, epochs, etc.)
- Dynamic configuration retrieval via `get_config(experiment_name)`

**Usage**:
```python
from config import get_config

config = get_config('LE')  # or 'Face', 'RE'
print(config['image_size'])  # 64
print(config['experiment_info']['name'])  # "Left Eye (Luminance Estimation)"
```

### data_loader.py
**Purpose**: Data loading, preprocessing, and batch generation

**Key Functions**:
- `load_images(directory, image_size=64)` - Load and preprocess images
- `preprocess_image(image, image_size=64)` - Single image preprocessing
- `CustomDataGenerator` - Keras Sequence for batched data loading with augmentation
- `get_augmentation_generator()` - Standard ImageDataGenerator for augmentation

**Features**:
- Automatic image resizing and normalization
- Label extraction from filenames (format: `prefix_label1_label2.jpg`)
- Data augmentation (rotation, shift, zoom, flip)
- Memory-efficient batch generation

### model_architecture.py
**Purpose**: Model architecture definitions

**Key Functions**:
- `create_cnn_model()` - Create CNN from scratch
- `compile_model()` - Compile with Adam optimizer
- `create_and_compile_model()` - Create and compile in one step

**Architecture**:
- Input: 64x64x3 images
- Conv2D(32) + BatchNorm + MaxPool
- Conv2D(64) + Conv2D(128) + BatchNorm
- Flatten + Dense(96) + Dropout + Dense(2)
- Output: 2 regression values

### training_utils.py
**Purpose**: Training pipeline and result management

**Key Functions**:
- `train_model()` - Main training function
- `plot_training_results()` - Visualization of loss and metrics
- `save_history_to_json()` - Save training history
- `get_callbacks()` - Create EarlyStopping and ReduceLROnPlateau callbacks
- `print_training_time()` - Display training duration

### system_monitoring.py
**Purpose**: Hardware resource monitoring

**Key Functions**:
- `get_system_memory()` - Print RAM usage
- `get_gpu_memory()` - Print GPU memory usage
- `print_system_info()` - Complete system information

### experiments.py
**Purpose**: Main experiment orchestrator

**Key Functions**:
- `train_baseline_model(experiment, save_dir)` - Train without augmentation
- `train_transfer_learning_model()` - Fine-tune baseline
- `train_from_scratch_model()` - Train new model with augmented data
- `run_complete_experiment()` - Run all three training approaches
- `run_all_experiments()` - Run all experiments (LE, Face, RE)

**Command-line Interface**:
```bash
# Run complete LE experiment
python experiments.py --experiment LE --mode full

# Run only baseline for Face
python experiments.py --experiment Face --mode baseline

# Run all experiments
python experiments.py --experiment all --mode full

# Analyze dataset sizes
python experiments.py --experiment RE --mode analyze

# Generate report
python experiments.py --experiment LE --mode report
```

## Usage Guide

### Option 1: Using Jupyter Notebooks

The easiest way to start is using the provided notebooks:

1. **LE Experiment**: `LE_Training_Clean.ipynb`
   - Section 1: Setup & Imports
   - Section 2: Configuration
   - Section 3-4: Load and prepare data
   - Section 5-7: Train baseline model
   - Section 8-9: Transfer learning and from-scratch
   - Section 10: Comparison and conclusions

2. **Face Experiment**: `Face_Training_Clean.ipynb`
   - Same structure as LE

3. **Right Eye Experiment**: `RE_Training_Clean.ipynb`
   - Same structure as LE

### Option 2: Command-line Experiments

```python
from experiments import run_complete_experiment

# Run LE experiment
run_complete_experiment(experiment='LE', save_dir='./le_results')

# Run Face experiment
run_complete_experiment(experiment='Face', save_dir='./face_results')

# Run all experiments
from experiments import run_all_experiments
run_all_experiments(save_dir='./all_results')
```

### Option 3: Custom Training Script

```python
from config import get_config
from data_loader import load_images, CustomDataGenerator, get_augmentation_generator
from model_architecture import create_and_compile_model
from training_utils import train_model, get_callbacks

# Get configuration
config = get_config('LE')
exp_info = config['experiment_info']

# Load data
train_samples, train_labels = load_images(
    exp_info['baseline'],
    image_size=config['image_size']
)
test_samples, test_labels = load_images(
    exp_info['test'],
    image_size=config['image_size']
)

# Create generators
datagen = get_augmentation_generator()
train_gen = CustomDataGenerator(train_samples, train_labels, datagen=datagen)
test_gen = CustomDataGenerator(test_samples, test_labels)

# Create and train model
model = create_and_compile_model(
    dense_units=96,
    dropout_rate=config['dropout_rate'],
    learning_rate=config['learning_rate']
)

callbacks = get_callbacks(
    patience_early_stopping=config['patience_early_stopping'],
    patience_reduce_lr=config['patience_reduce_lr'],
    factor_reduce_lr=config['factor_reduce_lr'],
    min_learning_rate=config['min_learning_rate']
)

results = train_model(
    model, train_gen, test_gen,
    epochs=config['epochs'],
    callbacks=callbacks
)
```

## Configuration

Key configuration parameters are in `config.py`:

```python
BASE_CONFIG = {
    'image_size': 64,           # Input image size (64x64)
    'batch_size': 16,           # Batch size for training
    'epochs': 200,              # Maximum epochs
    'learning_rate': 0.0001,    # Initial learning rate
    'dropout_rate': 0.2,        # Dropout rate
    'patience_early_stopping': 10,
    'patience_reduce_lr': 5,
    'min_learning_rate': 1e-5,
    'factor_reduce_lr': 0.2,
}
```

### Dataset Paths

Update the dataset paths in `config.py` for your local environment:

```python
LE_BASELINE_DATASET = r'E:\Data\...\LEWithoutOla-All'
LE_TEST_DATASET = r'E:\Data\...\LEWithOlaTest-All'
LE_TRAINING_DATASETS = {
    '100': r'E:\Data\...\LEWithOla\folder_100',
    '200': r'E:\Data\...\LEWithOla\folder_200',
    # ... etc
}
```

## Training Approaches

### 1. Baseline Model (Cross-Participant Foundation)
- Trained on data from MULTIPLE participants (e.g., 18 participants) - NOT including NEW target participant
- Creates a generalized, cross-participant foundation model
- Serves as starting point for transfer learning on NEW target participant
- Run ONCE and reuse for all transfer learning experiments
- Stored as: `participant_baseline.keras`

### 2. Transfer Learning (MAIN RESEARCH - Limited Data Analysis on NEW Participant)
- Starts from pre-trained baseline model weights (cross-participant foundation)
- Fine-tunes on NEW TARGET PARTICIPANT's data with augmentation
- Tests with EXTREMELY LIMITED target participant data: 100, 200, 300, 400, 500 samples
- Primary question: How does transfer learning adapt baseline to NEW participant with limited data?
- Secondary analysis: Data efficiency - How much NEW participant data is needed?
- Stored as: `participant_transfer_*.keras` (all fine-tuned for NEW target participant)

### 3. From-Scratch (Comparison)
- New model trained from random initialization on NEW TARGET PARTICIPANT's data
- Uses same limited NEW participant dataset sizes: 100, 200, 300, 400, 500 samples
- Compares against transfer learning to show benefit of pre-training vs training from scratch
- Stored as: `participant_scratch_*.keras` (all trained on NEW target participant)

## Output Files

After training, the following files are automatically generated:

```
├── participant_baseline.keras             # Baseline model (Step 1)
├── participant_baseline.json              # Training history
├── participant_transfer_500.keras         # Transfer learning (Step 2)
├── participant_transfer_500.json          # Training history
├── participant_transfer_400.keras
├── participant_transfer_400.json
├── participant_transfer_300.keras
├── participant_transfer_300.json
├── participant_transfer_200.keras
├── participant_transfer_200.json
├── participant_transfer_100.keras
├── participant_transfer_100.json
├── participant_scratch_500.keras          # From-scratch (Step 3)
├── participant_scratch_500.json           # Training history
├── ... participant_scratch_[size].* files
├── participant_report.json                # Summary report (Step 4)
```

All models use generic `participant_*` naming for privacy when uploading to GitHub.

### Training History JSON Format
```json
{
  "loss": [0.8234, 0.7123, ...],
  "val_loss": [0.7891, 0.7234, ...],
  "mae": [0.5234, 0.4123, ...],
  "val_mae": [0.5891, 0.4234, ...]
}
```

## Requirements

See `requirements.txt` for full dependencies:
- TensorFlow 2.x
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- psutil
- GPUtil

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run an experiment
python experiments.py --experiment LE --mode full
```

## Common Issues

### Dataset paths not found
Update the paths in `config.py` to match your local environment.

### Out of memory errors
Reduce `batch_size` in `config.py` from 16 to 8 or lower.

### GPU not detected
Set `GPU_DEVICE = -1` in `config.py` to force CPU training, or check CUDA installation.

## Performance Tips

1. **Increase batch size** to 32 if GPU memory allows (faster training)
2. **Reduce epochs** to 50-100 for quick testing
3. **Use transfer learning** - it's 2-3x faster than from-scratch
4. **Monitor GPU memory** with `python -c "from system_monitoring import print_system_info; print_system_info()"`

## Extending the Project

### Adding a new experiment (e.g., "Eyes")

1. Add configuration to `config.py`:
   ```python
   EYES_DATASET_FOLDERS = {...}
   EXPERIMENTS['Eyes'] = {...}
   ```

2. Create a notebook: `Eyes_Training_Clean.ipynb`
   - Change `experiment = 'Eyes'` in Section 2

3. Run with: `python experiments.py --experiment Eyes --mode full`

## Project Statistics

- **Total lines of code**: ~1,000+ (modular and clean)
- **Number of experiments**: 3 (LE, Face, RE)
- **Training approaches**: 3 (Baseline, Transfer Learning, From-Scratch)
- **Dataset sizes tested**: 5 (100, 200, 300, 400, 500 samples)
- **Total possible model variants**: 25+

## Citation & Notes

This project implements:
- Convolutional Neural Networks for gaze regression
- Multi-participant transfer learning methodology
- Data augmentation (rotation, shift, zoom, flip)
- Systematic evaluation across dataset sizes (100-500)
- Reproducible experiment tracking and reporting

## License

[Add your license here]

## Author

[PhD Research - Experiment Paper 3]

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
