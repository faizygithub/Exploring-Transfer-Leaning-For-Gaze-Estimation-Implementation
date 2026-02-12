# Gaze Estimation Transfer Learning - Experimental Suite

**Paper**: *Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability*

A comprehensive deep learning framework for training CNN-based models for gaze estimation experiments (Left Eye, Face, Right Eye). This suite investigates **model behavior and accuracy patterns with limited data** by implementing transfer learning across extremely low dataset sizes (100-500 samples). The framework evaluates convergence behavior, accuracy trends, and data efficiency to understand how transfer learning performs with minimal target participant data, as presented in the research paper above.

## Overview

This project implements the experimental framework for cross-participant transfer learning research in gaze estimation with **limited data scenarios**. It trains a foundation model on data from MULTIPLE participants (e.g., 18 participants), then fine-tunes this cross-participant baseline on a NEW target participant (e.g., 19th participant) with extremely low dataset sizes (100, 200, 300, 400, 500 samples). The primary research aim is to evaluate **model convergence patterns, accuracy trends, and data efficiency** - understanding how well transfer learning works when target participant data is severely limited. It includes:

- **Data Loading & Preprocessing**: Efficient image loading with automatic label extraction
- **Custom Data Generator**: Keras-compatible generator with on-the-fly augmentation
- **CNN Architecture**: Purpose-built convolutional neural network for regression tasks
- **Multiple Training Strategies**:
  - Baseline model training (without augmentation)
  - Transfer learning with fine-tuning
  - Training from scratch
- **Comprehensive Logging**: Training history saved as JSON, visualization plots
- **Privacy-Focused**: Generic model naming (participant_baseline, participant_transfer, etc.) for safe GitHub uploads

## Supported Experiments

This suite evaluates three key regions for gaze estimation:

- **LE** (Left Eye): Gaze direction estimation from left eye region
- **Face**: Face-based gaze features and overall head pose context
- **RE** (Right Eye): Gaze direction estimation from right eye region

All experiments follow the same CNN architecture and training pipeline, enabling direct comparison of transfer learning effectiveness across different facial regions.

## Requirements

```
tensorflow>=2.0
numpy
opencv-python
pandas
matplotlib
psutil
gputil
scikit-learn
```

### Installation

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install tensorflow numpy opencv-python pandas matplotlib psutil gputil scikit-learn
```

## Project Structure

```
.
├── config.py                     # Configuration (LE, Face, RE experiments)
├── data_loader.py               # Data loading & preprocessing
├── model_architecture.py         # CNN model definition
├── training_utils.py            # Training pipeline & callbacks
├── system_monitoring.py         # GPU/RAM monitoring
├── experiments.py               # Main experiment runner
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LE_Training_Clean.ipynb      # Clean notebook for LE
├── Face_Training_Clean.ipynb    # Clean notebook for Face
├── RE_Training_Clean.ipynb      # Clean notebook for RE
└── models/                      # Directory for saved models (auto-generated)
    ├── participant_baseline.keras           # Baseline model
    ├── participant_baseline.json            # Training history
    ├── participant_transfer_*.keras         # Transfer learning models
    ├── participant_transfer_*.json
    ├── participant_scratch_*.keras          # From-scratch models
    └── participant_scratch_*.json
```

## Model Naming Convention

**All models use generic "participant" names for privacy:**

| Type | Filename |
|------|----------|
| Baseline | `participant_baseline.keras` / `.json` |
| Transfer Learning (dataset size N) | `participant_transfer_N.keras` / `.json` |
| From Scratch (dataset size N) | `participant_scratch_N.keras` / `.json` |
| Report | `participant_report.json` |

This naming scheme keeps your experiment data private when uploading to GitHub. The configuration file tracks which experiment type (LE, Face, RE) is being used.

Example: `image_45_78.jpg`

Where:
- `45`: Gaze coordinate X (integer)
- `78`: Gaze coordinate Y (integer)

Note: Numbers represent gaze positions on the screen (pixel coordinates), directly extracted from image filenames for automatic label loading.

### Directory Structure

Your data folders (generic names shown; configure actual paths in config.py):

```
MultiParticipantBaseline/         # Multi-participant baseline data
├── image1_45_78.jpg
├── image2_52_85.jpg
└── ...

TargetParticipantAugmented/       # Target participant augmented data
├── folder_100/                   # 100 training samples
├── folder_200/                   # 200 training samples
├── folder_300/                   # 300 training samples
├── folder_400/                   # 400 training samples
└── folder_500/                   # 500 training samples

TargetParticipantTest/            # Target participant test set (constant)
├── test_image1_45_78.jpg
└── ...
```

**Note:** Actual folder paths are configured in [config.py](config.py). Model output uses generic 'participant_*' naming for privacy.

## Usage

### Command Line Interface (Recommended)

The main entry point is `experiments.py`, which supports multiple modes corresponding to your research strategy:

```bash
# Step 1: Train baseline on multi-participant data
python experiments.py --mode baseline --experiment LE

# Step 2: Transfer learning - fine-tune on target participant (MAIN RESEARCH)
python experiments.py --mode transfer --experiment LE

# Step 3: From-scratch training for comparison
python experiments.py --mode scratch --experiment LE

# Step 4: Generate report
python experiments.py --mode report --experiment LE

# Optional: Analyze datasets
python experiments.py --mode analyze --experiment LE

# Run complete pipeline for all regions
python experiments.py --mode full --experiment all
```

For detailed information on all modes and workflows, see [EXPERIMENTS_MODES.md](EXPERIMENTS_MODES.md).

### Python API

```python
from experiments import (
    train_baseline_model,
    train_transfer_learning_model,
    train_from_scratch_model,
    run_complete_experiment
)

# Train baseline model (Step 1)
train_baseline_model(experiment='LE')

# Fine-tune with 500 samples (Step 2)
train_transfer_learning_model(
    experiment='LE',
    base_model_path="participant_baseline.keras",
    dataset_key="500"
)

# Train from scratch with 300 samples (Step 3)
train_from_scratch_model(experiment='LE', dataset_key="300")

# Or run complete pipeline
run_complete_experiment(experiment='LE')
```

## Model Architecture

The CNN architecture consists of:

- **Input Layer**: 64×64×3 (RGB images)
- **Conv Block 1**: 32 filters, 3×3 kernel, ReLU activation
- **BatchNorm + MaxPool**: Normalization and downsampling
- **Conv Block 2**: 64 filters, 3×3 kernel
- **Conv Block 3**: 128 filters, 3×3 kernel
- **BatchNorm**: Normalization
- **Flatten**: Convert to 1D
- **Dense Layer**: 96 units, ReLU activation
- **Dropout**: 20% dropout rate
- **Output Layer**: 2 units (regression)

**Loss Function**: Mean Absolute Error (MAE)  
**Optimizer**: Adam (initial learning rate: 0.0001)

## Training Strategies

### 1. Baseline Model (Cross-Participant Foundation)
- Trained on data from MULTIPLE participants (e.g., 18 participants)
- Creates a generalized foundation model across many participants
- Target participant data NOT included in baseline
- Uses standard data augmentation (rotation, shift, zoom, flip)

### 2. Transfer Learning (Fine-tuning) - MAIN RESEARCH
- Loads the baseline model trained on multiple participants (Step 1)
- Fine-tunes on a NEW target participant with extremely limited dataset sizes (100, 200, 300, 400, 500 samples)
- Test set kept CONSTANT for fair evaluation
- **PRIMARY RESEARCH AIM**: Study model behavior, convergence patterns, and accuracy trends with extremely limited target participant data
- Shows data efficiency: minimum target participant data needed for acceptable performance
- Analyzes how accuracy changes as target participant dataset size decreases from 500→100

### 3. From Scratch Training (Comparison Baseline)
- Trains new models without pre-training on target participant data
- Uses same dataset sizes as transfer learning (100-500 samples)
- Comparison baseline to demonstrate transfer learning advantages with limited data
- Shows why pre-training is critical when working with extremely low sample counts (100-500)

## Callbacks and Regularization

- **Early Stopping**: Monitors validation loss, patience=10 epochs
- **Reduce LR on Plateau**: Reduces learning rate if validation loss plateaus, factor=0.2, patience=5
- **Dropout**: 20% dropout in the final dense layer
- **Batch Normalization**: Applied after convolutional blocks

## Output Files

For each training run, the following files are generated:

1. **Model Files**: `participant_*.keras`
   - Complete trained models using generic privacy-focused naming
   - Examples: `participant_baseline.keras`, `participant_transfer_500.keras`, `participant_scratch_100.keras`
   - Can be loaded with `tensorflow.keras.models.load_model()`

2. **History Files**: `participant_*.json`
   - Training history containing:
     - `loss`: Training loss per epoch
     - `val_loss`: Validation loss per epoch
     - `mae`: Training MAE per epoch
     - `val_mae`: Validation MAE per epoch

3. **Report File**: `participant_report.json`
   - Summary of all experimental results
   - Enables comparison across all approaches and dataset sizes

4. **Plots**: Matplotlib figures showing
   - Training vs Validation Loss
   - Training vs Validation MAE

## System Monitoring

The module includes utilities to monitor system resources:

```python
from le_model_training import print_system_info

# Print RAM and GPU memory information
print_system_info()
```

## Performance Metrics

Training results are evaluated using:

- **Mean Absolute Error (MAE)**: Primary metric for regression tasks
- **Training Time**: Recorded in seconds, minutes, and hours
- **GPU Memory**: Monitored throughout training
- **System Memory**: Monitored for memory efficiency

## GPU Configuration

The module is configured for single GPU training:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Dynamic memory allocation
```

Modify these settings in the configuration section to use different GPUs or CPU training.

## Data Format Flexibility

The `load_images()` function is designed to be flexible:

- Handles images in common formats (JPG, PNG, etc.)
- Automatically resizes images to 64×64
- Normalizes pixel values to [0, 1] range
- Extracts labels from filename using underscore-separated format

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in [config.py](config.py)
- Reduce `image_size` if necessary
- Ensure GPU memory is sufficient

### Label Parsing Error
- Verify image filenames follow the format: `prefix_X_Y.ext`
- Example: `image_45_78.jpg` is valid (gaze coordinates), `image_45.78.jpg` or `image_4578.jpg` are not
- Coordinates should be integers (pixel positions)

### Model Not Loading
- Verify the model file exists and path is correct
- Ensure TensorFlow version matches the version used to save the model
- Check file is not corrupted with `python -c "import tensorflow as tf; tf.keras.models.load_model('path')"`

### GPU Not Detected
- Check CUDA installation: `nvidia-smi`
- Verify TensorFlow GPU version: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## Documentation

For detailed information about the research methodology and experiment modes, see:

- [EXPERIMENTS_MODES.md](EXPERIMENTS_MODES.md) - Complete guide to all command-line modes and workflows
- [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) - Deep dive into system design and configuration
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed module descriptions
- [NAMING_CONVENTION.md](NAMING_CONVENTION.md) - Privacy-focused naming strategy
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Citation

This code is associated with the paper:

```
Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability
```

Citation information and repository will be available upon publication.



**Associated Code Repository**: This is the official implementation of experiments for the above paper.

## License

[Specify your license here - e.g., MIT, GPL, etc.]

## Contact

For questions or issues, please contact open an issue on GitHub.

## Acknowledgments

- TensorFlow/Keras for the deep learning framework
- OpenCV for image processing
- The research dataset contributors

---

**Last Updated**: February 12, 2026  
**Supported Regions**: Left Eye (LE), Face, Right Eye (RE)  
**Repository**: [Add your GitHub URL]
