"""
LE (Luminance Estimation) Model Training Suite

This module provides a comprehensive framework for training and fine-tuning deep learning
models for luminance estimation tasks with and without the "Ola" dataset augmentation.

The suite includes:
- Data loading and preprocessing utilities
- Custom data generation with augmentation
- CNN-based model architecture
- Training pipelines for both scratch and transfer learning approaches
- Experiment tracking with multiple dataset sizes

Author: [Your Name]
Date: 2026
"""

import os
import json
import time
import warnings
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import GPUtil
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation, Dropout, Flatten, Dense, MaxPooling2D, 
    Conv2D, BatchNormalization, Input, Attention
)
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import clear_session

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Training Configuration
CONFIG = {
    'image_size': 64,
    'batch_size': 16,
    'epochs': 200,
    'learning_rate': 0.0001,
    'dropout_rate': 0.2,
    'patience_early_stopping': 10,
    'patience_reduce_lr': 5,
    'min_learning_rate': 1e-5,
    'factor_reduce_lr': 0.2,
}

# Dataset Configuration
DATASET_FOLDERS = {
    '100': 'path_to_folder_100',  # Update with actual path
    '200': 'path_to_folder_200',  # Update with actual path
    '300': 'path_to_folder_300',  # Update with actual path
    '400': 'path_to_folder_400',  # Update with actual path
    '500': 'path_to_folder_500',  # Update with actual path
}

TEST_DATASET_FOLDER = 'path_to_test_dataset_folder'  # Update with actual path
BASELINE_DATASET_FOLDER = 'path_to_baseline_dataset_folder'  # Update with actual path


# ============================================================================
# System Monitoring Functions
# ============================================================================

def get_system_memory() -> None:
    """Print system RAM memory information."""
    mem = psutil.virtual_memory()
    total_memory = mem.total / (1024 ** 3)
    used_memory = mem.used / (1024 ** 3)
    available_memory = mem.available / (1024 ** 3)
    
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Used Memory: {used_memory:.2f} GB")
    print(f"Available Memory: {available_memory:.2f} GB")


def get_gpu_memory() -> None:
    """Print GPU memory information."""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
        print(f"Total Memory: {gpu.memoryTotal} MB")
        print(f"Used Memory: {gpu.memoryUsed} MB")
        print(f"Free Memory: {gpu.memoryFree} MB")


def print_system_info() -> None:
    """Print complete system information."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print("\nSystem Memory:")
    get_system_memory()
    print("\nGPU Memory:")
    get_gpu_memory()
    print("=" * 70 + "\n")


# ============================================================================
# Data Processing Functions
# ============================================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a single image.
    
    Args:
        image: Input image array
        
    Returns:
        Preprocessed image (resized and normalized)
    """
    image = cv2.resize(image, (CONFIG['image_size'], CONFIG['image_size']))
    image = image / 255.0
    return image


def load_images(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess images from a directory.
    
    The function expects filenames in the format: 'prefix_label1_label2.ext'
    where label1 and label2 are float values extracted from the filename.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        Tuple of (samples array, labels array)
    """
    samples = []
    labels = []
    
    for image_filename in os.listdir(directory):
        image_path = os.path.join(directory, image_filename)
        
        # Skip if not a file
        if not os.path.isfile(image_path):
            continue
        
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read {image_path}")
            continue
            
        image = preprocess_image(image)
        samples.append(image)
        
        # Extract labels from filename
        # Expected format: filename_label1_label2.ext
        parts = image_filename.split('_')
        if len(parts) >= 3:
            try:
                label = np.zeros(2)
                label[0] = float(parts[1])
                label[1] = float(parts[2].split('.')[0])
                labels.append(label)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse labels from {image_filename}: {e}")
                continue
    
    samples = np.array(samples, dtype="float32")
    labels = np.array(labels)
    
    print(f"Loaded {len(samples)} samples from {directory}")
    return samples, labels


# ============================================================================
# Custom Data Generator
# ============================================================================

class CustomDataGenerator(Sequence):
    """
    Custom data generator with augmentation support.
    
    Inherits from keras.utils.Sequence for compatibility with model.fit()
    """
    
    def __init__(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        datagen: ImageDataGenerator = None
    ) -> None:
        """
        Initialize the data generator.
        
        Args:
            samples: Array of input samples
            labels: Array of corresponding labels
            batch_size: Size of each batch
            shuffle: Whether to shuffle indices after each epoch
            datagen: ImageDataGenerator for augmentation
        """
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.datagen = datagen
        self.on_epoch_end()
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self.samples) // self.batch_size
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch at given index.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of (batch_samples, batch_labels)
        """
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = np.array(
            [self.samples[i] for i in indices], 
            dtype=np.float32
        )
        batch_labels = np.array(
            [self.labels[i] for i in indices], 
            dtype=np.float32
        )
        
        # Apply data augmentation if available
        if self.datagen:
            augmented_samples = next(
                self.datagen.flow(
                    batch_samples, 
                    batch_size=self.batch_size, 
                    shuffle=False
                )
            )
        else:
            augmented_samples = batch_samples
        
        return augmented_samples, batch_labels
    
    def on_epoch_end(self) -> None:
        """Shuffle indices after each epoch."""
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# Model Building Functions
# ============================================================================

def create_cnn_model(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    dense_units: int = 96
) -> Model:
    """
    Create a CNN model for luminance estimation.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        dense_units: Number of units in the dense layer
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # First block
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second and third blocks
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    outputs = Dense(2)(x)
    
    model = Model(inputs, outputs)
    return model


def compile_model(model: Model) -> None:
    """
    Compile the model with appropriate loss and metrics.
    
    Args:
        model: Keras model to compile
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='mean_absolute_error',
        metrics=['mae']
    )


# ============================================================================
# Training Functions
# ============================================================================

def get_callbacks() -> list:
    """
    Create callbacks for training.
    
    Returns:
        List of Keras callbacks
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['patience_early_stopping'],
        restore_best_weights=False
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=CONFIG['factor_reduce_lr'],
        patience=CONFIG['patience_reduce_lr'],
        min_lr=CONFIG['min_learning_rate']
    )
    
    return [reduce_lr]


def train_model(
    model: Model,
    train_generator: CustomDataGenerator,
    test_generator: CustomDataGenerator,
    epochs: int = None,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Train a model and return training information.
    
    Args:
        model: Model to train
        train_generator: Training data generator
        test_generator: Validation data generator
        epochs: Number of epochs (uses CONFIG['epochs'] if None)
        model_name: Name for saving outputs
        
    Returns:
        Dictionary with training metrics
    """
    if epochs is None:
        epochs = CONFIG['epochs']
    
    clear_session()
    compile_model(model)
    callbacks = get_callbacks()
    
    start_time = time.time()
    
    print(f"\nTraining {model_name}...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        verbose=1,
        callbacks=callbacks
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return {
        'history': history,
        'time': {
            'seconds': training_time,
            'minutes': training_time / 60,
            'hours': training_time / 3600
        }
    }


def plot_training_results(history, model_name: str = "") -> None:
    """
    Plot training and validation loss and metrics.
    
    Args:
        history: Keras history object
        model_name: Name for plot title
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(14, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=2)
    plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=2)
    plt.title(f'Training and Validation Loss {model_name}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mae, 'b-o', label='Training MAE', linewidth=2, markersize=2)
    plt.plot(epochs, val_mae, 'r-s', label='Validation MAE', linewidth=2, markersize=2)
    plt.title(f'Training and Validation MAE {model_name}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MAE Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def print_training_time(training_time: Dict[str, float]) -> None:
    """
    Print training time in multiple formats.
    
    Args:
        training_time: Dictionary with 'seconds', 'minutes', 'hours' keys
    """
    print(f"\nTraining Time:")
    print(f"  {training_time['seconds']:.2f} seconds")
    print(f"  {training_time['minutes']:.2f} minutes")
    print(f"  {training_time['hours']:.2f} hours")


def save_history_to_json(history, filename: str) -> None:
    """
    Save training history to JSON file.
    
    Args:
        history: Keras history object
        filename: Output filename
    """
    def convert_to_native_python(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_native_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_python(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    history_dict = convert_to_native_python(history.history)
    
    with open(filename, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"History saved to {filename}")


# ============================================================================
# Experiment Functions
# ============================================================================

def train_baseline_model(
    save_dir: str = ".",
    model_name: str = "LE_Participant"
) -> None:
    """
    Train baseline model without Participant data.
    
    Args:
        save_dir: Directory to save model and history
        model_name: Name for the model files
    """
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING (WITHOUT Participant Data)")
    print("=" * 70)
    
    # Load data
    train_samples, train_labels = load_images(BASELINE_DATASET_FOLDER)
    test_samples, test_labels = load_images(TEST_DATASET_FOLDER)
    
    # Create data generators
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = CustomDataGenerator(
        train_samples, train_labels,
        batch_size=CONFIG['batch_size'],
        datagen=datagen
    )
    test_generator = CustomDataGenerator(
        test_samples, test_labels,
        batch_size=CONFIG['batch_size']
    )
    
    # Create and train model
    model = create_cnn_model()
    results = train_model(model, train_generator, test_generator, model_name=model_name)
    
    # Print and save results
    print_training_time(results['time'])
    plot_training_results(results['history'], f"({model_name})")
    
    # Save model and history
    model_path = os.path.join(save_dir, f"{model_name}.keras")
    json_path = os.path.join(save_dir, f"{model_name}.json")
    
    model.save(model_path)
    save_history_to_json(results['history'], json_path)
    print(f"Model saved to {model_path}")


def train_transfer_learning_model(
    base_model_path: str,
    dataset_key: str,
    save_dir: str = ".",
    epochs: int = None
) -> None:
    """
    Fine-tune a pre-trained model with a new dataset.
    
    Args:
        base_model_path: Path to the base model
        dataset_key: Key for dataset folder (e.g., '100', '200', etc.)
        save_dir: Directory to save results
        epochs: Number of epochs
    """
    if dataset_key not in DATASET_FOLDERS:
        raise ValueError(f"Invalid dataset key: {dataset_key}")
    
    model_name = f"LEWithParticipantRetrain_{dataset_key}"
    
    print("\n" + "=" * 70)
    print(f"TRANSFER LEARNING MODEL ({dataset_key} samples)")
    print("=" * 70)
    
    # Load pre-trained model
    model = load_model(base_model_path)
    
    # Load data
    train_samples, train_labels = load_images(DATASET_FOLDERS[dataset_key])
    test_samples, test_labels = load_images(TEST_DATASET_FOLDER)
    
    # Create data generators
    train_generator = CustomDataGenerator(
        train_samples, train_labels,
        batch_size=CONFIG['batch_size']
    )
    test_generator = CustomDataGenerator(
        test_samples, test_labels,
        batch_size=CONFIG['batch_size']
    )
    
    # Train model
    results = train_model(
        model, train_generator, test_generator,
        epochs=epochs,
        model_name=model_name
    )
    
    # Print and save results
    print_training_time(results['time'])
    plot_training_results(results['history'], f"({model_name})")
    
    json_path = os.path.join(save_dir, f"{model_name}.json")
    save_history_to_json(results['history'], json_path)


def train_scratch_model(
    dataset_key: str,
    save_dir: str = ".",
    epochs: int = None
) -> None:
    """
    Train a model from scratch with a specific dataset.
    
    Args:
        dataset_key: Key for dataset folder (e.g., '100', '200', etc.)
        save_dir: Directory to save results
        epochs: Number of epochs
    """
    if dataset_key not in DATASET_FOLDERS:
        raise ValueError(f"Invalid dataset key: {dataset_key}")
    
    model_name = f"LEWithParticipantScratch_{dataset_key}"
    
    print("\n" + "=" * 70)
    print(f"FROM SCRATCH MODEL ({dataset_key} samples)")
    print("=" * 70)
    
    # Load data
    train_samples, train_labels = load_images(DATASET_FOLDERS[dataset_key])
    test_samples, test_labels = load_images(TEST_DATASET_FOLDER)
    
    # Create data generators
    train_generator = CustomDataGenerator(
        train_samples, train_labels,
        batch_size=CONFIG['batch_size']
    )
    test_generator = CustomDataGenerator(
        test_samples, test_labels,
        batch_size=CONFIG['batch_size']
    )
    
    # Create and train model
    model = create_cnn_model()
    results = train_model(
        model, train_generator, test_generator,
        epochs=epochs,
        model_name=model_name
    )
    
    # Print and save results
    print_training_time(results['time'])
    plot_training_results(results['history'], f"({model_name})")
    
    json_path = os.path.join(save_dir, f"{model_name}.json")
    save_history_to_json(results['history'], json_path)


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """Main execution function."""
    
    # Print system information
    print_system_info()
    
    # Step 1: Train baseline model without Participant data
    train_baseline_model(model_name="LEWithoutParticipant")
    
    # Step 2: Fine-tune baseline model with different dataset sizes
    base_model = "LEWithoutParticipant.keras"
    for dataset_key in ['500', '400', '300', '200', '100']:
        train_transfer_learning_model(
            base_model_path=base_model,
            dataset_key=dataset_key
        )
    
    # Step 3: Train from scratch with different dataset sizes
    for dataset_key in ['500', '400', '300', '200', '100']:
        train_scratch_model(dataset_key=dataset_key)
    
    print("\n" + "=" * 70)
    print("All training experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
