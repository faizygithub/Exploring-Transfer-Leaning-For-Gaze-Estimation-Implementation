"""
Training utilities for model training and evaluation.

Provides functions for training models, callbacks, visualization, and results saving.
"""

import os
import time
import json
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session

from data_loader import CustomDataGenerator


def get_callbacks(patience_early_stopping: int = 10, patience_reduce_lr: int = 5,
                  factor_reduce_lr: float = 0.2, min_learning_rate: float = 1e-5) -> list:
    """
    Create callbacks for training.
    
    Args:
        patience_early_stopping: Patience for early stopping
        patience_reduce_lr: Patience for learning rate reduction
        factor_reduce_lr: Factor to reduce learning rate
        min_learning_rate: Minimum learning rate
        
    Returns:
        List of Keras callbacks
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience_early_stopping,
        restore_best_weights=False
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor_reduce_lr,
        patience=patience_reduce_lr,
        min_lr=min_learning_rate
    )
    
    return [early_stopping, reduce_lr]


def train_model(
    model: Model,
    train_generator: CustomDataGenerator,
    test_generator: CustomDataGenerator,
    epochs: int = 200,
    callbacks: list = None,
    model_name: str = "model",
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Train a model and return training information.
    
    Args:
        model: Model to train
        train_generator: Training data generator
        test_generator: Validation data generator
        epochs: Number of epochs
        callbacks: List of callbacks
        model_name: Name for logging
        verbose: Verbosity level
        
    Returns:
        Dictionary with training metrics and history
    """
    if callbacks is None:
        callbacks = []
    
    clear_session()
    
    start_time = time.time()
    
    print(f"\nTraining {model_name}...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        verbose=verbose,
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


def plot_training_results(history, model_name: str = "", save_path: str = None) -> None:
    """
    Plot training and validation loss and metrics.
    
    Args:
        history: Keras history object
        model_name: Name for plot title
        save_path: Path to save the plot (optional)
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
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


def save_model(model: Model, filepath: str) -> None:
    """
    Save model to file.
    
    Args:
        model: Model to save
        filepath: Path where to save the model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model_from_file(filepath: str) -> Model:
    """
    Load model from file.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        Loaded Keras model
    """
    from tensorflow.keras.models import load_model
    model = load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model
