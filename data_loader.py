"""
Data loading and preprocessing module for model training.

Provides functions for loading images, preprocessing, and custom data generation.
"""

import os
from typing import Tuple
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


def preprocess_image(image: np.ndarray, image_size: int = 64) -> np.ndarray:
    """
    Preprocess a single image.
    
    Args:
        image: Input image array
        image_size: Target image size (default: 64)
        
    Returns:
        Preprocessed image (resized and normalized to [0, 1])
    """
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    return image


def load_images(directory: str, image_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess images from a directory.
    
    Expects filenames in the format: 'prefix_label1_label2.ext'
    where label1 and label2 are float values extracted from the filename.
    
    Args:
        directory: Path to directory containing images
        image_size: Target image size (default: 64)
        
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
            
        image = preprocess_image(image, image_size=image_size)
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


def get_augmentation_generator(config: dict = None) -> ImageDataGenerator:
    """
    Create an ImageDataGenerator with standard augmentation.
    
    Args:
        config: Dictionary with augmentation parameters. If None, uses defaults.
        
    Returns:
        Configured ImageDataGenerator
    """
    if config is None:
        config = {
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }
    
    return ImageDataGenerator(**config)
