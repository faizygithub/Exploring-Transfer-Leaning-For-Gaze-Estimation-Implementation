"""
Model architecture definitions for training experiments.

Provides functions for creating and compiling neural network models.
"""

from typing import Tuple
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Flatten, 
    Dense, Dropout, Input
)


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    dense_units: int = 96,
    dropout_rate: float = 0.2
) -> Model:
    """
    Create a CNN model for luminance/feature estimation.
    
    Architecture:
    - Conv2D (32 filters) + BatchNorm + MaxPool
    - Conv2D (64 filters) + Conv2D (128 filters) + BatchNorm
    - Flatten + Dense + Dropout + Dense (output: 2 values)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        dense_units: Number of units in the dense hidden layer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second and third convolutional blocks
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2)(x)  # Output 2 values
    
    model = Model(inputs, outputs)
    return model


def compile_model(
    model: Model,
    learning_rate: float = 0.0001,
    loss: str = 'mean_absolute_error'
) -> None:
    """
    Compile the model with appropriate loss and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        loss: Loss function to use
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['mae']
    )


def create_and_compile_model(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    dense_units: int = 96,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.0001
) -> Model:
    """
    Create and compile a model in one step.
    
    Args:
        input_shape: Shape of input images
        dense_units: Number of units in the dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model ready for training
    """
    model = create_cnn_model(input_shape, dense_units, dropout_rate)
    compile_model(model, learning_rate)
    return model
