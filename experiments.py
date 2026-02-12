"""
Main experiment runner for Gaze Estimation Transfer Learning Research with Limited Data.

Paper: "Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability"

Primary Research Focus:
  Investigating model behavior, convergence patterns, and accuracy trends when transfer learning
  is applied with EXTREMELY LIMITED target participant data (100-500 samples).

Research Strategy:
  1. BASELINE: Train on data from MULTIPLE participants (e.g., 18) - cross-participant foundation
  2. TRANSFER LEARNING (MAIN): Fine-tune on NEW target participant with progressively limited data
     - Study model behavior across dataset sizes: 500, 400, 300, 200, 100 samples
     - Analyze: accuracy trends, convergence patterns, minimum viable dataset
  3. FROM-SCRATCH: Train new models on target participant (comparison baseline)
  4. FAIR COMPARISON: Keep test set constant for unbiased evaluation

Supports three gaze regions: LE (Left Eye), Face, and RE (Right Eye).

Usage:
    python experiments.py --mode baseline --experiment LE      # Step 1: Train base model
    python experiments.py --mode transfer --experiment LE      # Step 2: Main research - limited data analysis
    python experiments.py --mode scratch --experiment LE       # Step 3: Comparison training
    python experiments.py --mode report --experiment LE        # Step 4: Generate results
    python experiments.py --mode analyze --experiment LE       # Optional: Data exploration
"""

import os
import json
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

import config
from data_loader import load_images, CustomDataGenerator, get_augmentation_generator
from model_architecture import create_and_compile_model
from training_utils import (
    train_model, plot_training_results, print_training_time,
    save_history_to_json, save_model, get_callbacks
)
from system_monitoring import print_system_info

warnings.filterwarnings('ignore')


# ============================================================================
# Baseline Training Functions
# ============================================================================

def train_baseline_model(experiment: str = 'LE', save_dir: str = ".", model_name: Optional[str] = None) -> None:
    """
    RESEARCH STEP 1: Train baseline model on MULTIPLE participants' data (cross-participant foundation).
    
    Strategy:
      - Trains on multi-participant baseline dataset (e.g., 18 participants, NOT including target participant)
      - Creates a generalized foundation model for cross-participant transfer learning
      - This pre-trained model's weights are later fine-tuned on the NEW target participant
      - Run this ONCE, then reuse result for all transfer learning experiments (Step 2)
    
    Args:
        experiment: Experiment name ('LE', 'Face', or 'RE')
        save_dir: Directory to save model and history
        model_name: Custom model name (default: "participant_baseline")
    
    Output:
        participant_baseline.keras - Pre-trained cross-participant foundation model
        participant_baseline.json  - Training history
    """
    exp_config = config.get_config(experiment)
    exp_info = exp_config['experiment_info']
    
    if model_name is None:
        model_name = "participant_baseline"
    
    print("\n" + "=" * 70)
    print(f"BASELINE MODEL TRAINING - {exp_info['name'].upper()}")
    print("=" * 70)
    
    # Load data
    train_samples, train_labels = load_images(
        exp_info['baseline_folder'], 
        image_size=exp_config['image_size']
    )
    test_samples, test_labels = load_images(
        exp_info['test_folder'], 
        image_size=exp_config['image_size']
    )
    
    # Create data generators
    datagen = get_augmentation_generator()
    
    train_generator = CustomDataGenerator(
        train_samples, train_labels,
        batch_size=exp_config['batch_size'],
        datagen=datagen
    )
    test_generator = CustomDataGenerator(
        test_samples, test_labels,
        batch_size=exp_config['batch_size']
    )
    
    # Create and train model
    model = create_and_compile_model(
        dense_units=96,
        dropout_rate=exp_config['dropout_rate'],
        learning_rate=exp_config['learning_rate']
    )
    
    callbacks = get_callbacks(
        patience_early_stopping=exp_config['patience_early_stopping'],
        patience_reduce_lr=exp_config['patience_reduce_lr'],
        factor_reduce_lr=exp_config['factor_reduce_lr'],
        min_learning_rate=exp_config['min_learning_rate']
    )
    
    results = train_model(
        model, train_generator, test_generator,
        epochs=exp_config['epochs'],
        callbacks=callbacks,
        model_name=model_name
    )
    
    # Print and save results
    print_training_time(results['time'])
    plot_training_results(results['history'], f"({model_name})")
    
    # Save model and history
    model_path = os.path.join(save_dir, f"{model_name}.keras")
    json_path = os.path.join(save_dir, f"{model_name}.json")
    
    save_model(model, model_path)
    save_history_to_json(results['history'], json_path)


def train_transfer_learning_model(
    experiment: str = 'LE',
    base_model_path: str = None,
    dataset_key: str = '500',
    save_dir: str = ".",
    epochs: Optional[int] = None
) -> None:
    """
    RESEARCH STEP 2: Fine-tune baseline model on target participant (MAIN RESEARCH).
    
    Strategy:
      - Loads pre-trained baseline model (trained on multiple participants, e.g., 18)
      - Fine-tunes on NEW TARGET PARTICIPANT's augmented data with EXTREMELY LIMITED samples
      - Experiments with dataset sizes: 500, 400, 300, 200, 100 samples
      - Test set kept CONSTANT for fair comparison across all sizes
      - PRIMARY EXPERIMENT: Study model behavior, convergence, and accuracy with very limited target participant data
      - Analyzes: How does accuracy change as dataset size decreases? What convergence patterns emerge?
    
    Args:
        experiment: Experiment name ('LE', 'Face', or 'RE')
        base_model_path: Path to baseline model (from Step 1)
        dataset_key: Dataset size to evaluate ('100', '200', '300', '400', '500')
        save_dir: Directory to save model and history
        epochs: Number of epochs (uses config value if None)
    
    Output:
        participant_transfer_500.keras - Fine-tuned with 500 samples
        participant_transfer_400.keras - Fine-tuned with 400 samples
        participant_transfer_300.keras - Fine-tuned with 300 samples
        participant_transfer_200.keras - Fine-tuned with 200 samples
        participant_transfer_100.keras - Fine-tuned with 100 samples
        (and corresponding .json history files)
    
    Note:
        This is your PRIMARY research methodology.
        Results show transfer learning effectiveness across different data quantities.
    """
    from tensorflow.keras.models import load_model as keras_load_model
    
    exp_config = config.get_config(experiment)
    exp_info = exp_config['experiment_info']
    
    if dataset_key not in exp_info['training']:
        raise ValueError(f"Invalid dataset key: {dataset_key}")
    
    if base_model_path is None:
        base_model_path = "participant_baseline.keras"
    
    model_name = f"participant_transfer_{dataset_key}"
    
    print("\n" + "=" * 70)
    print(f"TRANSFER LEARNING - {exp_info['name'].upper()} ({dataset_key} samples)")
    print("=" * 70)
    
    # Load pre-trained model
    model = keras_load_model(base_model_path)
    
    # Load data
    train_samples, train_labels = load_images(
        exp_info['training'][dataset_key],
        image_size=exp_config['image_size']
    )
    test_samples, test_labels = load_images(
        exp_info['test_folder'],
        image_size=exp_config['image_size']
    )
    
    # Create data generators
    train_generator = CustomDataGenerator(
        train_samples, train_labels,
        batch_size=exp_config['batch_size']
    )
    test_generator = CustomDataGenerator(
        test_samples, test_labels,
        batch_size=exp_config['batch_size']
    )
    
    # Train model
    callbacks = get_callbacks(
        patience_early_stopping=exp_config['patience_early_stopping'],
        patience_reduce_lr=exp_config['patience_reduce_lr'],
        factor_reduce_lr=exp_config['factor_reduce_lr'],
        min_learning_rate=exp_config['min_learning_rate']
    )
    
    results = train_model(
        model, train_generator, test_generator,
        epochs=epochs or exp_config['epochs'],
        callbacks=callbacks,
        model_name=model_name
    )
    
    # Print and save results
    print_training_time(results['time'])
    plot_training_results(results['history'], f"({model_name})")
    
    json_path = os.path.join(save_dir, f"{model_name}.json")
    save_history_to_json(results['history'], json_path)


def train_from_scratch_model(
    experiment: str = 'LE',
    dataset_key: str = '500',
    save_dir: str = ".",
    epochs: Optional[int] = None
) -> None:
    """
    RESEARCH STEP 3: Train from scratch on target participant data (COMPARISON BASELINE).
    
    Strategy:
      - Trains completely new models WITHOUT any pre-training
      - Uses NEW TARGET PARTICIPANT's augmented data (same as transfer learning)
      - Same dataset sizes: 500, 400, 300, 200, 100
      - Test set kept CONSTANT (fair comparison)
      - Shows benefit of transfer learning vs. training from scratch
    
    Args:
        experiment: Experiment name ('LE', 'Face', or 'RE')
        dataset_key: Dataset size to train ('100', '200', '300', '400', '500')
        save_dir: Directory to save model and history
        epochs: Number of epochs (uses config value if None)
    
    Output:
        participant_scratch_500.keras - Trained from scratch with 500 samples
        participant_scratch_400.keras - Trained from scratch with 400 samples
        participant_scratch_300.keras - Trained from scratch with 300 samples
        participant_scratch_200.keras - Trained from scratch with 200 samples
        participant_scratch_100.keras - Trained from scratch with 100 samples
        (and corresponding .json history files)
    
    Note:
        Optional comparison baseline. Shows effectiveness of transfer learning.
        Compare participant_transfer_* with participant_scratch_* results.
    """
    exp_config = config.get_config(experiment)
    exp_info = exp_config['experiment_info']
    
    if dataset_key not in exp_info['training']:
        raise ValueError(f"Invalid dataset key: {dataset_key}")
    
    model_name = f"participant_scratch_{dataset_key}"
    
    print("\n" + "=" * 70)
    print(f"FROM SCRATCH MODEL - {exp_info['name'].upper()} ({dataset_key} samples)")
    print("=" * 70)
    
    # Load data
    train_samples, train_labels = load_images(
        exp_info['training'][dataset_key],
        image_size=exp_config['image_size']
    )
    test_samples, test_labels = load_images(
        exp_info['test_folder'],
        image_size=exp_config['image_size']
    )
    
    # Create data generators
    train_generator = CustomDataGenerator(
        train_samples, train_labels,
        batch_size=exp_config['batch_size']
    )
    test_generator = CustomDataGenerator(
        test_samples, test_labels,
        batch_size=exp_config['batch_size']
    )
    
    # Create and train model
    model = create_and_compile_model(
        dense_units=96,
        dropout_rate=exp_config['dropout_rate'],
        learning_rate=exp_config['learning_rate']
    )
    
    callbacks = get_callbacks(
        patience_early_stopping=exp_config['patience_early_stopping'],
        patience_reduce_lr=exp_config['patience_reduce_lr'],
        factor_reduce_lr=exp_config['factor_reduce_lr'],
        min_learning_rate=exp_config['min_learning_rate']
    )
    
    results = train_model(
        model, train_generator, test_generator,
        epochs=epochs or exp_config['epochs'],
        callbacks=callbacks,
        model_name=model_name
    )
    
    # Print and save results
    print_training_time(results['time'])
    plot_training_results(results['history'], f"({model_name})")
    
    json_path = os.path.join(save_dir, f"{model_name}.json")
    save_history_to_json(results['history'], json_path)


# ============================================================================
# Experiment Orchestration Functions
# ============================================================================

def run_complete_experiment(experiment: str = 'LE', save_dir: str = ".") -> None:
    """
    Run COMPLETE EXPERIMENTAL PIPELINE for one gaze region.
    
    Executes all research steps in sequence:
    
    Step 1: BASELINE (Cross-Participant Foundation)
      - Train foundation model on MULTIPLE participants' data (e.g., 18 participants)
      - Output: participant_baseline.keras
    
    Step 2: TRANSFER LEARNING (MAIN RESEARCH - Limited Data Analysis)
      - Fine-tune baseline on NEW TARGET PARTICIPANT with EXTREMELY LOW dataset sizes: 500→100
      - Output: participant_transfer_500.keras through participant_transfer_100.keras
      - PRIMARY ANALYSIS: Model behavior across extremely limited data (100-500 samples)
      - Evaluates: Accuracy trends, convergence patterns, minimum viable dataset size
    
    Step 3: FROM-SCRATCH (COMPARISON)
      - Train new models without pre-training on NEW TARGET PARTICIPANT with same data sizes
      - Output: participant_scratch_500.keras through participant_scratch_100.keras
      - Shows: Benefit of transfer learning vs. training from scratch
    
    Args:
        experiment: Experiment name ('LE', 'Face', or 'RE')
        save_dir: Directory to save all models and histories
    
    Note:
        This runs the complete research workflow for one region.
        Repeat for other regions: LE, Face, RE
    """
    exp_config = config.get_config(experiment)
    exp_info = exp_config['experiment_info']
    dataset_sizes = list(exp_info['training'].keys())
    
    print("\n" + "=" * 70)
    print(f"COMPLETE EXPERIMENT: {exp_info['name'].upper()}")
    print("=" * 70)
    
    # Step 1: Train baseline model
    print("\n[1/3] Training baseline model...")
    train_baseline_model(experiment=experiment, save_dir=save_dir)
    
    # Step 2: Fine-tune with all dataset sizes
    print("\n[2/3] Training transfer learning models...")
    base_model = os.path.join(save_dir, "participant_baseline.keras")
    for dataset_key in dataset_sizes:
        try:
            train_transfer_learning_model(
                experiment=experiment,
                base_model_path=base_model,
                dataset_key=dataset_key,
                save_dir=save_dir
            )
        except Exception as e:
            print(f"Error training transfer learning with {dataset_key} samples: {e}")
    
    # Step 3: Train from scratch with all dataset sizes
    print("\n[3/3] Training models from scratch...")
    for dataset_key in dataset_sizes:
        try:
            train_from_scratch_model(
                experiment=experiment,
                dataset_key=dataset_key,
                save_dir=save_dir
            )
        except Exception as e:
            print(f"Error training from scratch with {dataset_key} samples: {e}")
    
    print("\n" + "=" * 70)
    print(f"Experiment {exp_info['name'].upper()} completed!")
    print("=" * 70)


def run_all_experiments(save_dir: str = ".") -> None:
    """
    Run complete training pipeline for all three experiments.
    
    Executes: LE → Face → Right Eye
    
    Args:
        save_dir: Directory to save all results
    """
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    
    for experiment in ['LE', 'Face', 'RE']:
        try:
            exp_save_dir = os.path.join(save_dir, experiment)
            os.makedirs(exp_save_dir, exist_ok=True)
            run_complete_experiment(experiment=experiment, save_dir=exp_save_dir)
        except Exception as e:
            print(f"Error running {experiment} experiment: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)


# ============================================================================
# Quick Training Functions
# ============================================================================

def quick_baseline_training(experiment: str = 'LE'):
    """
    Quick baseline training.
    
    Trains foundation model on multi-participant data.
    Used by --mode baseline flag.
    """
    print(f"Starting baseline training for {experiment}...")
    train_baseline_model(experiment=experiment)
    print("Baseline training complete!")


def quick_transfer_learning_all(experiment: str = 'LE'):
    """
    Quick transfer learning pipeline.
    
    Trains baseline (if needed) then fine-tunes on all dataset sizes (100-500).
    Main research methodology.
    Used by --mode transfer flag.
    """
    print("Training baseline model...")
    train_baseline_model(experiment=experiment)
    
    base_model = "participant_baseline.keras"
    exp_config = config.get_config(experiment)
    dataset_sizes = list(exp_config['experiment_info']['training'].keys())
    
    print("\nFine-tuning on target participant with varying data sizes...")
    for size in dataset_sizes:
        print(f"\nFine-tuning with {size} samples...")
        try:
            train_transfer_learning_model(
                experiment=experiment,
                base_model_path=base_model,
                dataset_key=size
            )
        except Exception as e:
            print(f"Error training with {size} samples: {e}")


def quick_from_scratch_all(experiment: str = 'LE'):
    """
    Quick from-scratch training pipeline.
    
    Trains new models on all dataset sizes (100-500) without pre-training.
    Comparison baseline to show transfer learning benefit.
    Used by --mode scratch flag.
    """
    print(f"Training from scratch on target participant with varying data sizes...")
    exp_config = config.get_config(experiment)
    dataset_sizes = list(exp_config['experiment_info']['training'].keys())
    
    for size in dataset_sizes:
        print(f"\nTraining from scratch with {size} samples...")
        try:
            train_from_scratch_model(experiment=experiment, dataset_key=size)
        except Exception as e:
            print(f"Error training with {size} samples: {e}")


# ============================================================================
# Analysis and Comparison Functions
# ============================================================================

def compare_training_histories(history_files: List[str]) -> None:
    """
    Compare multiple training histories.
    
    Args:
        history_files: List of JSON history file paths
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for history_file in history_files:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        label = Path(history_file).stem
        epochs = range(1, len(history['loss']) + 1)
        
        axes[0].plot(epochs, history['val_loss'], label=label, linewidth=2)
        axes[1].plot(epochs, history['val_mae'], label=label, linewidth=2)
    
    axes[0].set_title('Validation Loss Comparison', fontsize=14)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title('Validation MAE Comparison', fontsize=14)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_dataset_sizes(experiment: str = 'LE') -> None:
    """
    OPTIONAL: Analyze dataset statistics and structure.
    
    Purpose:
      - Data exploration before training
      - Verify dataset sizes are correct
      - Check label distributions and value ranges
      - Quality assurance for data integrity
    
    Displays statistics for:
      1. Baseline dataset (multi-participant, used for Step 1)
      2. Test dataset (target participant, constant for all experiments)
      3. Training datasets (target participant, sizes 100-500 for Steps 2-3)
    """
    exp_config = config.get_config(experiment)
    exp_info = exp_config['experiment_info']
    
    print("\n" + "=" * 70)
    print(f"DATASET ANALYSIS - {exp_info['name'].upper()}")
    print("=" * 70)
    
    print("\n[1] BASELINE DATASET (Multi-Participant - used for Step 1):")
    try:
        samples, labels = load_images(exp_info['baseline'])
        print(f"  Samples shape: {samples.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label min: {labels.min(axis=0)}")
        print(f"  Label max: {labels.max(axis=0)}")
        print(f"  Label mean: {labels.mean(axis=0)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n[2] TEST DATASET (Target Participant - CONSTANT for all experiments):")
    try:
        samples, labels = load_images(exp_info['test'])
        print(f"  Samples shape: {samples.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label min: {labels.min(axis=0)}")
        print(f"  Label max: {labels.max(axis=0)}")
        print(f"  Label mean: {labels.mean(axis=0)}")
        print(f"\n  NOTE: This test set is used for ALL steps (baseline, transfer, scratch)")
        print(f"        Keeping it constant ensures FAIR COMPARISON")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n[3] TRAINING DATASETS (Target Participant - varies by size):")
    print("    Used for Step 2 (Transfer Learning) and Step 3 (From-Scratch)")
    for size in sorted(exp_info['training'].keys()):
        try:
            samples, labels = load_images(exp_info['training'][size])
            print(f"\n  {size} samples:")
            print(f"    Samples shape: {samples.shape}")
            print(f"    Labels shape: {labels.shape}")
            print(f"    Label min: {labels.min(axis=0)}")
            print(f"    Label max: {labels.max(axis=0)}")
        except Exception as e:
            print(f"    Error: {e}")


def generate_experiment_report(experiment: str = 'LE') -> None:
    """
    RESEARCH STEP 4: Generate comprehensive results report.
    
    Purpose:
      - Summarize all training results
      - Compare all approaches (baseline, transfer, from-scratch)
      - Compare across dataset sizes (500, 400, 300, 200, 100)
      - Show quantitative results for paper
    
    Output:
        participant_report.json - Contains:
          - Final validation losses and MAE for all models
          - Best validation metrics achieved
          - Epochs completed for each model
          - Enables easy comparison between approaches
    """
    exp_config = config.get_config(experiment)
    exp_info = exp_config['experiment_info']
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT REPORT - {exp_info['name'].upper()}")
    print("=" * 70)
    
    # Find all JSON history files
    history_files = list(Path('.').glob('participant_*.json'))
    
    if not history_files:
        print("No history files found!")
        return
    
    report = {
        'experiment': experiment,
        'results': {}
    }
    
    for history_file in history_files:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        final_val_mae = history['val_mae'][-1]
        epochs_completed = len(history['loss'])
        
        report['results'][history_file.stem] = {
            'epochs': epochs_completed,
            'final_train_loss': round(final_train_loss, 6),
            'final_val_loss': round(final_val_loss, 6),
            'final_val_mae': round(final_val_mae, 6),
            'best_val_loss': round(min(history['val_loss']), 6),
            'best_val_mae': round(min(history['val_mae']), 6),
        }
    
    # Save report
    report_file = 'participant_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {report_file}")
    print("\nSummary:")
    for exp_name, metrics in report['results'].items():
        print(f"\n{exp_name}:")
        print(f"  Epochs: {metrics['epochs']}")
        print(f"  Final Val Loss: {metrics['final_val_loss']}")
        print(f"  Best Val MAE: {metrics['best_val_mae']}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Main execution function."""
    print_system_info()
    
    # Train complete experiment for LE
    run_complete_experiment(experiment='LE')
    
    # Optionally run Face and RE experiments
    # run_complete_experiment(experiment='Face')
    # run_complete_experiment(experiment='RE')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gaze Estimation Transfer Learning Experiment Suite",
        epilog="""
Research Strategy:
  Step 1 (BASELINE):      python experiments.py --mode baseline --experiment LE
  Step 2 (TRANSFER):      python experiments.py --mode transfer --experiment LE    [MAIN RESEARCH]
  Step 3 (FROM-SCRATCH):  python experiments.py --mode scratch --experiment LE     [Comparison]
  Step 4 (ANALYSIS):      python experiments.py --mode report --experiment LE

Optional:
  Data exploration:       python experiments.py --mode analyze --experiment LE
  Full pipeline:          python experiments.py --mode full --experiment LE        [All steps at once]
        """
    )
    parser.add_argument(
        '--experiment',
        choices=['LE', 'Face', 'RE', 'all'],
        default='LE',
        help='Gaze region: LE (Left Eye), Face, RE (Right Eye), all'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'baseline', 'transfer', 'scratch', 'analyze', 'report'],
        default='full',
        help="""Training mode:
  - full:     Run all steps (baseline → transfer → scratch)
  - baseline: Step 1 only - Train foundation model on multi-participant data
  - transfer: Step 2 only - Fine-tune on target participant (MAIN research)
  - scratch:  Step 3 only - Train from scratch for comparison
  - analyze:  Optional - Inspect dataset sizes and statistics
  - report:   Step 4 only - Generate summary report from results"""
    )
    parser.add_argument(
        '--save-dir',
        default='.',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    print_system_info()
    
    if args.mode == 'full':
        if args.experiment == 'all':
            run_all_experiments(save_dir=args.save_dir)
        else:
            run_complete_experiment(experiment=args.experiment, save_dir=args.save_dir)
    elif args.mode == 'baseline':
        train_baseline_model(experiment=args.experiment, save_dir=args.save_dir)
    elif args.mode == 'transfer':
        quick_transfer_learning_all(experiment=args.experiment)
    elif args.mode == 'scratch':
        quick_from_scratch_all(experiment=args.experiment)
    elif args.mode == 'analyze':
        analyze_dataset_sizes(experiment=args.experiment)
    elif args.mode == 'report':
        generate_experiment_report(experiment=args.experiment)

