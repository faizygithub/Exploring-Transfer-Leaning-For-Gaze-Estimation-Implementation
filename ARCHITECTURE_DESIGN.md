# Architecture Design & Research Strategy

## Document Purpose
This document explains:
1. **Configuration Design**: Why config is structured as it is
2. **Research Strategy**: How the code implements your transfer learning approach
3. **Module Roles**: What each Python module does and why
4. **Running Experiments**: How to execute your research with different modes

---

## 1. RESEARCH STRATEGY

### Overview
Your research: **"Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability"**

### Multi-Participant Transfer Learning Approach

```
MULTIPLE Participants (e.g., 18 participants) → Baseline Training
│
├─→ [Baseline Model] 
│   └─ Trained on 18 participants' data (cross-participant foundation)
│   └ No fine-tuning at this stage, creates generalized model
│
├─→ [Transfer Learning on NEW Participant] ⭐ MAIN RESEARCH
│   ├─ Load baseline model (trained on 18 participants)
│   ├─ Fine-tune on NEW TARGET PARTICIPANT with limited data:
│   │  ├─ 500 NEW participant samples (largest)
│   │  ├─ 400 NEW participant samples
│   │  ├─ 300 NEW participant samples
│   │  ├─ 200 NEW participant samples
│   │  └─ 100 NEW participant samples (smallest)
│   └─ Test set: CONSTANT for NEW participant (fair comparison)
│   └─ Primary question: How well does baseline adapt to NEW participant?
│
└─→ [From-Scratch Comparison]
    └─ Train new model from scratch on NEW participant with same sample counts
    └ No pre-training, comparison to show transfer learning benefit
    └ Shows: Transfer (with baseline) vs No-Transfer (from scratch) on NEW participant
```

### Cross-Participant Transfer Learning Flow
1. **Step 1**: Train ONCE on MULTIPLE participants (e.g., 18) → `participant_baseline.keras`
2. **Step 2**: Fine-tune baseline on NEW participant (19th, not in baseline) with 100-500 samples → Produces 5 transfer models  
3. **Step 3**: Train from scratch on NEW participant (same data) → Produces 5 from-scratch models
4. **Step 4**: Compare transfer vs from-scratch results on NEW participant

### Three Gaze Regions Evaluated
1. **LE (Left Eye)**: Gaze features from left eye region
2. **Face**: Face-based gaze features and head pose context
3. **RE (Right Eye)**: Gaze features from right eye region

### Key Research Aspects
- **Cross-Participant Baseline**: Foundation model trained on 18 participants (generalized model)
- **NEW Participant Adaptation**: How well baseline transfers to unseen 19th participant
- **Data Efficiency Study**: Evaluate 500→100 NEW participant samples to quantify data requirements
- **Fair Comparison**: Test set CONSTANT for NEW participant across all dataset sizes
- **Transfer vs From-Scratch**: Shows benefit of pre-training over random initialization on NEW participant
- **Multi-Region**: Same strategy applied to three different gaze regions (LE, Face, RE)

---

## 2. CONFIGURATION DESIGN

### Question: Why is config in multiple places?

**Short Answer**: It's NOT. Config is defined ONCE in `config.py`, but it's used FLEXIBLY throughout:

### Architecture Diagram

```
config.py (SINGLE SOURCE OF TRUTH)
  ├─ GLOBAL SETTINGS: paths, hyperparameters
  ├─ EXPERIMENTS dict: experiment-specific paths
  └─ get_config(experiment) → returns experiment config
         │
         ├─→ experiments.py uses it: config.get_config('LE')
         ├─→ data_loader.py imports it when needed
         └─→ model_architecture.py uses defaults + config values
```

### Module Structure & Responsibilities

#### **config.py** (Configuration)
- **Single source of truth** for all settings
- Defines dataset paths (baseline, test, training with 5 size variants)
- Defines hyperparameters (batch_size, epochs, learning_rate, etc.)
- Provides `get_config(experiment)` helper function
- **Never depends on other modules** (pure config)

```python
# Example usage:
config_le = get_config('LE')  # Returns dict with LE-specific paths & hyperparameters
config_face = get_config('Face')  # Returns dict with Face-specific paths
```

#### **data_loader.py** (Data Handling)
- Functions with **FLEXIBLE parameters**:
  - `load_images(directory, image_size=64)` - takes image_size as parameter, not from config
  - `CustomDataGenerator(samples, labels, batch_size, ...)` - takes batch_size as parameter
- Functions work **independently** of global config
- Can be imported and used with custom parameters

```python
# Flexible approach - works with any config:
samples, labels = load_images(
    exp_info['training']['500'],  # From config
    image_size=config['image_size']  # From config
)

# But could also be called directly:
samples, labels = load_images('/any/path', image_size=128)  # Works without config!
```

#### **model_architecture.py** (Model Definition)
- Defines CNN architecture
- Takes hyperparameters as arguments: `create_and_compile_model(input_shape, dense_units, dropout_rate, learning_rate)`
- Doesn't import config - pure functionality

#### **experiments.py** (Experiment Orchestration)
- Imports config to get experiment paths and hyperparameters
- Calls data_loader functions with config values
- Manages the experimental workflow

---

## 3. CONFIG DICTIONARY STRUCTURE

### What `get_config('LE')` Returns

```python
{
    'experiment': 'LE',
    'experiment_info': {
        'name': 'Left Eye (LE) Gaze Estimation',
        'baseline': '/path/to/baseline/data',      # Multi-participant data
        'test': '/path/to/test/data',               # Target participant test
        'training': {                                # Target participant train
            '100': '/path/folder_100',
            '200': '/path/folder_200',
            '300': '/path/folder_300',
            '400': '/path/folder_400',
            '500': '/path/folder_500',
        }
    },
    'image_size': 64,
    'batch_size': 16,
    'epochs': 200,
    'learning_rate': 0.0001,
    'dropout_rate': 0.2,
    'dense_units': 96,
    'patience_early_stopping': 10,
    'patience_reduce_lr': 5,
    'factor_reduce_lr': 0.2,
    'min_learning_rate': 1e-5,
}
```

### Data Paths Explained

**For Cross-Participant Transfer Learning (Your Main Approach):**

1. **`baseline`** - Multi-participant baseline data (e.g., 18 participants)
   - Used to train the cross-participant BASE model
   - Contains data from MULTIPLE participants (NOT including NEW target participant)
   - Creates generalized foundation model
   - No augmentation at this stage
   
2. **`test`** - NEW target participant test data
   - FIXED evaluation set for NEW target participant
   - SAME test set across all experiments (fair comparison!)
   - Used in ALL training runs for consistent evaluation of NEW participant
   - Participant #19, separate from baseline (#1-18)

3. **`training[size]`** - NEW target participant training data
   - Variable sizes: 100, 200, 300, 400, 500 samples from NEW participant
   - Used for fine-tuning the baseline model on NEW participant
   - Different amounts to test data efficiency for NEW participant adaptation
   - Same NEW participant test set maintained for all sizes

---

## 4. EXPERIMENT MODES

Your `experiments.py` supports different running modes:

### Mode: `baseline` (Step 1 - Cross-Participant Foundation)
```
Train cross-participant baseline model from MULTIPLE participants (e.g., 18 participants)
├─ Input: config['experiment_info']['baseline'] (multi-participant data)
├─ Process: Train from scratch (creates generalized model)
└─ Output: participant_baseline.keras (cross-participant model)
```

**When to use**: Once at the beginning to create cross-participant base model for NEW participant transfer learning

### Mode: `transfer` (Step 2 - MAIN RESEARCH: NEW Participant Adaptation)
```
Fine-tune cross-participant baseline model on NEW TARGET PARTICIPANT with varying data sizes
├─ Input: 
│  ├─ participant_baseline.keras (cross-participant pre-trained)
│  ├─ config['experiment_info']['training'][size] (NEW participant: 100-500 samples)
│  └─ config['experiment_info']['test'] (NEW participant: constant)
├─ Process: Load baseline → fine-tune on NEW participant for each data size
└─ Output: participant_transfer_100.keras through participant_transfer_500.keras
          (all fine-tuned for NEW participant)
```

**When to use**: Main experiments - evaluate how well baseline transfers to NEW participant with different data quantities

### Mode: `scratch` (Step 3 - Comparison: From-Scratch on NEW Participant)
```
Train from scratch on NEW TARGET PARTICIPANT data (no pre-training, comparison baseline)
├─ Input:
│  ├─ config['experiment_info']['training'][size] (NEW participant: 100-500 samples)
│  └─ config['experiment_info']['test'] (NEW participant: constant)
├─ Process: Random initialization → full training on NEW participant
└─ Output: participant_scratch_100.keras through participant_scratch_500.keras
          (all trained from scratch on NEW participant)
```

**When to use**: Comparison baseline - shows benefit of transfer learning vs. training from scratch

### Mode: `analyze`
```
Analyze dataset statistics and structure
├─ Input: All dataset paths
├─ Process: Calculate sizes, label distributions, etc.
└─ Output: Statistics printed to console or JSON report
```

**When to use**: Data exploration and quality assurance

### Mode: `report`
```
Generate experiment results and comparison report
├─ Input: All trained models and their histories
├─ Process: Compile metrics, create comparisons
└─ Output: participant_report.json (structured results)
```

**When to use**: After experiments complete, to summarize findings

---

## 5. WHY THIS DESIGN?

### Benefits of Configuration Approach

1. **Single Source of Truth**
   - Change paths once in config.py
   - All experiments use new paths automatically
   - No scattered configuration

2. **Experiment Flexibility**
   - Easy to add new regions: just add to EXPERIMENTS dict
   - Easy to change dataset sizes: modify TRAINING_DATASETS
   - Easy to swap paths for different data

3. **Code Reusability**
   - data_loader functions work with any config
   - model_architecture is framework-agnostic
   - training_utils works with any model

4. **Research Clarity**
   - Comments explain the multi-participant strategy
   - Paths clearly labeled: baseline vs test vs training
   - Variable sizes documented for data efficiency study

5. **Reproducibility**
   - All hyperparameters in one place
   - RANDOM_SEED for consistency
   - Easy to document what runs on what settings

---

## 6. RUNNING YOUR EXPERIMENTS

### Step 1: Configure Paths
Edit `config.py`:
```python
LE_BASELINE_DATASET = r'C:\your\path\to\baseline'  # Multi-participant data (18 participants)
LE_TEST_DATASET = r'C:\your\path\to\test'          # NEW target participant test (e.g., 19th participant)
LE_TRAINING_DATASETS = {                             # NEW target participant training
    '500': r'C:\your\path\to\train_500',           # 500 samples from NEW participant
    '400': r'C:\your\path\to\train_400',           # 400 samples from NEW participant
    '300': r'C:\your\path\to\train_300',           # etc...
    '200': r'C:\your\path\to\train_200',
    '100': r'C:\your\path\to\train_100',
}
```

### Step 2: Run Baseline Once (Cross-Participant Foundation)
```bash
python experiments.py
# Select: baseline mode, LE experiment
# This trains ONE cross-participant baseline model
# Output: participant_baseline.keras (trained on 18 participants)
# Run this ONCE, reuse for all transfer experiments
```

### Step 3: Run Transfer Learning (Main Research - NEW Participant Adaptation)
```bash
python experiments.py
# Select: transfer mode, LE experiment
# Automatically evaluates all dataset sizes: 500, 400, 300, 200, 100
# Output: participant_transfer_500.keras through participant_transfer_100.keras
```

### Step 4: Comparison (Optional)
```bash
python experiments.py
# Select: scratch mode, LE experiment
# Output: participant_scratch_500.keras through participant_scratch_100.keras
```

### Step 5: Generate Report
```bash
python experiments.py
# Select: report mode
# Output: participant_report.json (all results summarized)
```

### Repeat for Other Regions
```bash
# Face region
python experiments.py  # Select: transfer, Face

# Right Eye region
python experiments.py  # Select: transfer, RE
```

---

## 7. KEY FILES EXPLAINED

| File | Purpose | Imports Config? |
|------|---------|-----------------|
| `config.py` | Central settings, paths, hyperparameters | No (is the config) |
| `data_loader.py` | Load images, create generators | Only if needed |
| `model_architecture.py` | Define CNN architecture | No |
| `training_utils.py` | Training loop, callbacks, visualization | No |
| `system_monitoring.py` | GPU/RAM monitoring | No |
| `experiments.py` | Orchestrate experiments, main entry point | Yes (uses config.get_config) |
| `LE_Training_Clean.ipynb` | Interactive notebook for LE | Uses config directly |
| `Face_Training_Clean.ipynb` | Interactive notebook for Face | Uses config directly |
| `RE_Training_Clean.ipynb` | Interactive notebook for RE | Uses config directly |

---

## 8. NAMING CONVENTION

All saved models use **generic privacy-focused names**:

- `participant_baseline.keras` → Baseline model (multi-participant)
- `participant_transfer_500.keras` → Transfer learning with 500 samples
- `participant_transfer_400.keras` → Transfer learning with 400 samples
- `participant_transfer_300.keras` → Transfer learning with 300 samples
- `participant_transfer_200.keras` → Transfer learning with 200 samples
- `participant_transfer_100.keras` → Transfer learning with 100 samples
- `participant_scratch_500.keras` → From-scratch with 500 samples (comparison)
- `participant_report.json` → Results summary

The actual experiment region (LE, Face, RE) is tracked in `config.py`, not in filenames.

---

## 9. SUMMARY

**Your Research Implementation:**
- ✅ Multi-participant baseline training (all-except-one)
- ✅ Transfer learning on target participant with 5 data sizes
- ✅ Constant test set for fair comparison
- ✅ Three gaze regions (LE, Face, RE)
- ✅ Privacy-first generic model naming
- ✅ Clear configuration management
- ✅ Flexible, reusable modules

**How Modes Work:**
- `baseline` → Train multi-participant base model once
- `transfer` → Fine-tune on target with varying data (MAIN)
- `scratch` → Train from scratch for comparison
- `analyze` → Explore data structure
- `report` → Summarize all results

**Configuration Design:**
- Single source of truth in `config.py`
- Flexible functions in other modules that use config values
- Clear separation of concerns
- Easy to reproduce and extend

---

**Questions?** Refer to README.md, CONTRIBUTING.md, and NAMING_CONVENTION.md for additional details.
