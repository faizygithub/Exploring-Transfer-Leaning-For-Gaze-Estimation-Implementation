# Experiments.py Modes - Research Strategy Guide

## Overview

The `experiments.py` file is the main orchestrator for your gaze estimation transfer learning research. It supports multiple **modes** that correspond directly to your research methodology for studying **model behavior with extremely limited data**.

**Paper:** "Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability"

**Key Research Focus:** Investigating model convergence patterns, accuracy trends, and data efficiency when transfer learning is applied with extremely limited target participant data (100-500 samples).

---

## Research Workflow: 4 Steps

Your research follows a structured 4-step experimental pipeline:

### Step 1: BASELINE (Cross-Participant Foundation Model)
- **Purpose:** Create a foundation model trained on MULTIPLE participants' data (e.g., 18 participants)
- **Data:** Multi-participant baseline dataset (NEW target participant excluded)
- **Why:** Generalized cross-participant model creates better starting weights for transfer learning
- **Output:** `participant_baseline.keras` (pre-trained cross-participant model)
- **Run Once:** Use this baseline for ALL subsequent transfer learning experiments on new participant
- **Command:**
```bash
python experiments.py --mode baseline --experiment LE
python experiments.py --mode baseline --experiment Face
python experiments.py --mode baseline --experiment RE
```

### Step 2: TRANSFER LEARNING (Main Research - Limited Data Analysis on NEW Participant)
- **Purpose:** Study model behavior, convergence, and accuracy when adapting to NEW participant with EXTREMELY LIMITED data
- **Approach:** Fine-tune cross-participant baseline on NEW TARGET PARTICIPANT with progressively smaller datasets
- **Data Sizes:** 500, 400, 300, 200, 100 samples (increasingly limited scenarios for NEW participant)
- **Test Set:** CONSTANT for NEW participant across all sizes (critical for fair evaluation and trend analysis)
- **Primary Research Questions:**
  - How quickly does transfer learning adapt the baseline to the NEW participant?
  - How does model accuracy change as target participant dataset decreases from 500→100?
  - What are the convergence patterns with extremely limited target participant data?
  - How does model behavior differ across different target participant data quantities?
  - What's the minimum viable dataset size for acceptable performance on NEW participant?
- **Key Insight:** Transfer learning allows meaningful performance on NEW participant even with just 100 samples!
- **Output:** 5 models per region
  - `participant_transfer_500.keras` (baseline fine-tuned on 500 samples from NEW participant)
  - `participant_transfer_400.keras` (baseline fine-tuned on 400 samples from NEW participant)
  - `participant_transfer_300.keras` (baseline fine-tuned on 300 samples from NEW participant)
  - `participant_transfer_200.keras` (baseline fine-tuned on 200 samples from NEW participant)
  - `participant_transfer_100.keras` (baseline fine-tuned on 100 samples from NEW participant)
- **Command (MAIN RESEARCH):**
```bash
python experiments.py --mode transfer --experiment LE    # Primary experiments
python experiments.py --mode transfer --experiment Face
python experiments.py --mode transfer --experiment RE
```

### Step 3: FROM-SCRATCH (Comparison Baseline - Training on NEW Participant)
- **Purpose:** Show why transfer learning is better than training from scratch on NEW participant
- **Approach:** Train completely new models WITHOUT pre-training on NEW TARGET PARTICIPANT data
- **Data Sizes:** Same as transfer (500 → 100) for NEW participant
- **Test Set:** Same constant test set as transfer
- **Comparison:** Transfer learning results vs From-Scratch results on NEW participant
- **Output:** 5 models per region
  - `participant_scratch_500.keras` (trained from scratch on 500 NEW participant samples)
  - `participant_scratch_400.keras` (trained from scratch on 400 NEW participant samples)
  - `participant_scratch_300.keras` (trained from scratch on 300 NEW participant samples)
  - `participant_scratch_200.keras` (trained from scratch on 200 NEW participant samples)
  - `participant_scratch_100.keras` (trained from scratch on 100 NEW participant samples)
- **Command (Optional Comparison):**
```bash
python experiments.py --mode scratch --experiment LE
python experiments.py --mode scratch --experiment Face
python experiments.py --mode scratch --experiment RE
```

### Step 4: REPORT (Generate Results)
- **Purpose:** Summarize all results into a single report
- **Output:** `participant_report.json` with final metrics
- **Use:** For paper tables, figures, and statistical analysis
- **Command:**
```bash
python experiments.py --mode report --experiment LE
python experiments.py --mode report --experiment Face
python experiments.py --mode report --experiment RE
```

---

## Command-Line Modes Explained

### 1. `--mode full` (All Steps At Once)
Runs the complete pipeline: baseline → transfer → from-scratch

**Best For:** Complete experimental run from beginning to end

```bash
# One region
python experiments.py --mode full --experiment LE

# All regions
python experiments.py --mode full --experiment all
```

### 2. `--mode baseline` (Step 1 Only)
Trains cross-participant foundation model on MULTIPLE participants' data (e.g., 18 participants)

**Best For:** Creating/updating the baseline before transfer learning on NEW participant

```bash
python experiments.py --mode baseline --experiment LE
```

**Output:** 
- `participant_baseline.keras` (cross-participant model)
- `participant_baseline.json`

### 3. `--mode transfer` (Step 2 Only)
Fine-tunes cross-participant baseline on NEW target participant with all dataset sizes

**Best For:** Main research - evaluating transfer learning effectiveness on NEW participant with limited data

```bash
python experiments.py --mode transfer --experiment LE
```

**Workflow:** 
1. Loads cross-participant baseline model (or trains it if missing)
2. Fine-tunes on each dataset size from NEW participant (500→100)

**Output:** 5 models + 5 JSON history files (all fine-tuned for NEW participant)

### 4. `--mode scratch` (Step 3 Only)
Trains new models from scratch on NEW target participant with all dataset sizes

**Best For:** Comparison - shows transfer learning benefit vs training from scratch on NEW participant

```bash
python experiments.py --mode scratch --experiment LE
```

**Output:** 5 models + 5 JSON history files (all trained on NEW participant from scratch)

### 5. `--mode analyze` (Optional Data Exploration)
Inspects dataset structure and statistics

**Best For:** 
- Verifying data was loaded correctly
- Checking label distributions
- Quality assurance before training

```bash
python experiments.py --mode analyze --experiment LE
```

**Output:**
- Baseline dataset shape and statistics
- Test dataset shape and statistics (reminder: kept constant)
- Training datasets (all 5 sizes) - shapes and statistics

### 6. `--mode report` (Step 4 Only)
Generates comprehensive results report

**Best For:** Creating summary of all experiments for paper

```bash
python experiments.py --mode report --experiment LE
```

**Output:** `participant_report.json`

---

## Three Gaze Regions

Your research applies the same methodology to three regions:

1. **LE (Left Eye)**
   - Gaze direction from left eye
   - Related directory: `LEWithoutOla` (training data)

2. **Face**
   - Face-based gaze features
   - Related directory: `FaceGeometryAdaptation` (training data)

3. **RE (Right Eye)**
   - Gaze direction from right eye
   - Related directory: `REWithoutOla` (training data)

---

## Typical Experimental Workflow

### Quick Start (All Regions, Full Pipeline)
```bash
python experiments.py --mode full --experiment all
```

### Recommended Sequence (Step-by-Step Control)

**Day 1 - Setup Baselines:**
```bash
python experiments.py --mode baseline --experiment LE
python experiments.py --mode baseline --experiment Face
python experiments.py --mode baseline --experiment RE
```

**Day 2-3 - Main Research (Transfer Learning):**
```bash
python experiments.py --mode transfer --experiment LE
python experiments.py --mode transfer --experiment Face
python experiments.py --mode transfer --experiment RE
```

**Day 4 - Comparison Baseline (Optional):**
```bash
python experiments.py --mode scratch --experiment LE
python experiments.py --mode scratch --experiment Face
python experiments.py --mode scratch --experiment RE
```

**Day 5 - Generate Reports:**
```bash
python experiments.py --mode report --experiment LE
python experiments.py --mode report --experiment Face
python experiments.py --mode report --experiment RE
```

---

## Data Flow & Fair Comparison

### Critical Point: Constant Test Set

Your research keeps the **test set CONSTANT** across:
- All baseline experiments
- All transfer learning experiments (500→100 sizes)
- All from-scratch experiments (500→100 sizes)

This ensures **fair comparison** - performance differences are due to:
- Amount of training data (500 vs 100)
- Approach (transfer vs from-scratch)
- NOT differences in evaluation criteria

### Dataset Structure (Per Region)

```
config.py defines:
├── baseline_folder/          ← MULTIPLE participants (e.g., 18 participants) - Cross-Participant Foundation
│                               Used in: Step 1 only
├── test_folder/              ← NEW TARGET PARTICIPANT (CONSTANT across all experiments)
│                               Used in: All steps (Step 2 transfer & Step 3 scratch)
└── training/                 ← NEW TARGET PARTICIPANT training data
    ├── 500_folder/           ← 500 samples from NEW participant
    ├── 400_folder/           ← 400 samples from NEW participant
    ├── 300_folder/           ← 300 samples from NEW participant
    ├── 200_folder/           ← 200 samples from NEW participant
    └── 100_folder/           ← 100 samples from NEW participant
                                Used in: Step 2 (transfer finetune) & Step 3 (scratch training)
```

**Key Point:** 
- **Baseline**: Trained ONCE on multi-participant data (e.g., 18 participants)
- **Transfer Learning**: Fine-tunes baseline on NEW participant with limited data sizes
- **From-Scratch**: Trains new models on NEW participant with same limited data sizes
- **Test**: Always evaluated on constant NEW participant test set

---

## Output Files Organization

After running experiments, you'll have:

```
./ (project root)
├── participant_baseline.keras           ← Step 1: Foundation model
├── participant_baseline.json            ← Step 1: Training history
├── participant_transfer_500.keras       ← Step 2: Fine-tuned (500 samples)
├── participant_transfer_500.json        ← Step 2: History
├── participant_transfer_400.keras       ← Step 2: Fine-tuned (400 samples)
├── participant_transfer_400.json
├── ... (300, 200, 100 similarly)
├── participant_scratch_500.keras        ← Step 3: From-scratch (500 samples)
├── participant_scratch_500.json         ← Step 3: History
├── ... (400, 300, 200, 100 similarly)
└── participant_report.json              ← Step 4: Summary report
```

---

## Research Questions Answered

### Question 1: How Does Model Behavior Change With Limited Data?
**Analyze:** `participant_transfer_500` → `participant_transfer_100`

- Accuracy trends: How much does performance degrade with 100 vs 500 samples?
- Convergence patterns: Do models learn faster/slower with different data quantities?
- Training dynamics: What do loss curves reveal about learning with minimal data?

### Question 2: What's the Minimum Viable Dataset Size?
**Compare:** `participant_transfer_100`, `participant_transfer_200`, etc.

Find the minimum data needed for acceptable performance. Can you use just 100-200 samples?

### Question 3: Why Does Transfer Learning Help With Limited Data?
**Compare:** `participant_transfer_*` vs `participant_scratch_*`

Transfer learning advantages are most pronounced with extremely limited data (100-200 samples).

### Question 4: Consistency Across Regions?
**Compare:** LE vs Face vs RE limited-data learning curves

Do accuracy trends and convergence patterns differ across gaze regions when data is extremely limited?

---

## Usage Tips

### Skip Baseline Retraining
After running Step 1 (baseline), you have `participant_baseline.keras`

Next time, running `--mode transfer` will:
1. Check if baseline exists
2. Use existing baseline (no retraining)
3. Go directly to fine-tuning

### Combine Modes Efficiently
```bash
# Option 1: Run everything sequentially
python experiments.py --mode full --experiment all

# Option 2: More control
python experiments.py --mode baseline --experiment all
python experiments.py --mode transfer --experiment all
python experiments.py --mode scratch --experiment all
python experiments.py --mode report --experiment all
```

### Debug Mode: Analyze First
```bash
python experiments.py --mode analyze --experiment LE
# Check data is correct, then run training
python experiments.py --mode transfer --experiment LE
```

---

## Configuration

All hyperparameters are defined in `config.py`:

```python
{
    'batch_size': 16,
    'epochs': 200,
    'learning_rate': 0.0001,
    'optimizer': 'Adam',
    'loss': 'mae',  # Mean Absolute Error
    'early_stopping_patience': 15,
    'reduce_lr_patience': 10,
    'factor_reduce_lr': 0.5,
    'min_learning_rate': 0.00001
}
```

These apply uniformly to:
- Baseline training (Step 1)
- Transfer learning fine-tuning (Step 2)
- From-scratch training (Step 3)

---

## Summary

| Mode | Steps | Purpose | Use When |
|------|-------|---------|----------|
| `full` | 1-3 | Complete pipeline | Initial experiments |
| `baseline` | 1 | Create foundation | Once, at start |
| `transfer` | 2 | MAIN RESEARCH | Always - primary methodology |
| `scratch` | 3 | Comparison | Show transfer value |
| `analyze` | - | Data exploration | Debug data issues |
| `report` | 4 | Results summary | After all training |

**Remember:** Your primary research is in **`--mode transfer`** - this mode evaluates how transfer learning scales with target participant data quantity.
