# Quick Start Guide

Get started with the Gaze Estimation Transfer Learning experimental suite in 5 minutes!

**Research**: *Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability*

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gaze-estimation-transfer-learning.git
cd gaze-estimation-transfer-learning

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 2. Select an Experiment

Choose which gaze region to train on by editing `config.py`:

```python
# config.py - Select your experiment
CURRENT_EXPERIMENT = 'LE'    # Options: 'LE' (Left Eye), 'Face', 'RE' (Right Eye)

# The configuration automatically loads the correct dataset paths
# based on your selection
```

## 3. Check Your Environment

```bash
python -c "from system_monitoring import print_system_info; print_system_info()"
```

## 4. Run Your First Experiment

```bash
# Train baseline model for Left Eye
python experiments.py --mode baseline

# Or use interactive menu
python experiments.py
```

## 5. View Results

Results saved with generic naming:
- Model: `participant_baseline.keras`
- History: `participant_baseline.json`
- Transfer models: `participant_transfer_*.keras`, `participant_transfer_*.json`
- Plots: Displayed automatically during training

---

## Quick Commands

```bash
# Interactive menu
python experiments.py

# Analyze datasets
python experiments.py --mode analyze

# Generate report
python experiments.py --mode report
```

---

## Citation

```bibtex
@article{gaze_transfer_learning_2026,
  title={Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability},
  author={[Your Name]},
  year={2026},
  journal={[Journal Name]},
}
```
