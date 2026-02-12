# Privacy-First Naming Convention

*For: Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability*

This project uses **generic naming for all model files** to protect research methodology and data when uploading to GitHub or sharing publicly.

## Why Generic Naming?

Hardcoded experiment names (like "LE_WithoutOla", "Face_Baseline", etc.) could unintentionally reveal:
- Specific experimental regions or features being studied
- Research methodology details before publication
- Identifying information about gaze estimation approach

Using generic "participant" naming keeps your local work flexible and GitHub-safe.

## Naming Scheme

### File Names

| Component | Format | Example |
|-----------|--------|---------|
| **Baseline** | `participant_baseline.{keras,json}` | `participant_baseline.keras` |
| **Transfer Learning** | `participant_transfer_{SIZE}.{keras,json}` | `participant_transfer_500.keras` |
| **From-Scratch** | `participant_scratch_{SIZE}.{keras,json}` | `participant_scratch_100.keras` |
| **Report** | `participant_report.json` | `participant_report.json` |

### Size Notation

- `100` = 100 training samples
- `200` = 200 training samples
- `300` = 300 training samples
- `400` = 400 training samples
- `500` = 500 training samples

### Complete File Set Example

After running complete training:
```
participant_baseline.keras
participant_baseline.json

participant_transfer_500.keras
participant_transfer_500.json
participant_transfer_400.keras
participant_transfer_400.json
participant_transfer_300.keras
participant_transfer_300.json
participant_transfer_200.keras
participant_transfer_200.json
participant_transfer_100.keras
participant_transfer_100.json

participant_scratch_500.keras
participant_scratch_500.json
participant_scratch_400.keras
participant_scratch_400.json
participant_scratch_300.keras
participant_scratch_300.json
participant_scratch_200.keras
participant_scratch_200.json
participant_scratch_100.keras
participant_scratch_100.json

participant_report.json
```

## Mapping Files Locally

The local `config.py` file tracks which gaze estimation region each model corresponds to:

```python
EXPERIMENTS = {
    'LE': {'name': 'Left Eye', 'description': 'Gaze features from left eye region', ...},
    'Face': {'name': 'Face', 'description': 'Face-based gaze features and head pose', ...},
    'RE': {'name': 'Right Eye', 'description': 'Gaze features from right eye region', ...},
}
```

So locally, you know which region you're training on, but the saved files use generic naming.

## .gitignore Configuration

Important: Keep data and personal identifiers out of version control:

```
*.keras          # Model files
*.json           # History files
data/            # Dataset directories
local_config.py  # Any personal configuration
```

The generic naming works with this structure to ensure privacy.

## When Sharing Results

When sharing results in papers or publicly:

1. **Report generation**: Use `participant_report.json` which contains only quantitative metrics
2. **Model comparison**: Compare models with generic filenames while publishing results separately
3. **Publication reference**: The paper *Exploring Transfer Learning for Gaze Estimation: A Study on Model Adaptability* provides context for all gaze estimation experiments
4. **Documentation**: Reference gaze regions by their full names (Left Eye, Face, Right Eye) in published materials

## Customization

If you need different naming, modify these functions in `experiments.py`:

```python
# Currently uses generic names
model_name = f"participant_transfer_{dataset_key}"

# Could become anything by editing these two functions:
def train_transfer_learning_model(...)
def train_from_scratch_model(...)
```

## For Developers

When contributing new training functions:
- Use generic model naming
- Avoid embedding experiment-specific identifiers in filenames
- Document experiment type in config.py instead

## FAQ

**Q: How do I know which experiment is which?**  
A: The configuration in `config.py` tracks experiment types locally. Your local working directories can be organized by experiment name.

**Q: Can I add my own metadata?**  
A: Yes! Modify JSON history files to include metadata before uploading, or create a separate `metadata.json` file.

**Q: What about the old experiment names?**  
A: They still appear in console output and configuration for clarity during development. Only the saved model files use generic names.

---

This approach balances privacy with functionality, allowing you to:
- ✅ Share code on GitHub safely
- ✅ Maintain local clarity about your work
- ✅ Publish results without revealing methodological details
- ✅ Keep data private while sharing models
