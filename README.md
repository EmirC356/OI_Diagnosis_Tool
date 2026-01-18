# OI-Pred: Osteogenesis Imperfecta Variant Pathogenicity Predictor

**A Disease-Specific Machine Learning Tool for COL1A1/COL1A2 Variant Interpretation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

OI-Pred is a machine learning-based tool that predicts the pathogenicity of genetic variants in COL1A1 and COL1A2 genes associated with Osteogenesis Imperfecta (brittle bone disease). By incorporating disease-specific features like glycine substitutions in the collagen triple helix, OI-Pred outperforms generic prediction tools.

## Key Results

| Metric | OI-Pred (RF) | SIFT | Improvement |
|--------|--------------|------|-------------|
| Accuracy | 97.3% | 94.2% | +3.3% |
| Specificity | 99.9% | 46.7% | +114% |
| MCC | 0.979 | 0.614 | +59% |
| ROC-AUC | 98.9% | 78.9% | +25% |

*Comparison on same test set of 154 COL1A1/COL1A2 missense variants*

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/oi-pred.git
cd oi-pred

# Install dependencies
pip install -r requirements.txt

# Train the model (if not already trained)
python src/train.py
```

### Predict a Single Variant

```bash
# Using variant string
python predict.py "COL1A1 p.Gly992Ser"

# Using explicit parameters
python predict.py --gene COL1A1 --protein G992S --consequence missense
```

**Output:**
```
============================================================
OI-Pred Prediction Result
============================================================
  Gene:            COL1A1
  Protein Change:  G992S
  Consequence:     missense
------------------------------------------------------------
  Prediction:      Pathogenic
  Probability:     98.4% pathogenic
  Interpretation:  Likely Pathogenic (High Confidence)
============================================================
```

### Batch Prediction

```bash
# From CSV file
python predict.py --file variants.csv --output predictions.csv
```

Input CSV format:
```csv
Gene,Protein_change,Consequence
COL1A1,G992S,missense
COL1A2,G259R,missense
COL1A1,D1413G,missense
```

### Interactive Mode

```bash
python predict.py --interactive
```

## Project Structure

```
oi-pred/
├── predict.py              # Main prediction script (USER INTERFACE)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── src/                    # Source code
│   ├── config.py           # Configuration and constants
│   ├── feature_engineering.py  # Feature extraction
│   └── train.py            # Model training
│
├── models/                 # Trained models
│   ├── oi_pred_rf_model.pkl    # Random Forest model
│   └── feature_list.json       # Feature names
│
├── data/                   # Data files
│   ├── cleaned_COL1_variants.csv   # Processed variant data
│   └── feature_matrix.csv          # Extracted features
│
├── 06_results/             # Analysis results
│   ├── sota_benchmark_results.csv
│   ├── sota_benchmark_comparison.png
│   └── external_validation_results.png
│
└── 07_documentation/       # Documentation
    ├── FEATURE_INTERPRETABILITY.md
    └── COMPREHENSIVE_PROJECT_REPORT.md
```

## Features Used

OI-Pred uses 25 carefully selected features:

| Category | Features | Importance |
|----------|----------|------------|
| **Risk Indicators** | low_risk_consequence, high_risk_consequence | 47.3% |
| **Molecular Consequence** | is_intron, is_synonymous, is_missense, etc. | 21.5% |
| **Collagen-Specific** | glycine_substitution | 6.1% |
| **Biochemical** | size_change, flexibility_change, hydrophobic_change | 12.7% |
| **Positional** | normalized_position | 3.9% |

The **glycine_substitution** feature captures the biological principle that glycine must occupy every third position in the collagen triple helix - substitutions disrupt helix formation and cause OI.

## Model Performance

### Cross-Validation Results (5-Fold)
```
Accuracy:  97.26% +/- 0.75%
Precision: 98.36% +/- 0.53%
Recall:    96.55% +/- 1.04%
F1-Score:  97.45% +/- 0.71%
ROC-AUC:   98.91% +/- 0.26%
MCC:       0.9788
```

### External Validation
- Holdout set (20%): 97.7% accuracy
- Cross-gene validation: 97.6% (COL1A1 to COL1A2)

## Why OI-Pred Outperforms Generic Tools

1. **Disease-Specific Features**: Encodes glycine substitutions critical for collagen structure
2. **Trained on OI Data**: Learns patterns specific to COL1A1/COL1A2 variants
3. **Collagen Biochemistry**: Incorporates amino acid property changes relevant to triple helix
4. **Higher Specificity**: Reduces false positives (99.9% vs 46.7% for SIFT)

## Interpretation Guide

| Probability | Interpretation |
|-------------|----------------|
| >= 90% | Likely Pathogenic (High Confidence) |
| 70-90% | Likely Pathogenic |
| 30-70% | Uncertain Significance |
| 10-30% | Likely Benign |
| < 10% | Likely Benign (High Confidence) |

## Citation

If you use OI-Pred in your research, please cite:

```bibtex
@software{oipred2025,
  author = {Ceylan, Emir},
  title = {OI-Pred: Disease-Specific Variant Pathogenicity Prediction for Osteogenesis Imperfecta},
  year = {2025},
  url = {https://github.com/yourusername/oi-pred}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Marini JC, et al. Osteogenesis imperfecta. Nat Rev Dis Primers. 2017;3:17052.
2. Forlino A, Marini JC. Osteogenesis imperfecta. Lancet. 2016;387:1657-1671.
3. Van Dijk FS, Sillence DO. Osteogenesis imperfecta: clinical diagnosis. Am J Med Genet A. 2014;164A:1470-1481.

## Contact

- **Author**: Emir Ceylan
- **Course**: ENS 210 - Bioinformatics
- **Institution**: Sabanci University

---

## Development

### Training a New Model

```bash
# Ensure data/feature_matrix.csv exists
python src/train.py
```

### Running the Full Pipeline

```bash
# 1. Feature engineering (if starting from raw data)
python src/feature_engineering.py

# 2. Train model
python src/train.py

# 3. Make predictions
python predict.py "COL1A1 G992S"
```

### Running Benchmarks

```bash
# SOTA comparison
python 05_tool_comparison/06_sota_benchmark.py

# External validation
python 05_tool_comparison/08_external_validation.py
```
