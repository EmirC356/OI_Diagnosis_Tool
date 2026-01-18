"""
Configuration file for OI-Pred: Osteogenesis Imperfecta Variant Predictor
=========================================================================
Centralizes all paths and constants for reproducibility.
"""

from pathlib import Path

# ============================================================
# Directory Configuration
# ============================================================

# Project root (automatically detected)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "06_results"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# File Paths
# ============================================================

# Input data
RAW_DATA_COL1A1 = DATA_DIR / "COL1A1_All.txt"
RAW_DATA_COL1A2 = DATA_DIR / "COL1A2_All.txt"
CLEANED_DATA = DATA_DIR / "cleaned_COL1_variants.csv"
FEATURE_MATRIX = DATA_DIR / "feature_matrix.csv"

# Model files
TRAINED_MODEL = MODELS_DIR / "oi_pred_rf_model.pkl"
SCALER_FILE = MODELS_DIR / "feature_scaler.pkl"
FEATURE_LIST = MODELS_DIR / "feature_list.json"

# ============================================================
# Feature Configuration
# ============================================================

# Original 25 features (proven optimal after feature selection)
FEATURE_COLS = [
    # Molecular consequence features
    'is_missense', 'is_nonsense', 'is_frameshift', 'is_splice',
    'is_synonymous', 'is_intron', 'is_utr', 'is_inframe_indel',
    # Variant type features
    'is_snv', 'is_deletion', 'is_insertion', 'is_duplication',
    # Gene features
    'is_COL1A1', 'is_COL1A2',
    # Amino acid property changes
    'hydrophobic_change', 'charge_change', 'polar_change',
    'aromatic_change', 'size_change', 'flexibility_change',
    'has_aa_change',
    # Position features
    'normalized_position',
    # Derived features
    'high_risk_consequence', 'low_risk_consequence', 'glycine_substitution'
]

# ============================================================
# Amino Acid Properties
# ============================================================

AA_PROPERTIES = {
    'A': {'hydrophobic': 1.8, 'charge': 0, 'polar': 0, 'aromatic': 0, 'size': 89, 'flexibility': 0.36},
    'R': {'hydrophobic': -4.5, 'charge': 1, 'polar': 1, 'aromatic': 0, 'size': 174, 'flexibility': 0.53},
    'N': {'hydrophobic': -3.5, 'charge': 0, 'polar': 1, 'aromatic': 0, 'size': 132, 'flexibility': 0.46},
    'D': {'hydrophobic': -3.5, 'charge': -1, 'polar': 1, 'aromatic': 0, 'size': 133, 'flexibility': 0.51},
    'C': {'hydrophobic': 2.5, 'charge': 0, 'polar': 1, 'aromatic': 0, 'size': 121, 'flexibility': 0.35},
    'Q': {'hydrophobic': -3.5, 'charge': 0, 'polar': 1, 'aromatic': 0, 'size': 146, 'flexibility': 0.49},
    'E': {'hydrophobic': -3.5, 'charge': -1, 'polar': 1, 'aromatic': 0, 'size': 147, 'flexibility': 0.50},
    'G': {'hydrophobic': -0.4, 'charge': 0, 'polar': 0, 'aromatic': 0, 'size': 75, 'flexibility': 0.54},
    'H': {'hydrophobic': -3.2, 'charge': 0.5, 'polar': 1, 'aromatic': 1, 'size': 155, 'flexibility': 0.32},
    'I': {'hydrophobic': 4.5, 'charge': 0, 'polar': 0, 'aromatic': 0, 'size': 131, 'flexibility': 0.46},
    'L': {'hydrophobic': 3.8, 'charge': 0, 'polar': 0, 'aromatic': 0, 'size': 131, 'flexibility': 0.37},
    'K': {'hydrophobic': -3.9, 'charge': 1, 'polar': 1, 'aromatic': 0, 'size': 146, 'flexibility': 0.47},
    'M': {'hydrophobic': 1.9, 'charge': 0, 'polar': 0, 'aromatic': 0, 'size': 149, 'flexibility': 0.30},
    'F': {'hydrophobic': 2.8, 'charge': 0, 'polar': 0, 'aromatic': 1, 'size': 165, 'flexibility': 0.31},
    'P': {'hydrophobic': -1.6, 'charge': 0, 'polar': 0, 'aromatic': 0, 'size': 115, 'flexibility': 0.51},
    'S': {'hydrophobic': -0.8, 'charge': 0, 'polar': 1, 'aromatic': 0, 'size': 105, 'flexibility': 0.51},
    'T': {'hydrophobic': -0.7, 'charge': 0, 'polar': 1, 'aromatic': 0, 'size': 119, 'flexibility': 0.44},
    'W': {'hydrophobic': -0.9, 'charge': 0, 'polar': 0, 'aromatic': 1, 'size': 204, 'flexibility': 0.31},
    'Y': {'hydrophobic': -1.3, 'charge': 0, 'polar': 1, 'aromatic': 1, 'size': 181, 'flexibility': 0.42},
    'V': {'hydrophobic': 4.2, 'charge': 0, 'polar': 0, 'aromatic': 0, 'size': 117, 'flexibility': 0.39},
}

# Three-letter to one-letter amino acid codes
THREE_TO_ONE = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'fs': 'X'
}

# ============================================================
# Model Configuration
# ============================================================

RANDOM_STATE = 42
CV_FOLDS = 5

# Random Forest hyperparameters (optimized)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
