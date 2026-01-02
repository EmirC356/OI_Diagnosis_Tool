"""
Feature Engineering for Osteogenesis Imperfecta Variant Prediction
This script extracts predictive features from the variant dataset:
1. Categorical features (variant type, molecular consequence)
2. Amino acid biochemical properties (for missense variants)
3. Variant position features
4. Derived features
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Amino acid property tables
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

# Single letter amino acid codes
THREE_TO_ONE = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'fs': 'X'  # Stop codon and frameshift
}


def parse_protein_change(protein_change):
    """
    Parse protein change notation (e.g., 'G1448D', 'p.Gly1448Asp')
    Returns: (ref_aa, position, alt_aa)
    """
    if pd.isna(protein_change) or protein_change == '':
        return None, None, None

    # Remove 'p.' prefix if present
    protein_change = protein_change.replace('p.', '')

    # Pattern for single-letter notation: G1448D
    pattern1 = r'^([A-Z])(\d+)([A-Z\*])$'
    match1 = re.match(pattern1, protein_change)
    if match1:
        return match1.group(1), int(match1.group(2)), match1.group(3)

    # Pattern for three-letter notation: Gly1448Asp
    pattern2 = r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)$'
    match2 = re.match(pattern2, protein_change)
    if match2:
        ref = THREE_TO_ONE.get(match2.group(1))
        alt = THREE_TO_ONE.get(match2.group(3), match2.group(3))
        return ref, int(match2.group(2)), alt

    # Pattern for frameshift: G1448fs
    pattern3 = r'^([A-Z][a-z]{2})?(\d+)(fs|del|ins|\=)$'
    match3 = re.match(pattern3, protein_change)
    if match3:
        ref = THREE_TO_ONE.get(match3.group(1), 'X') if match3.group(1) else 'X'
        return ref, int(match3.group(2)), 'X'

    return None, None, None


def extract_features(df):
    """Extract all features from the variant dataset"""

    df = df.copy()

    # === 1. Basic categorical features ===
    print("Extracting basic features...")

    # Binary encoding for variant types
    df['is_missense'] = (df['Molecular consequence'] == 'missense variant').astype(int)
    df['is_nonsense'] = (df['Molecular consequence'] == 'nonsense').astype(int)
    df['is_frameshift'] = (df['Molecular consequence'] == 'frameshift variant').astype(int)
    df['is_splice'] = df['Molecular consequence'].str.contains('splice', na=False).astype(int)
    df['is_synonymous'] = (df['Molecular consequence'] == 'synonymous variant').astype(int)
    df['is_intron'] = (df['Molecular consequence'] == 'intron variant').astype(int)
    df['is_utr'] = df['Molecular consequence'].str.contains('UTR', na=False).astype(int)
    df['is_inframe_indel'] = df['Molecular consequence'].str.contains('inframe', na=False).astype(int)

    # Variant type features
    df['is_snv'] = (df['Variant type'] == 'single nucleotide variant').astype(int)
    df['is_deletion'] = (df['Variant type'] == 'Deletion').astype(int)
    df['is_insertion'] = (df['Variant type'] == 'Insertion').astype(int)
    df['is_duplication'] = (df['Variant type'] == 'Duplication').astype(int)

    # Gene feature
    df['is_COL1A1'] = df['Gene(s)'].str.contains('COL1A1', na=False).astype(int)
    df['is_COL1A2'] = df['Gene(s)'].str.contains('COL1A2', na=False).astype(int)

    # === 2. Amino acid change features (for missense variants) ===
    print("Extracting amino acid properties...")

    # Initialize columns
    for prop in ['hydrophobic_change', 'charge_change', 'polar_change',
                 'aromatic_change', 'size_change', 'flexibility_change']:
        df[prop] = 0.0

    df['has_aa_change'] = 0

    # Parse protein changes
    for idx, row in df.iterrows():
        if pd.notna(row['Protein change']):
            ref_aa, pos, alt_aa = parse_protein_change(row['Protein change'])

            if ref_aa and alt_aa and ref_aa in AA_PROPERTIES and alt_aa in AA_PROPERTIES:
                df.at[idx, 'has_aa_change'] = 1
                ref_props = AA_PROPERTIES[ref_aa]
                alt_props = AA_PROPERTIES[alt_aa]

                # Calculate property changes
                df.at[idx, 'hydrophobic_change'] = alt_props['hydrophobic'] - ref_props['hydrophobic']
                df.at[idx, 'charge_change'] = abs(alt_props['charge'] - ref_props['charge'])
                df.at[idx, 'polar_change'] = abs(alt_props['polar'] - ref_props['polar'])
                df.at[idx, 'aromatic_change'] = abs(alt_props['aromatic'] - ref_props['aromatic'])
                df.at[idx, 'size_change'] = alt_props['size'] - ref_props['size']
                df.at[idx, 'flexibility_change'] = alt_props['flexibility'] - ref_props['flexibility']

    # === 3. Position features ===
    print("Extracting position features...")

    # Extract position from cDNA notation (e.g., c.4391T>C)
    df['cdna_position'] = df['Name'].str.extract(r'c\.([0-9]+)')[0].astype(float)

    # Normalize position (0-1 scale) - COL1A1 is ~4400bp, COL1A2 is ~4200bp
    df['normalized_position'] = df['cdna_position'] / df['cdna_position'].max()

    # === 4. Derived risk features ===
    print("Creating derived risk features...")

    # High-confidence pathogenic indicators
    df['high_risk_consequence'] = (
        (df['is_nonsense'] == 1) |
        (df['is_frameshift'] == 1) |
        (df['is_splice'] == 1)
    ).astype(int)

    # Low-risk indicators
    df['low_risk_consequence'] = (
        (df['is_synonymous'] == 1) |
        (df['is_intron'] == 1) |
        (df['is_utr'] == 1)
    ).astype(int)

    # Glycine substitution in collagen (critical for triple helix)
    df['glycine_substitution'] = 0
    for idx, row in df.iterrows():
        if pd.notna(row['Protein change']):
            ref_aa, pos, alt_aa = parse_protein_change(row['Protein change'])
            if ref_aa == 'G' and alt_aa != 'G' and alt_aa != 'X':
                df.at[idx, 'glycine_substitution'] = 1

    return df


# Load data
print("Loading cleaned variant data...")
data_path = Path("data") / "cleaned_COL1_variants.csv"
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} variants")

# Extract features
df_features = extract_features(df)

# Select feature columns for modeling
feature_cols = [
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

# Save feature matrix
output_path = Path("data") / "feature_matrix.csv"
df_features.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print("FEATURE ENGINEERING COMPLETE")
print(f"{'='*60}")
print(f"Total features extracted: {len(feature_cols)}")
print(f"Feature matrix saved to: {output_path}")
print(f"\nFeature list:")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {feat}")

# Show feature statistics
print(f"\n{'='*60}")
print("FEATURE STATISTICS")
print(f"{'='*60}")
print(df_features[feature_cols].describe().T)

# Show correlation with label
print(f"\n{'='*60}")
print("FEATURE CORRELATION WITH PATHOGENICITY (Top 15)")
print(f"{'='*60}")
correlations = df_features[feature_cols + ['label']].corr()['label'].drop('label').sort_values(ascending=False)
print(correlations.head(15))
