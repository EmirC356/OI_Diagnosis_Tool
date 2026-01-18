"""
Feature Engineering Module for OI-Pred
======================================
Extracts predictive features from variant data for pathogenicity prediction.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional

from config import AA_PROPERTIES, THREE_TO_ONE, FEATURE_COLS


def parse_protein_change(protein_change: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse protein change notation (e.g., 'G1448D', 'p.Gly1448Asp').

    Args:
        protein_change: Protein change string in various formats

    Returns:
        Tuple of (reference_aa, position, alternate_aa)
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

    # Pattern for frameshift: G1448fs or Gly1448fs
    pattern3 = r'^([A-Z][a-z]{2})?(\d+)(fs|del|ins|\=)$'
    match3 = re.match(pattern3, protein_change)
    if match3:
        ref = THREE_TO_ONE.get(match3.group(1), 'X') if match3.group(1) else 'X'
        return ref, int(match3.group(2)), 'X'

    return None, None, None


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all features from the variant dataset.

    Args:
        df: DataFrame with variant information

    Returns:
        DataFrame with extracted features
    """
    df = df.copy()

    # === 1. Basic categorical features ===

    # Binary encoding for molecular consequences
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

    # Gene features
    df['is_COL1A1'] = df['Gene(s)'].str.contains('COL1A1', na=False).astype(int)
    df['is_COL1A2'] = df['Gene(s)'].str.contains('COL1A2', na=False).astype(int)

    # === 2. Amino acid change features ===

    # Initialize columns
    for prop in ['hydrophobic_change', 'charge_change', 'polar_change',
                 'aromatic_change', 'size_change', 'flexibility_change']:
        df[prop] = 0.0

    df['has_aa_change'] = 0

    # Parse protein changes and calculate property differences
    for idx, row in df.iterrows():
        if pd.notna(row.get('Protein change')):
            ref_aa, pos, alt_aa = parse_protein_change(row['Protein change'])

            if ref_aa and alt_aa and ref_aa in AA_PROPERTIES and alt_aa in AA_PROPERTIES:
                df.at[idx, 'has_aa_change'] = 1
                ref_props = AA_PROPERTIES[ref_aa]
                alt_props = AA_PROPERTIES[alt_aa]

                df.at[idx, 'hydrophobic_change'] = alt_props['hydrophobic'] - ref_props['hydrophobic']
                df.at[idx, 'charge_change'] = abs(alt_props['charge'] - ref_props['charge'])
                df.at[idx, 'polar_change'] = abs(alt_props['polar'] - ref_props['polar'])
                df.at[idx, 'aromatic_change'] = abs(alt_props['aromatic'] - ref_props['aromatic'])
                df.at[idx, 'size_change'] = alt_props['size'] - ref_props['size']
                df.at[idx, 'flexibility_change'] = alt_props['flexibility'] - ref_props['flexibility']

    # === 3. Position features ===

    # Extract position from cDNA notation (e.g., c.4391T>C)
    df['cdna_position'] = df['Name'].str.extract(r'c\.([0-9]+)')[0].astype(float)

    # Normalize position (0-1 scale)
    max_pos = df['cdna_position'].max()
    df['normalized_position'] = df['cdna_position'] / max_pos if max_pos > 0 else 0

    # === 4. Derived risk features ===

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

    # Glycine substitution (critical for collagen triple helix)
    df['glycine_substitution'] = 0
    for idx, row in df.iterrows():
        if pd.notna(row.get('Protein change')):
            ref_aa, pos, alt_aa = parse_protein_change(row['Protein change'])
            if ref_aa == 'G' and alt_aa != 'G' and alt_aa != 'X':
                df.at[idx, 'glycine_substitution'] = 1

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature matrix for model training/prediction.

    Args:
        df: DataFrame with extracted features

    Returns:
        DataFrame with only the feature columns, NaN filled with 0
    """
    return df[FEATURE_COLS].fillna(0)


if __name__ == "__main__":
    # Test the module
    from config import CLEANED_DATA, FEATURE_MATRIX

    print("Loading data...")
    df = pd.read_csv(CLEANED_DATA)

    print(f"Extracting features from {len(df)} variants...")
    df_features = extract_features(df)

    print(f"Saving feature matrix to {FEATURE_MATRIX}...")
    df_features.to_csv(FEATURE_MATRIX, index=False)

    print(f"Done! Features: {len(FEATURE_COLS)}")
