#!/usr/bin/env python3
"""
OI-Pred: Osteogenesis Imperfecta Variant Pathogenicity Predictor
================================================================

Predicts pathogenicity of COL1A1/COL1A2 variants associated with
Osteogenesis Imperfecta using a disease-specific Random Forest model.

Usage:
    python predict.py "COL1A1 p.Gly345Ser"
    python predict.py "COL1A1" "G345S" "missense"
    python predict.py --file variants.csv
    python predict.py --interactive

Examples:
    # Single variant (auto-parsed)
    python predict.py "COL1A1 p.Gly345Ser"

    # Explicit format
    python predict.py --gene COL1A1 --protein "G345S" --consequence missense

    # Batch prediction from file
    python predict.py --file my_variants.csv --output predictions.csv

Author: ENS210 Project
Model: Random Forest (97.3% accuracy, 98.9% ROC-AUC)
"""

import argparse
import sys
import re
import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    TRAINED_MODEL, FEATURE_LIST, FEATURE_COLS,
    AA_PROPERTIES, THREE_TO_ONE
)


class OIPred:
    """Osteogenesis Imperfecta Variant Pathogenicity Predictor."""

    def __init__(self, model_path=None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model file (optional)
        """
        model_path = model_path or TRAINED_MODEL

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please run 'python src/train.py' first to train the model."
            )

        self.model = joblib.load(model_path)

        # Load feature list
        if Path(FEATURE_LIST).exists():
            with open(FEATURE_LIST, 'r') as f:
                self.features = json.load(f)
        else:
            self.features = FEATURE_COLS

    def parse_variant(self, variant_str):
        """
        Parse a variant string into components.

        Args:
            variant_str: String like "COL1A1 p.Gly345Ser" or "COL1A1 G345S"

        Returns:
            dict with gene, protein_change, and parsed amino acids
        """
        variant_str = variant_str.strip()

        # Extract gene first and remove it from string for cleaner parsing
        gene = None
        clean_str = variant_str
        if 'COL1A1' in variant_str.upper():
            gene = 'COL1A1'
            clean_str = re.sub(r'COL1A1\s*', '', variant_str, flags=re.IGNORECASE)
        elif 'COL1A2' in variant_str.upper():
            gene = 'COL1A2'
            clean_str = re.sub(r'COL1A2\s*', '', variant_str, flags=re.IGNORECASE)

        # Remove 'p.' prefix if present
        clean_str = clean_str.replace('p.', '').strip()

        # Parse amino acid change
        ref_aa, position, alt_aa = None, None, None
        protein_change = None

        # Try three letter notation first: Gly345Ser
        three_match = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', clean_str)
        if three_match:
            ref_aa = THREE_TO_ONE.get(three_match.group(1))
            position = int(three_match.group(2))
            alt_aa = THREE_TO_ONE.get(three_match.group(3))
            protein_change = clean_str

        # Try single letter: G345S (must have at least 2 digit position)
        if not protein_change:
            single_match = re.match(r'^([A-Z])(\d{2,})([A-Z])$', clean_str)
            if single_match:
                ref_aa = single_match.group(1)
                position = int(single_match.group(2))
                alt_aa = single_match.group(3)
                protein_change = clean_str

        return {
            'gene': gene,
            'protein_change': protein_change,
            'ref_aa': ref_aa,
            'position': position,
            'alt_aa': alt_aa
        }

    def extract_features(self, gene, protein_change=None, consequence='missense',
                         variant_type='single nucleotide variant'):
        """
        Extract features for a single variant.

        Args:
            gene: 'COL1A1' or 'COL1A2'
            protein_change: Protein change string (e.g., 'G345S')
            consequence: Molecular consequence
            variant_type: Type of variant

        Returns:
            dict of feature values
        """
        features = {col: 0 for col in self.features}

        # Gene features
        features['is_COL1A1'] = 1 if gene == 'COL1A1' else 0
        features['is_COL1A2'] = 1 if gene == 'COL1A2' else 0

        # Consequence features
        consequence_lower = consequence.lower()
        features['is_missense'] = 1 if 'missense' in consequence_lower else 0
        features['is_nonsense'] = 1 if 'nonsense' in consequence_lower else 0
        features['is_frameshift'] = 1 if 'frameshift' in consequence_lower else 0
        features['is_splice'] = 1 if 'splice' in consequence_lower else 0
        features['is_synonymous'] = 1 if 'synonymous' in consequence_lower else 0
        features['is_intron'] = 1 if 'intron' in consequence_lower else 0
        features['is_utr'] = 1 if 'utr' in consequence_lower else 0
        features['is_inframe_indel'] = 1 if 'inframe' in consequence_lower else 0

        # Variant type features
        variant_lower = variant_type.lower()
        features['is_snv'] = 1 if 'single' in variant_lower or 'snv' in variant_lower else 0
        features['is_deletion'] = 1 if 'deletion' in variant_lower else 0
        features['is_insertion'] = 1 if 'insertion' in variant_lower else 0
        features['is_duplication'] = 1 if 'duplication' in variant_lower else 0

        # Parse protein change
        if protein_change:
            parsed = self.parse_variant(f"{gene} {protein_change}")
            ref_aa = parsed['ref_aa']
            alt_aa = parsed['alt_aa']
            position = parsed['position']

            if ref_aa and alt_aa:
                features['has_aa_change'] = 1

                # Amino acid property changes
                if ref_aa in AA_PROPERTIES and alt_aa in AA_PROPERTIES:
                    ref_props = AA_PROPERTIES[ref_aa]
                    alt_props = AA_PROPERTIES[alt_aa]

                    features['hydrophobic_change'] = alt_props['hydrophobic'] - ref_props['hydrophobic']
                    features['charge_change'] = abs(alt_props['charge'] - ref_props['charge'])
                    features['polar_change'] = abs(alt_props['polar'] - ref_props['polar'])
                    features['aromatic_change'] = abs(alt_props['aromatic'] - ref_props['aromatic'])
                    features['size_change'] = alt_props['size'] - ref_props['size']
                    features['flexibility_change'] = alt_props['flexibility'] - ref_props['flexibility']

                # Glycine substitution (critical for OI)
                if ref_aa == 'G' and alt_aa != 'G':
                    features['glycine_substitution'] = 1

            # Normalized position (approximate based on typical gene length)
            if position:
                max_pos = 4400 if gene == 'COL1A1' else 4200
                features['normalized_position'] = min(position * 3 / max_pos, 1.0)

        # Derived features
        features['high_risk_consequence'] = max(
            features['is_nonsense'],
            features['is_frameshift'],
            features['is_splice']
        )
        features['low_risk_consequence'] = max(
            features['is_synonymous'],
            features['is_intron'],
            features['is_utr']
        )

        return features

    def predict(self, gene, protein_change=None, consequence='missense',
                variant_type='single nucleotide variant'):
        """
        Predict pathogenicity for a single variant.

        Args:
            gene: 'COL1A1' or 'COL1A2'
            protein_change: Protein change string
            consequence: Molecular consequence
            variant_type: Type of variant

        Returns:
            dict with prediction, probability, and interpretation
        """
        features = self.extract_features(gene, protein_change, consequence, variant_type)

        # Create feature vector in correct order
        X = pd.DataFrame([features])[self.features]

        # Predict
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        # Interpretation
        pathogenic_prob = proba[1]
        if pathogenic_prob >= 0.9:
            interpretation = "Likely Pathogenic (High Confidence)"
        elif pathogenic_prob >= 0.7:
            interpretation = "Likely Pathogenic"
        elif pathogenic_prob >= 0.3:
            interpretation = "Uncertain Significance"
        elif pathogenic_prob >= 0.1:
            interpretation = "Likely Benign"
        else:
            interpretation = "Likely Benign (High Confidence)"

        return {
            'prediction': 'Pathogenic' if pred == 1 else 'Benign',
            'pathogenic_probability': round(pathogenic_prob, 4),
            'benign_probability': round(proba[0], 4),
            'interpretation': interpretation,
            'gene': gene,
            'protein_change': protein_change,
            'consequence': consequence
        }

    def predict_from_string(self, variant_str, consequence='missense'):
        """
        Predict from a variant string.

        Args:
            variant_str: String like "COL1A1 p.Gly345Ser"
            consequence: Molecular consequence (if not inferrable)

        Returns:
            Prediction dict
        """
        parsed = self.parse_variant(variant_str)

        if not parsed['gene']:
            raise ValueError(f"Could not determine gene from: {variant_str}")

        return self.predict(
            gene=parsed['gene'],
            protein_change=parsed['protein_change'] or variant_str,
            consequence=consequence
        )

    def predict_batch(self, df):
        """
        Predict pathogenicity for multiple variants.

        Args:
            df: DataFrame with 'Gene', 'Protein_change' columns

        Returns:
            DataFrame with predictions added
        """
        results = []
        for _, row in df.iterrows():
            gene = row.get('Gene') or row.get('gene') or row.get('Gene(s)')
            protein = row.get('Protein_change') or row.get('protein_change') or row.get('Protein change')
            consequence = row.get('Consequence') or row.get('consequence') or row.get('Molecular consequence') or 'missense'

            try:
                pred = self.predict(gene, protein, consequence)
                results.append(pred)
            except Exception as e:
                results.append({
                    'prediction': 'Error',
                    'pathogenic_probability': None,
                    'error': str(e)
                })

        return pd.DataFrame(results)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='OI-Pred: Predict pathogenicity of COL1A1/COL1A2 variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "COL1A1 p.Gly345Ser"
  python predict.py --gene COL1A1 --protein G345S --consequence missense
  python predict.py --file variants.csv --output predictions.csv
  python predict.py --interactive
        """
    )

    parser.add_argument('variant', nargs='?', help='Variant string (e.g., "COL1A1 p.Gly345Ser")')
    parser.add_argument('--gene', '-g', help='Gene name (COL1A1 or COL1A2)')
    parser.add_argument('--protein', '-p', help='Protein change (e.g., G345S)')
    parser.add_argument('--consequence', '-c', default='missense',
                        help='Molecular consequence (default: missense)')
    parser.add_argument('--file', '-f', help='Input CSV file for batch prediction')
    parser.add_argument('--output', '-o', help='Output file for batch predictions')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--model', '-m', help='Path to custom model file')

    args = parser.parse_args()

    # Initialize predictor
    try:
        predictor = OIPred(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("OI-Pred: Interactive Mode")
        print("="*60)
        print("Enter variants in format: COL1A1 p.Gly345Ser")
        print("Type 'quit' to exit\n")

        while True:
            try:
                variant = input("Variant> ").strip()
                if variant.lower() in ['quit', 'exit', 'q']:
                    break
                if not variant:
                    continue

                result = predictor.predict_from_string(variant)
                print(f"\n  Prediction: {result['prediction']}")
                print(f"  Probability: {result['pathogenic_probability']:.1%} pathogenic")
                print(f"  Interpretation: {result['interpretation']}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  Error: {e}\n")

        print("\nGoodbye!")
        return

    # Batch mode
    if args.file:
        print(f"Loading variants from: {args.file}")
        df = pd.read_csv(args.file)

        print(f"Predicting pathogenicity for {len(df)} variants...")
        results = predictor.predict_batch(df)

        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")
        else:
            print("\nResults:")
            print(results.to_string())
        return

    # Single variant mode
    if args.variant:
        result = predictor.predict_from_string(args.variant, args.consequence)
    elif args.gene and args.protein:
        result = predictor.predict(args.gene, args.protein, args.consequence)
    else:
        parser.print_help()
        sys.exit(1)

    # Print result
    print("\n" + "="*60)
    print("OI-Pred Prediction Result")
    print("="*60)
    print(f"  Gene:            {result['gene']}")
    print(f"  Protein Change:  {result['protein_change']}")
    print(f"  Consequence:     {result['consequence']}")
    print("-"*60)
    print(f"  Prediction:      {result['prediction']}")
    print(f"  Probability:     {result['pathogenic_probability']:.1%} pathogenic")
    print(f"  Interpretation:  {result['interpretation']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
