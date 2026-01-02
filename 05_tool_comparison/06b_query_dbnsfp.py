"""
Query dbNSFP Database for Tool Predictions
This script matches missense variants with dbNSFP and extracts
SIFT, PolyPhen-2, CADD, and REVEL scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

print("="*70)
print("QUERYING dbNSFP FOR TOOL PREDICTIONS")
print("="*70)

# ===================================================================
# PART 1: Load Missense Variants
# ===================================================================

print("\n1. Loading Missense Variants to Query")
print("-" * 70)

df_missense = pd.read_csv("missense_variants_for_tools.tsv", sep='\t')
print(f"Loaded {len(df_missense)} missense variants")
print(f"  COL1A1 (chr17): {(df_missense['gene']=='COL1A1').sum()}")
print(f"  COL1A2 (chr7):  {(df_missense['gene']=='COL1A2').sum()}")

# ===================================================================
# PART 2: Load dbNSFP Database Files
# ===================================================================

print("\n2. Loading dbNSFP Database")
print("-" * 70)

# COL1A1 is on chr17, COL1A2 is on chr7
dbnsfp_files = {
    'chr7': 'dbNSFP4.5a_variant.chr7',
    'chr17': 'dbNSFP4.5a_variant.chr17'
}

# Check if files exist
for chrom, filepath in dbnsfp_files.items():
    if not Path(filepath).exists():
        print(f"\n[ERROR] dbNSFP file not found: {filepath}")
        print(f"\nPlease download:")
        print(f"  wget https://dbnsfp.s3.amazonaws.com/dbNSFP4.5a_variant.{chrom}.gz")
        print(f"  gunzip dbNSFP4.5a_variant.{chrom}.gz")
        exit(1)

print("\n[OK] dbNSFP files found")

# Define columns we need from dbNSFP
# Note: dbNSFP has MANY columns (>100), we only need specific ones
dbnsfp_cols_needed = [
    '#chr',                    # Chromosome
    'pos(1-based)',            # Position (GRCh38)
    'ref',                     # Reference allele
    'alt',                     # Alternate allele
    'aaref',                   # Reference amino acid
    'aaalt',                   # Alternate amino acid
    'aapos',                   # Amino acid position
    'genename',                # Gene name
    'Ensembl_transcriptid',    # Transcript ID
    'SIFT_pred',               # SIFT prediction
    'SIFT_score',              # SIFT score
    'Polyphen2_HDIV_pred',     # PolyPhen-2 prediction (HumDiv model)
    'Polyphen2_HDIV_score',    # PolyPhen-2 score (HumDiv model)
    'CADD_phred',              # CADD score (phred-scaled)
    'REVEL_score'              # REVEL score
]

# Load dbNSFP files
print("\nLoading dbNSFP chr7 (COL1A2)...")
# Read header first to check column availability
with open(dbnsfp_files['chr7'], 'r') as f:
    header = f.readline().strip().split('\t')

# Check which columns exist
available_cols = [col for col in dbnsfp_cols_needed if col in header]
missing_cols = [col for col in dbnsfp_cols_needed if col not in header]

if missing_cols:
    print(f"\n[WARNING] Some columns not found in dbNSFP:")
    for col in missing_cols:
        print(f"  - {col}")
    print("\nWill use available columns only.")

# Load chr7 (COL1A2)
print(f"\nReading chr7 file... (this may take 2-3 minutes)")
df_chr7 = pd.read_csv(
    dbnsfp_files['chr7'],
    sep='\t',
    usecols=available_cols,
    low_memory=False
)
print(f"[OK] Loaded {len(df_chr7):,} variants from chr7")

# Load chr17 (COL1A1)
print(f"\nReading chr17 file... (this may take 2-3 minutes)")
df_chr17 = pd.read_csv(
    dbnsfp_files['chr17'],
    sep='\t',
    usecols=available_cols,
    low_memory=False
)
print(f"[OK] Loaded {len(df_chr17):,} variants from chr17")

# Filter for COL1A1 and COL1A2 only
print("\nFiltering for COL1A1 and COL1A2...")
df_chr7_filtered = df_chr7[df_chr7['genename'] == 'COL1A2'].copy()
df_chr17_filtered = df_chr17[df_chr17['genename'] == 'COL1A1'].copy()

print(f"  COL1A2 variants in dbNSFP: {len(df_chr7_filtered):,}")
print(f"  COL1A1 variants in dbNSFP: {len(df_chr17_filtered):,}")

# Combine
df_dbnsfp = pd.concat([df_chr7_filtered, df_chr17_filtered], ignore_index=True)
print(f"\n[OK] Total COL1A1/COL1A2 variants in dbNSFP: {len(df_dbnsfp):,}")

# ===================================================================
# PART 3: Match Variants
# ===================================================================

print("\n3. Matching Variants with dbNSFP")
print("-" * 70)

# For matching, we need genomic coordinates from ClinVar
# Load original cleaned data to get GRCh38 positions
df_full = pd.read_csv(Path("data") / "cleaned_COL1_variants.csv")

# Merge to get genomic coordinates
df_missense_with_coords = df_missense.merge(
    df_full[['VariationID', 'GRCh38Chromosome', 'GRCh38Location']],
    on='VariationID',
    how='left'
)

# Parse GRCh38 coordinates
def parse_grch38_location(loc):
    """
    Parse GRCh38 location string
    Examples: '50184683', '50184683-50184686', '50184683:50184686'
    """
    if pd.isna(loc):
        return None, None

    loc_str = str(loc)
    # Handle range (take first position)
    if '-' in loc_str:
        return loc_str.split('-')[0], None
    elif ':' in loc_str:
        return loc_str.split(':')[0], None
    else:
        return loc_str, None

# Extract positions
print("\nParsing genomic coordinates...")
df_missense_with_coords['pos'] = None
for idx, row in df_missense_with_coords.iterrows():
    pos, _ = parse_grch38_location(row['GRCh38Location'])
    df_missense_with_coords.at[idx, 'pos'] = pos

# Count how many have coordinates
has_coords = df_missense_with_coords['pos'].notna().sum()
print(f"Variants with genomic coordinates: {has_coords}/{len(df_missense_with_coords)}")

# Try matching by position
print("\nMatching by genomic position...")
matches = []
for idx, variant in df_missense_with_coords.iterrows():
    if pd.isna(variant['pos']):
        continue

    # Get chromosome
    chrom = variant['GRCh38Chromosome']
    pos = variant['pos']
    gene = variant['gene']

    # Find match in dbNSFP
    mask = (df_dbnsfp['#chr'] == chrom) & \
           (df_dbnsfp['pos(1-based)'].astype(str) == str(pos)) & \
           (df_dbnsfp['genename'] == gene)

    matched = df_dbnsfp[mask]

    if len(matched) > 0:
        # Take first match if multiple
        match = matched.iloc[0]
        matches.append({
            'VariationID': variant['VariationID'],
            'Name': variant['Name'],
            'gene': variant['gene'],
            'protein_change': variant['protein_change'],
            'true_label': variant['true_label'],
            'SIFT_pred': match['SIFT_pred'] if 'SIFT_pred' in match else np.nan,
            'SIFT_score': match['SIFT_score'] if 'SIFT_score' in match else np.nan,
            'PolyPhen2_pred': match['Polyphen2_HDIV_pred'] if 'Polyphen2_HDIV_pred' in match else np.nan,
            'PolyPhen2_score': match['Polyphen2_HDIV_score'] if 'Polyphen2_HDIV_score' in match else np.nan,
            'CADD_score': match['CADD_phred'] if 'CADD_phred' in match else np.nan,
            'REVEL_score': match['REVEL_score'] if 'REVEL_score' in match else np.nan,
            'prediction_source': 'dbNSFP'
        })

df_matched = pd.DataFrame(matches)

print(f"\n[OK] Successfully matched {len(df_matched)}/{len(df_missense_with_coords)} variants")
print(f"    Match rate: {len(df_matched)/len(df_missense_with_coords)*100:.1f}%")

# ===================================================================
# PART 4: Analyze Tool Predictions
# ===================================================================

print("\n4. Analyzing Tool Predictions")
print("-" * 70)

if len(df_matched) > 0:
    # Count available predictions
    print("\nPrediction availability:")
    for tool in ['SIFT_score', 'PolyPhen2_score', 'CADD_score', 'REVEL_score']:
        available = df_matched[tool].notna().sum()
        print(f"  {tool:20s}: {available:4d}/{len(df_matched)} ({available/len(df_matched)*100:5.1f}%)")

    # Show sample predictions
    print("\nSample predictions:")
    sample = df_matched.head(5)[['Name', 'SIFT_score', 'PolyPhen2_score', 'CADD_score', 'REVEL_score']]
    print(sample.to_string(index=False))

    # Save matched predictions
    output_file = "missense_dbnsfp_predictions.tsv"
    df_matched.to_csv(output_file, sep='\t', index=False)
    print(f"\n[OK] Saved matched predictions to: {output_file}")
else:
    print("\n[WARNING] No matches found!")
    print("This could be due to:")
    print("  1. Coordinate system mismatch (GRCh37 vs GRCh38)")
    print("  2. Different position notation")
    print("  3. Missing coordinates in ClinVar data")

# ===================================================================
# PART 5: Merge with Consensus Predictions
# ===================================================================

print("\n5. Merging with Consensus Predictions")
print("-" * 70)

# Load consensus predictions
df_consensus = pd.read_csv("variant_predictions_with_consensus.tsv", sep='\t')

# Update predictions for matched missense variants
if len(df_matched) > 0:
    for idx, match in df_matched.iterrows():
        var_id = match['VariationID']
        mask = df_consensus['VariationID'] == var_id

        # Update predictions
        df_consensus.loc[mask, 'SIFT_pred'] = match['SIFT_pred']
        df_consensus.loc[mask, 'SIFT_score'] = match['SIFT_score']
        df_consensus.loc[mask, 'PolyPhen2_pred'] = match['PolyPhen2_pred']
        df_consensus.loc[mask, 'PolyPhen2_score'] = match['PolyPhen2_score']
        df_consensus.loc[mask, 'CADD_score'] = match['CADD_score']
        df_consensus.loc[mask, 'REVEL_score'] = match['REVEL_score']
        df_consensus.loc[mask, 'prediction_source'] = 'dbNSFP'

# Count prediction sources
print("\nFinal prediction source distribution:")
source_counts = df_consensus['prediction_source'].value_counts()
for source, count in source_counts.items():
    print(f"  {source:40s}: {count:4d} ({count/len(df_consensus)*100:5.1f}%)")

# Save final predictions
final_file = "all_variants_with_tool_predictions.tsv"
df_consensus.to_csv(final_file, sep='\t', index=False)
print(f"\n[OK] Saved final predictions to: {final_file}")

# ===================================================================
# PART 6: Summary
# ===================================================================

print("\n6. Summary")
print("=" * 70)

summary = f"""
dbNSFP QUERY RESULTS:

Total variants: {len(df_consensus)}

Prediction sources:
  - dbNSFP (matched missense):     {len(df_matched):4d} ({len(df_matched)/len(df_consensus)*100:5.1f}%)
  - Consensus (loss-of-function):  {(df_consensus['prediction_source']=='Consensus (Loss-of-Function)').sum():4d}
  - Consensus (silent/non-coding): {(df_consensus['prediction_source']=='Consensus (Silent/Non-coding)').sum():4d}
  - Pending (no match):            {(df_consensus['prediction_source']=='Pending').sum():4d}

FILES GENERATED:
  1. missense_dbnsfp_predictions.tsv       - Matched missense variants ({len(df_matched)})
  2. all_variants_with_tool_predictions.tsv - All variants with predictions ({len(df_consensus)})

NEXT STEPS:
  1. For unmatched missense variants, try alternative matching methods:
     - Match by protein change instead of genomic position
     - Check for coordinate system differences (GRCh37 vs GRCh38)

  2. Run final comparison analysis:
     - Execute 06c_compare_all_tools.py to compare ML models with real tool predictions
     - Generate updated visualizations with actual tool performance
"""

print(summary)

with open("dbNSFP_QUERY_SUMMARY.txt", 'w') as f:
    f.write(summary)

print("\n[OK] Summary saved to: dbNSFP_QUERY_SUMMARY.txt")

print("\n" + "=" * 70)
print("dbNSFP QUERY COMPLETE")
print("=" * 70)
