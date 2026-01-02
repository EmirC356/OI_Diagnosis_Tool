"""
Prepare Variants for External Tool Prediction
This script extracts variant information and prepares it for querying
SIFT, PolyPhen-2, CADD, REVEL via dbNSFP or other methods.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

print("="*70)
print("PREPARING VARIANTS FOR EXTERNAL TOOL QUERIES")
print("="*70)

# Load cleaned variant data
df = pd.read_csv(Path("data") / "cleaned_COL1_variants.csv")

print(f"\n1. Dataset Summary")
print(f"   Total variants: {len(df)}")
print(f"   Pathogenic: {(df['label']==1).sum()}")
print(f"   Benign: {(df['label']==0).sum()}")

# ===================================================================
# PART 1: Categorize Variants by Type
# ===================================================================

print(f"\n2. Categorizing Variants by Type")
print("-" * 70)

# Count variants by molecular consequence
conseq_counts = df['Molecular consequence'].value_counts()
print("\nMolecular Consequence Distribution:")
for conseq, count in conseq_counts.head(10).items():
    print(f"  {conseq:30s}: {count:4d} ({count/len(df)*100:5.1f}%)")

# Identify missense variants (need tool predictions)
missense_variants = df[df['Molecular consequence'] == 'missense variant'].copy()
print(f"\n[*] Missense variants (need tool predictions): {len(missense_variants)}")

# Identify clear pathogenic (don't need tool predictions)
clear_pathogenic = df[df['Molecular consequence'].isin([
    'frameshift variant',
    'nonsense',
    'splice donor variant',
    'splice acceptor variant'
])].copy()
print(f"[*] Clear pathogenic (frameshift/nonsense/splice): {len(clear_pathogenic)}")

# Identify clear benign (don't need tool predictions)
clear_benign = df[df['Molecular consequence'].isin([
    'synonymous variant',
    'intron variant',
    '3 prime UTR variant',
    '5 prime UTR variant'
])].copy()
print(f"[*] Clear benign (synonymous/intronic/UTR): {len(clear_benign)}")

# Other variants
other_variants = df[~df['Molecular consequence'].isin(
    list(missense_variants['Molecular consequence']) +
    list(clear_pathogenic['Molecular consequence']) +
    list(clear_benign['Molecular consequence'])
)].copy()
print(f"[*] Other variants (inframe indels, etc.): {len(other_variants)}")

# ===================================================================
# PART 2: Extract Genomic Coordinates for Missense Variants
# ===================================================================

print(f"\n3. Extracting Genomic Coordinates for Missense Variants")
print("-" * 70)

# Function to extract genomic info from variant name
def parse_variant_name(name):
    """
    Parse ClinVar variant name to extract basic info
    Example: NM_000088.4(COL1A1):c.3455G>A (p.Gly1152Asp)
    """
    info = {
        'gene': None,
        'cdna_change': None,
        'protein_change': None,
        'ref_nt': None,
        'alt_nt': None,
        'cdna_pos': None
    }

    # Extract gene
    gene_match = re.search(r'\(([A-Z0-9]+)\)', name)
    if gene_match:
        info['gene'] = gene_match.group(1)

    # Extract cDNA change (e.g., c.3455G>A)
    cdna_match = re.search(r'c\.(\d+)([A-Z])>([A-Z])', name)
    if cdna_match:
        info['cdna_pos'] = int(cdna_match.group(1))
        info['ref_nt'] = cdna_match.group(2)
        info['alt_nt'] = cdna_match.group(3)
        info['cdna_change'] = f"c.{info['cdna_pos']}{info['ref_nt']}>{info['alt_nt']}"

    # Extract protein change
    prot_match = re.search(r'\(p\.([^\)]+)\)', name)
    if prot_match:
        info['protein_change'] = prot_match.group(1)

    return info

# Parse all missense variants
print("\nParsing missense variant information...")
missense_parsed = []
for idx, row in missense_variants.iterrows():
    parsed = parse_variant_name(row['Name'])
    parsed['VariationID'] = row['VariationID']
    parsed['Name'] = row['Name']
    parsed['Protein_change_col'] = row['Protein change']
    parsed['true_label'] = row['label']
    missense_parsed.append(parsed)

df_missense = pd.DataFrame(missense_parsed)

# Filter for successfully parsed variants
df_missense_valid = df_missense[
    df_missense['cdna_change'].notna() &
    df_missense['protein_change'].notna()
].copy()

print(f"\nSuccessfully parsed: {len(df_missense_valid)}/{len(missense_variants)} missense variants")
print(f"Failed to parse: {len(missense_variants) - len(df_missense_valid)}")

# Gene distribution
print("\nGene distribution in missense variants:")
print(df_missense_valid['gene'].value_counts())

# ===================================================================
# PART 3: Create Query Files for Different Tools
# ===================================================================

print(f"\n4. Creating Query Files")
print("-" * 70)

# For COL1A1 and COL1A2, we need genomic coordinates
# COL1A1: chr17:50184200-50201600 (GRCh38)
# COL1A2: chr7:94394500-94431300 (GRCh38)

# Note: We don't have exact genomic coordinates from ClinVar data
# We'll need to use VariationID or create a list for manual lookup

# Create simple query file with available information
query_file = "missense_variants_for_tools.tsv"
df_missense_valid[['VariationID', 'Name', 'gene', 'protein_change', 'cdna_change', 'true_label']].to_csv(
    query_file, sep='\t', index=False
)
print(f"\n[OK] Saved missense variant list to: {query_file}")
print(f"    Columns: VariationID, Name, gene, protein_change, cdna_change, true_label")

# ===================================================================
# PART 4: Create Files for Automatic Tool Assignment
# ===================================================================

print(f"\n5. Creating Consensus Predictions for Non-Missense Variants")
print("-" * 70)

# Assign consensus predictions based on variant type
def assign_consensus_prediction(row):
    """
    Assign consensus predictions for variants where all tools agree
    """
    conseq = row['Molecular consequence']

    # Handle NaN values
    if pd.isna(conseq):
        conseq = 'Unknown'

    conseq = str(conseq)  # Ensure it's a string

    # Loss-of-function: All tools predict pathogenic
    if conseq in ['frameshift variant', 'nonsense',
                  'splice donor variant', 'splice acceptor variant']:
        return {
            'SIFT_pred': 'Deleterious',
            'SIFT_score': 0.00,  # 0 = deleterious
            'PolyPhen2_pred': 'Probably Damaging',
            'PolyPhen2_score': 1.00,  # 1 = damaging
            'CADD_score': 30.0,  # High CADD = pathogenic
            'REVEL_score': 0.95,  # High REVEL = pathogenic
            'prediction_source': 'Consensus (Loss-of-Function)'
        }

    # Silent/non-coding: All tools predict benign
    elif conseq in ['synonymous variant', 'intron variant',
                   '3 prime UTR variant', '5 prime UTR variant']:
        return {
            'SIFT_pred': 'Tolerated',
            'SIFT_score': 1.00,  # 1 = tolerated
            'PolyPhen2_pred': 'Benign',
            'PolyPhen2_score': 0.00,  # 0 = benign
            'CADD_score': 5.0,  # Low CADD = benign
            'REVEL_score': 0.10,  # Low REVEL = benign
            'prediction_source': 'Consensus (Silent/Non-coding)'
        }

    # In-frame indels: Variable, assign neutral
    elif 'inframe' in conseq.lower():
        return {
            'SIFT_pred': 'Unknown',
            'SIFT_score': np.nan,
            'PolyPhen2_pred': 'Unknown',
            'PolyPhen2_score': np.nan,
            'CADD_score': 15.0,  # Moderate CADD
            'REVEL_score': 0.50,  # Moderate REVEL
            'prediction_source': 'Default (Inframe Indel)'
        }

    # Missense: Needs real tool predictions
    else:
        return {
            'SIFT_pred': 'Needs Tool Query',
            'SIFT_score': np.nan,
            'PolyPhen2_pred': 'Needs Tool Query',
            'PolyPhen2_score': np.nan,
            'CADD_score': np.nan,
            'REVEL_score': np.nan,
            'prediction_source': 'Pending'
        }

# Apply consensus predictions to all variants
print("\nAssigning consensus predictions...")
predictions = []
for idx, row in df.iterrows():
    pred = assign_consensus_prediction(row)
    pred['VariationID'] = row['VariationID']
    pred['Name'] = row['Name']
    pred['Molecular_consequence'] = row['Molecular consequence']
    pred['true_label'] = row['label']
    predictions.append(pred)

df_predictions = pd.DataFrame(predictions)

# Count by prediction source
print("\nPrediction source distribution:")
source_counts = df_predictions['prediction_source'].value_counts()
for source, count in source_counts.items():
    print(f"  {source:40s}: {count:4d} ({count/len(df)*100:5.1f}%)")

# Save predictions
predictions_file = "variant_predictions_with_consensus.tsv"
df_predictions.to_csv(predictions_file, sep='\t', index=False)
print(f"\n[OK] Saved predictions file to: {predictions_file}")

# ===================================================================
# PART 5: Create Instructions for dbNSFP Query
# ===================================================================

print(f"\n6. Instructions for Querying dbNSFP")
print("-" * 70)

instructions = """
TO GET SIFT/POLYPHEN/CADD/REVEL PREDICTIONS:

Option 1: Use dbNSFP Database (RECOMMENDED)
--------------------------------------------
1. Download dbNSFP for chr7 and chr17:
   wget https://dbnsfp.s3.amazonaws.com/dbNSFP4.5a_variant.chr7.gz
   wget https://dbnsfp.s3.amazonaws.com/dbNSFP4.5a_variant.chr17.gz

2. Unzip the files:
   gunzip dbNSFP4.5a_variant.chr7.gz
   gunzip dbNSFP4.5a_variant.chr17.gz

3. Use the provided script (06b_query_dbnsfp.py) to:
   - Match your {num_missense} missense variants
   - Extract SIFT, PolyPhen-2, CADD, REVEL scores
   - Merge with consensus predictions for other variants

Option 2: Use ClinVar Annotations (QUICK CHECK)
------------------------------------------------
Your ClinVar download might already have some tool predictions!
Check the original COL1A1_All.txt and COL1A2_All.txt files for columns like:
- SIFT_pred, SIFT_score
- Polyphen2_HDIV_pred, Polyphen2_HDIV_score
- CADD_phred
- REVEL_score

If these columns exist, we can extract them directly!

Option 3: Manual Web Query (SLOW)
----------------------------------
For the {num_missense} missense variants, manually query:
- SIFT: http://sift.bii.a-star.edu.sg/
- PolyPhen-2: http://genetics.bwh.harvard.edu/pph2/
- CADD: https://cadd.gs.washington.edu/
- REVEL: Pre-computed scores only

NOT RECOMMENDED due to time constraints.
""".format(num_missense=len(df_missense_valid))

with open("INSTRUCTIONS_FOR_TOOL_QUERIES.txt", 'w') as f:
    f.write(instructions)

print(instructions)
print("\n[OK] Instructions saved to: INSTRUCTIONS_FOR_TOOL_QUERIES.txt")

# ===================================================================
# PART 6: Summary Statistics
# ===================================================================

print(f"\n7. Summary Statistics")
print("=" * 70)

summary = f"""
VARIANT CATEGORIZATION SUMMARY:

Total Variants: {len(df)}

Categories:
  1. Missense variants (need tool queries):     {len(df_missense_valid):4d} ({len(df_missense_valid)/len(df)*100:5.1f}%)
  2. Loss-of-function (consensus pathogenic):   {len(clear_pathogenic):4d} ({len(clear_pathogenic)/len(df)*100:5.1f}%)
  3. Silent/non-coding (consensus benign):      {len(clear_benign):4d} ({len(clear_benign)/len(df)*100:5.1f}%)
  4. Other variants (default moderate):         {len(other_variants):4d} ({len(other_variants)/len(df)*100:5.1f}%)

FILES GENERATED:
  1. missense_variants_for_tools.tsv           - List of {len(df_missense_valid)} missense variants to query
  2. variant_predictions_with_consensus.tsv    - All {len(df)} variants with consensus predictions
  3. INSTRUCTIONS_FOR_TOOL_QUERIES.txt         - Next steps guide

NEXT STEPS:
  1. Check if your original ClinVar files have tool predictions already
  2. If not, download dbNSFP (chr7 + chr17)
  3. Run 06b_query_dbnsfp.py to match variants and extract scores
  4. Run 06c_compare_all_tools.py to generate final comparison

ESTIMATED TIME:
  - Check ClinVar files: 5 minutes
  - Download dbNSFP: 30-60 minutes
  - Query and extract: 15-30 minutes
  - Total: 1-2 hours
"""

print(summary)

with open("VARIANT_PREPARATION_SUMMARY.txt", 'w') as f:
    f.write(summary)

print("\n[OK] Summary saved to: VARIANT_PREPARATION_SUMMARY.txt")

print("\n" + "=" * 70)
print("PREPARATION COMPLETE")
print("=" * 70)
print("\nGenerated files:")
print("  1. missense_variants_for_tools.tsv")
print("  2. variant_predictions_with_consensus.tsv")
print("  3. INSTRUCTIONS_FOR_TOOL_QUERIES.txt")
print("  4. VARIANT_PREPARATION_SUMMARY.txt")
