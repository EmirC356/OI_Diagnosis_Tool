"""
Create Test Files for REVEL Comparison
Extracts test variants and creates files in formats suitable for REVEL and other tools.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

print("="*70)
print("CREATING TEST FILES FOR REVEL COMPARISON")
print("="*70)

# Set random seed for reproducibility (same as ML models)
np.random.seed(42)

# ===================================================================
# PART 1: Load Data and Extract Test Set
# ===================================================================

print("\n1. Loading Data")
print("-" * 70)

# Load feature matrix (same as used in ML models)
df_features = pd.read_csv(Path("data") / "feature_matrix.csv")

# Load original data with variant details
df_original = pd.read_csv(Path("data") / "cleaned_COL1_variants.csv")

# Load full ClinVar data for genomic coordinates
df_col1a1 = pd.read_csv(Path("data") / "COL1A1_All.txt", sep='\t', dtype=str, low_memory=False)
df_col1a2 = pd.read_csv(Path("data") / "COL1A2_All.txt", sep='\t', dtype=str, low_memory=False)
df_clinvar = pd.concat([df_col1a1, df_col1a2], ignore_index=True)

print(f"Loaded {len(df_features)} variants")

# ===================================================================
# PART 2: Extract Test Fold from Cross-Validation
# ===================================================================

print("\n2. Extracting Test Fold from Cross-Validation")
print("-" * 70)

# Use same cross-validation split as ML models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get indices for one test fold (use fold 1 for consistency)
X = df_features.drop(['label'], axis=1)
y = df_features['label']

fold_idx = 0
for train_idx, test_idx in cv.split(X, y):
    if fold_idx == 0:  # Use first fold as test set
        break
    fold_idx += 1

# Extract test set
df_test = df_features.iloc[test_idx].copy()
print(f"Test set size: {len(df_test)} variants")
print(f"  Pathogenic: {(df_test['label']==1).sum()}")
print(f"  Benign: {(df_test['label']==0).sum()}")

# Get VariationIDs for test set
test_variation_ids = df_test['VariationID'].values
print(f"Sample test VariationIDs: {test_variation_ids[:5]}")

# Merge with original data to get variant details using VariationID
df_test = df_test.merge(
    df_original[['VariationID', 'Name', 'Gene(s)', 'Protein change',
                 'Molecular consequence', 'Variant type']],
    on='VariationID',
    how='left',
    suffixes=('', '_orig')
)

print(f"Columns after merge: {df_test.columns.tolist()}")
print(f"Has 'Molecular consequence': {'Molecular consequence' in df_test.columns}")

# ===================================================================
# PART 3: Filter for Missense Variants (REVEL works on missense)
# ===================================================================

print("\n3. Filtering for Missense Variants")
print("-" * 70)

df_test_missense = df_test[df_test['Molecular consequence'] == 'missense variant'].copy()
print(f"Missense variants in test set: {len(df_test_missense)}")
print(f"  Pathogenic: {(df_test_missense['label']==1).sum()}")
print(f"  Benign: {(df_test_missense['label']==0).sum()}")

# ===================================================================
# PART 4: Merge with Genomic Coordinates
# ===================================================================

print("\n4. Adding Genomic Coordinates")
print("-" * 70)

# Merge with ClinVar data to get genomic coordinates
# Convert VariationID to string for matching (ClinVar has string type)
df_test_missense['VariationID_str'] = df_test_missense['VariationID'].astype(str)
df_clinvar['VariationID_str'] = df_clinvar['VariationID'].astype(str)

df_test_missense = df_test_missense.merge(
    df_clinvar[['VariationID', 'GRCh38Chromosome', 'GRCh38Location',
                'dbSNP ID', 'Canonical SPDI']].assign(VariationID_str=lambda x: x['VariationID'].astype(str)),
    on='VariationID_str',
    how='left',
    suffixes=('', '_clinvar')
)

# Drop the temporary string column
df_test_missense = df_test_missense.drop(columns=['VariationID_str', 'VariationID_clinvar'], errors='ignore')

# Parse GRCh38 coordinates
def parse_coordinates(row):
    """Extract chromosome, position, ref, alt from GRCh38 location"""
    chrom = row['GRCh38Chromosome']
    loc = row['GRCh38Location']

    if pd.isna(chrom) or pd.isna(loc):
        return None, None, None, None

    # Extract position (handle ranges by taking first position)
    loc_str = str(loc)
    if '-' in loc_str:
        pos = loc_str.split('-')[0]
    elif ':' in loc_str:
        pos = loc_str.split(':')[0]
    else:
        pos = loc_str

    # Try to extract ref/alt from variant name
    import re
    name = row['Name']
    match = re.search(r'c\.\d+([A-Z])>([A-Z])', str(name))
    if match:
        ref = match.group(1)
        alt = match.group(2)
        return chrom, pos, ref, alt

    return chrom, pos, None, None

print("Parsing genomic coordinates...")
coords = df_test_missense.apply(parse_coordinates, axis=1)
df_test_missense['chr'] = coords.apply(lambda x: x[0])
df_test_missense['pos'] = coords.apply(lambda x: x[1])
df_test_missense['ref'] = coords.apply(lambda x: x[2])
df_test_missense['alt'] = coords.apply(lambda x: x[3])

# Count how many have full coordinates
has_coords = (
    df_test_missense['chr'].notna() &
    df_test_missense['pos'].notna() &
    df_test_missense['ref'].notna() &
    df_test_missense['alt'].notna()
).sum()

print(f"Variants with complete coordinates: {has_coords}/{len(df_test_missense)}")

# ===================================================================
# PART 5: Create REVEL Input File (VCF-like format)
# ===================================================================

print("\n5. Creating REVEL Input File")
print("-" * 70)

# REVEL typically uses: chr, pos, ref, alt
# Create simplified variant file
df_revel_input = df_test_missense[
    df_test_missense['chr'].notna() &
    df_test_missense['pos'].notna()
].copy()

# Create REVEL input format
revel_file = "test_variants_for_revel.tsv"
df_revel_input[['VariationID', 'chr', 'pos', 'ref', 'alt', 'Name',
                'Protein change', 'label']].to_csv(
    revel_file, sep='\t', index=False
)

print(f"[OK] Saved REVEL input file: {revel_file}")
print(f"    Format: VariationID, chr, pos, ref, alt, Name, Protein change, label")
print(f"    Variants: {len(df_revel_input)}")

# ===================================================================
# PART 6: Create Protein Sequence FASTA File
# ===================================================================

print("\n6. Creating Protein Sequence FASTA File")
print("-" * 70)

# For protein-based tools, create FASTA with protein sequences
# Note: We don't have full protein sequences, but we can create reference entries

fasta_file = "test_variants_proteins.fasta"
with open(fasta_file, 'w') as f:
    # COL1A1 protein sequence (from UniProt: P02452)
    # Note: This is a simplified version - you'd want the full sequence for real analysis
    f.write(">sp|P02452|CO1A1_HUMAN Collagen alpha-1(I) chain\n")
    f.write("# Full protein sequence would go here\n")
    f.write("# For actual use, download from UniProt: https://www.uniprot.org/uniprot/P02452\n\n")

    # COL1A2 protein sequence (from UniProt: P08123)
    f.write(">sp|P08123|CO1A2_HUMAN Collagen alpha-2(I) chain\n")
    f.write("# Full protein sequence would go here\n")
    f.write("# For actual use, download from UniProt: https://www.uniprot.org/uniprot/P08123\n\n")

    # Add variant information as comments
    f.write("# MISSENSE VARIANTS IN TEST SET:\n")
    for idx, row in df_test_missense.head(20).iterrows():  # Show first 20
        f.write(f"# {row['VariationID']}: {row['Name']} - {row['Protein change']} "
                f"(Label: {'Pathogenic' if row['label']==1 else 'Benign'})\n")

print(f"[OK] Saved FASTA template: {fasta_file}")
print(f"    Note: Contains UniProt references and variant list")

# ===================================================================
# PART 7: Create VCF-like File for Web Tools
# ===================================================================

print("\n7. Creating VCF-like File")
print("-" * 70)

# Create a simple VCF-like file for web upload
vcf_file = "test_variants.vcf"
with open(vcf_file, 'w') as f:
    # VCF header
    f.write("##fileformat=VCFv4.2\n")
    f.write("##reference=GRCh38\n")
    f.write("##INFO=<ID=VariationID,Number=1,Type=String,Description=\"ClinVar Variation ID\">\n")
    f.write("##INFO=<ID=Label,Number=1,Type=Integer,Description=\"True label: 1=Pathogenic, 0=Benign\">\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

    # Variant lines
    for idx, row in df_revel_input.iterrows():
        if pd.notna(row['chr']) and pd.notna(row['pos']) and pd.notna(row['ref']) and pd.notna(row['alt']):
            rsid = row['dbSNP ID'] if pd.notna(row['dbSNP ID']) else '.'
            info = f"VariationID={row['VariationID']};Label={int(row['label'])}"
            f.write(f"{row['chr']}\t{row['pos']}\t{rsid}\t{row['ref']}\t{row['alt']}\t.\t.\t{info}\n")

print(f"[OK] Saved VCF file: {vcf_file}")
print(f"    Format: Standard VCF with ClinVar VariationID and true labels")

# ===================================================================
# PART 8: Create Simple Variant List
# ===================================================================

print("\n8. Creating Simple Variant List")
print("-" * 70)

# Create human-readable list
list_file = "test_variants_list.txt"
with open(list_file, 'w') as f:
    f.write("TEST SET VARIANTS FOR REVEL COMPARISON\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total variants: {len(df_test)}\n")
    f.write(f"Missense variants: {len(df_test_missense)}\n")
    f.write(f"Variants with coordinates: {has_coords}\n\n")

    f.write("MISSENSE VARIANTS:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'VariationID':<12} {'Gene':<8} {'Variant':<30} {'Protein Change':<20} {'Label':<10}\n")
    f.write("-"*70 + "\n")

    for idx, row in df_test_missense.iterrows():
        var_id = str(row['VariationID'])[:12]
        gene = str(row['Gene(s)'])[:8]
        variant = str(row['Name'])[:30]
        protein = str(row['Protein change'])[:20]
        label = 'Pathogenic' if row['label'] == 1 else 'Benign'
        f.write(f"{var_id:<12} {gene:<8} {variant:<30} {protein:<20} {label:<10}\n")

print(f"[OK] Saved variant list: {list_file}")

# ===================================================================
# PART 9: Save Full Test Set
# ===================================================================

print("\n9. Saving Full Test Set")
print("-" * 70)

# Save complete test set with all information
test_file = "complete_test_set.tsv"
df_test.to_csv(test_file, sep='\t', index=False)
print(f"[OK] Saved complete test set: {test_file}")
print(f"    Contains all {len(df_test)} test variants with features and labels")

# Save missense test set
missense_test_file = "missense_test_set.tsv"
df_test_missense.to_csv(missense_test_file, sep='\t', index=False)
print(f"[OK] Saved missense test set: {missense_test_file}")
print(f"    Contains {len(df_test_missense)} missense variants for REVEL comparison")

# ===================================================================
# PART 10: Summary and Instructions
# ===================================================================

print("\n10. Summary and Instructions")
print("="*70)

summary = f"""
TEST FILES CREATED FOR REVEL COMPARISON

Dataset Summary:
  - Total test variants: {len(df_test)}
  - Missense test variants: {len(df_test_missense)}
  - Variants with coordinates: {has_coords}

  Breakdown:
    Pathogenic: {(df_test['label']==1).sum()} ({(df_test['label']==1).sum()/len(df_test)*100:.1f}%)
    Benign: {(df_test['label']==0).sum()} ({(df_test['label']==0).sum()/len(df_test)*100:.1f}%)

Files Generated:
  1. test_variants_for_revel.tsv      - REVEL input format ({len(df_revel_input)} variants)
  2. test_variants.vcf                - VCF format for web tools
  3. test_variants_proteins.fasta     - Protein sequence template
  4. test_variants_list.txt           - Human-readable variant list
  5. complete_test_set.tsv            - Full test set with all features
  6. missense_test_set.tsv            - Missense variants only

HOW TO USE THESE FILES WITH REVEL:

Option 1: Pre-computed REVEL Scores (EASIEST)
----------------------------------------------
REVEL scores are pre-computed for all possible missense variants in dbNSFP.

1. Download dbNSFP (see TOOL_PREDICTIONS_STATUS.md for details)
2. Look up REVEL scores using chr:pos:ref:alt from test_variants_for_revel.tsv
3. Compare REVEL predictions with true labels

Option 2: REVEL Website
-----------------------
Visit: https://sites.google.com/site/revelgenomics/

Note: REVEL doesn't have a web interface for batch queries.
Scores are only available via dbNSFP database.

Option 3: Using dbNSFP with Our Query Script
--------------------------------------------
1. Download dbNSFP chr7 and chr17
2. Run: python 06b_query_dbnsfp.py
3. Filter results for VariationIDs in test_variants_for_revel.tsv

EVALUATION APPROACH:

Once you have REVEL scores for test variants:

1. Load test_variants_for_revel.tsv (ground truth labels)
2. Load REVEL predictions
3. Calculate metrics:
   - Accuracy: % correct predictions
   - Precision: % of REVEL "pathogenic" that are truly pathogenic
   - Recall: % of true pathogenic that REVEL catches
   - ROC-AUC: Area under ROC curve

4. Compare with our model's performance on same test set:
   - Our model: 97% accuracy on full dataset
   - REVEL: Expected ~90% accuracy (literature)
   - Direct comparison on same {len(df_test)} variants

EXPECTED RESULTS:

Based on literature and our analysis:
  - REVEL accuracy: ~85-90% (on missense variants)
  - Our model accuracy: ~97% (on all variant types)
  - Direct comparison will show disease-specific advantage

Note: REVEL is designed for missense variants only, while our model
handles all variant types (frameshift, nonsense, splice, etc.)
"""

print(summary)

# Save summary
with open("TEST_SET_INSTRUCTIONS.txt", 'w') as f:
    f.write(summary)

print("\n[OK] Instructions saved to: TEST_SET_INSTRUCTIONS.txt")

print("\n" + "="*70)
print("TEST FILE GENERATION COMPLETE")
print("="*70)
print("\nNext steps:")
print("  1. Download dbNSFP or use existing REVEL scores")
print("  2. Look up REVEL predictions for test variants")
print("  3. Compare REVEL vs. our model on same test set")
print("  4. Create final comparison visualization")
