"""
Query Ensembl VEP API for Tool Predictions
Alternative approach: Use Ensembl's REST API to get SIFT, PolyPhen-2, CADD predictions
without downloading the entire dbNSFP database.
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from pathlib import Path

print("="*70)
print("QUERYING ENSEMBL VEP API FOR TOOL PREDICTIONS")
print("="*70)

# ===================================================================
# PART 1: Load Variants
# ===================================================================

print("\n1. Loading Variants")
print("-" * 70)

# Load original ClinVar data with all columns including genomic coordinates
df_col1a1 = pd.read_csv(Path("data") / "COL1A1_All.txt", sep='\t', dtype=str, low_memory=False)
df_col1a2 = pd.read_csv(Path("data") / "COL1A2_All.txt", sep='\t', dtype=str, low_memory=False)
df = pd.concat([df_col1a1, df_col1a2], ignore_index=True)
print(f"Loaded {len(df)} variants from ClinVar files")

# Load cleaned data to get labels
df_cleaned = pd.read_csv(Path("data") / "cleaned_COL1_variants.csv")
# Convert VariationID to string for both dataframes for proper matching
label_map = dict(zip(df_cleaned['VariationID'].astype(str), df_cleaned['label']))

# Add labels to full dataset
df['label'] = df['VariationID'].astype(str).map(label_map)

# Focus on missense variants first (most critical for tool predictions)
df_missense = df[df['Molecular consequence'] == 'missense variant'].copy()
# Only keep variants that have labels (i.e., were in cleaned dataset)
df_missense = df_missense[df_missense['label'].notna()].copy()
print(f"Missense variants to query: {len(df_missense)}")

# ===================================================================
# PART 2: Prepare VEP Query Format
# ===================================================================

print("\n2. Preparing Variants for VEP")
print("-" * 70)

def parse_grch38_location(loc):
    """Parse GRCh38 location to get start position"""
    if pd.isna(loc):
        return None
    loc_str = str(loc)
    if '-' in loc_str:
        return loc_str.split('-')[0]
    elif ':' in loc_str:
        return loc_str.split(':')[0]
    else:
        return loc_str

# Prepare VEP input format: chr-start-end-ref/alt-strand
vep_variants = []
for idx, row in df_missense.iterrows():
    chrom = row['GRCh38Chromosome']
    pos = parse_grch38_location(row['GRCh38Location'])

    if pd.isna(chrom) or pos is None:
        continue

    # Try to extract ref/alt from variant name
    # Example: NM_000088.4(COL1A1):c.3455G>A (p.Gly1152Asp)
    import re
    match = re.search(r'c\.\d+([A-Z])>([A-Z])', row['Name'])
    if match:
        ref = match.group(1)
        alt = match.group(2)

        # VEP format: chr-start-end-ref/alt-1
        vep_id = f"{chrom}-{pos}-{pos}-{ref}/{alt}-1"
        vep_variants.append({
            'id': vep_id,
            'VariationID': row['VariationID'],
            'Name': row['Name'],
            'true_label': row['label']
        })

print(f"Prepared {len(vep_variants)} variants for VEP query")

# ===================================================================
# PART 3: Query Ensembl VEP API
# ===================================================================

print("\n3. Querying Ensembl VEP API")
print("-" * 70)
print("\nNOTE: VEP API has rate limits (15 requests/second, max 200 variants/request)")
print(f"      This will take approximately {len(vep_variants)/200*2:.1f} minutes for {len(vep_variants)} variants")

# Ensembl VEP REST API endpoint
vep_endpoint = "https://rest.ensembl.org/vep/human/region"

# Process in batches of 200 (API limit)
batch_size = 200
results = []

for batch_start in range(0, len(vep_variants), batch_size):
    batch_end = min(batch_start + batch_size, len(vep_variants))
    batch = vep_variants[batch_start:batch_end]

    print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(vep_variants)-1)//batch_size + 1} ({len(batch)} variants)...")

    # Prepare POST request
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = {
        "variants": [v['id'] for v in batch],
        "SIFT": "p",  # Request SIFT predictions
        "PolyPhen": "p"  # Request PolyPhen predictions
    }

    try:
        response = requests.post(vep_endpoint, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            vep_results = response.json()

            # Parse results
            for i, result in enumerate(vep_results):
                variant_info = batch[i]

                # Extract predictions from transcript consequences
                sift_scores = []
                polyphen_scores = []

                if 'transcript_consequences' in result:
                    for tc in result['transcript_consequences']:
                        # SIFT
                        if 'sift_score' in tc:
                            sift_scores.append(tc['sift_score'])

                        # PolyPhen
                        if 'polyphen_score' in tc:
                            polyphen_scores.append(tc['polyphen_score'])

                # Use most deleterious prediction
                results.append({
                    'VariationID': variant_info['VariationID'],
                    'Name': variant_info['Name'],
                    'true_label': variant_info['true_label'],
                    'SIFT_score': min(sift_scores) if sift_scores else np.nan,  # Lower = more deleterious
                    'PolyPhen2_score': max(polyphen_scores) if polyphen_scores else np.nan,  # Higher = more deleterious
                    'num_transcripts': len(result.get('transcript_consequences', [])),
                    'prediction_source': 'Ensembl VEP'
                })

            print(f"  [OK] Retrieved predictions for {len(batch)} variants")
        else:
            print(f"  [ERROR] API returned status code {response.status_code}")
            print(f"  Response: {response.text[:200]}")

    except Exception as e:
        print(f"  [ERROR] Failed to query VEP: {e}")

    # Rate limiting: wait 1 second between batches
    if batch_end < len(vep_variants):
        time.sleep(1)

df_vep = pd.DataFrame(results)
print(f"\n[OK] Successfully retrieved predictions for {len(df_vep)} variants")

# ===================================================================
# PART 4: Convert Scores to Predictions
# ===================================================================

print("\n4. Converting Scores to Predictions")
print("-" * 70)

if len(df_vep) > 0:
    # SIFT: score <= 0.05 = deleterious
    df_vep['SIFT_pred'] = df_vep['SIFT_score'].apply(
        lambda x: 'Deleterious' if not pd.isna(x) and x <= 0.05 else ('Tolerated' if not pd.isna(x) else 'Unknown')
    )

    # PolyPhen-2: score > 0.85 = probably damaging, 0.15-0.85 = possibly damaging, < 0.15 = benign
    df_vep['PolyPhen2_pred'] = df_vep['PolyPhen2_score'].apply(
        lambda x: 'Probably Damaging' if not pd.isna(x) and x > 0.85 else (
            'Possibly Damaging' if not pd.isna(x) and x >= 0.15 else (
                'Benign' if not pd.isna(x) else 'Unknown'
            )
        )
    )

    # Show prediction summary
    print("\nSIFT predictions:")
    print(df_vep['SIFT_pred'].value_counts())

    print("\nPolyPhen-2 predictions:")
    print(df_vep['PolyPhen2_pred'].value_counts())
else:
    print("\n[WARNING] No variants were successfully queried from VEP")

# Save VEP results
vep_file = "missense_vep_predictions.tsv"
df_vep.to_csv(vep_file, sep='\t', index=False)
print(f"\n[OK] Saved VEP predictions to: {vep_file}")

# ===================================================================
# PART 5: Add CADD and REVEL Scores (if available from ClinVar)
# ===================================================================

print("\n5. Checking for CADD/REVEL in ClinVar Data")
print("-" * 70)

# CADD and REVEL are not available through Ensembl VEP
# We'll assign default values for now
print("\n[NOTE] CADD and REVEL scores not available through VEP API")
print("       Options:")
print("       1. Use CADD web interface for individual queries")
print("       2. Download full dbNSFP (~20GB) for pre-computed scores")
print("       3. Use literature-reported average values as proxy")
print("\n       For now, we'll leave CADD and REVEL as missing (NaN)")

df_vep['CADD_score'] = np.nan
df_vep['REVEL_score'] = np.nan

# ===================================================================
# PART 6: Merge with Consensus Predictions
# ===================================================================

print("\n6. Merging with Consensus Predictions")
print("-" * 70)

# Load consensus predictions
df_consensus = pd.read_csv("variant_predictions_with_consensus.tsv", sep='\t')

# Update predictions for VEP-queried variants
for idx, vep_result in df_vep.iterrows():
    var_id = vep_result['VariationID']
    mask = df_consensus['VariationID'] == var_id

    df_consensus.loc[mask, 'SIFT_pred'] = vep_result['SIFT_pred']
    df_consensus.loc[mask, 'SIFT_score'] = vep_result['SIFT_score']
    df_consensus.loc[mask, 'PolyPhen2_pred'] = vep_result['PolyPhen2_pred']
    df_consensus.loc[mask, 'PolyPhen2_score'] = vep_result['PolyPhen2_score']
    df_consensus.loc[mask, 'prediction_source'] = 'Ensembl VEP'

# Count prediction sources
print("\nFinal prediction source distribution:")
source_counts = df_consensus['prediction_source'].value_counts()
for source, count in source_counts.items():
    print(f"  {source:40s}: {count:4d} ({count/len(df_consensus)*100:5.1f}%)")

# Save final predictions
final_file = "all_variants_with_tool_predictions_vep.tsv"
df_consensus.to_csv(final_file, sep='\t', index=False)
print(f"\n[OK] Saved final predictions to: {final_file}")

# ===================================================================
# PART 7: Summary and Next Steps
# ===================================================================

print("\n7. Summary")
print("=" * 70)

summary = f"""
ENSEMBL VEP QUERY RESULTS:

Total variants: {len(df_consensus)}

Predictions obtained:
  - SIFT scores:       {df_vep['SIFT_score'].notna().sum()}/{len(df_vep)} ({df_vep['SIFT_score'].notna().sum()/len(df_vep)*100:.1f}%)
  - PolyPhen-2 scores: {df_vep['PolyPhen2_score'].notna().sum()}/{len(df_vep)} ({df_vep['PolyPhen2_score'].notna().sum()/len(df_vep)*100:.1f}%)

Prediction sources:
  - Ensembl VEP (missense):         {len(df_vep):4d} ({len(df_vep)/len(df_consensus)*100:5.1f}%)
  - Consensus (loss-of-function):   {(df_consensus['prediction_source']=='Consensus (Loss-of-Function)').sum():4d}
  - Consensus (silent/non-coding):  {(df_consensus['prediction_source']=='Consensus (Silent/Non-coding)').sum():4d}
  - Pending:                        {(df_consensus['prediction_source']=='Pending').sum():4d}

LIMITATIONS:
  - CADD and REVEL scores not available through VEP API
  - Only missense variants queried (other variant types use consensus)

NEXT STEPS:
  1. Run 06d_compare_with_real_tools.py to compare ML models with actual tool predictions
  2. For CADD/REVEL:
     - Option A: Download full dbNSFP database (~20GB)
     - Option B: Use literature-reported average performance as proxy

FILES GENERATED:
  1. missense_vep_predictions.tsv              - VEP predictions for {len(df_vep)} missense variants
  2. all_variants_with_tool_predictions_vep.tsv - All {len(df_consensus)} variants with predictions
"""

print(summary)

with open("VEP_QUERY_SUMMARY.txt", 'w') as f:
    f.write(summary)

print("\n[OK] Summary saved to: VEP_QUERY_SUMMARY.txt")

print("\n" + "=" * 70)
print("VEP QUERY COMPLETE")
print("=" * 70)
