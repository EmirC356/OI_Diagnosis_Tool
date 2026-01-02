"""
Create SIFT Input Files
SIFT requires:
1. Protein sequence file (FASTA format)
2. Substitutions file (listing variants to test)
"""

import pandas as pd
from pathlib import Path

print("="*70)
print("CREATING SIFT INPUT FILES")
print("="*70)

# Load test variants
df = pd.read_csv("test_variants_for_revel.tsv", sep='\t')
print(f"\nLoaded {len(df)} test variants")

# ===================================================================
# PART 1: Get Protein Sequences from UniProt
# ===================================================================

print("\n1. Protein Sequences")
print("-" * 70)

# COL1A1 and COL1A2 protein sequences
# These should be downloaded from UniProt, but we'll create the structure

protein_sequences = {
    'COL1A1': {
        'uniprot_id': 'P02452',
        'gene': 'COL1A1',
        'length': 1464,  # amino acids
        'url': 'https://www.uniprot.org/uniprot/P02452.fasta'
    },
    'COL1A2': {
        'uniprot_id': 'P08123',
        'gene': 'COL1A2',
        'length': 1366,  # amino acids
        'url': 'https://www.uniprot.org/uniprot/P08123.fasta'
    }
}

# Create instructions for downloading sequences
instructions = """
TO GET PROTEIN SEQUENCES FOR SIFT:

1. Download COL1A1 sequence:
   URL: https://www.uniprot.org/uniprot/P02452.fasta
   Save as: COL1A1_protein.fasta

2. Download COL1A2 sequence:
   URL: https://www.uniprot.org/uniprot/P08123.fasta
   Save as: COL1A2_protein.fasta

OR use the pre-formatted sequences below:
"""

print(instructions)

# ===================================================================
# PART 2: Create Substitutions File
# ===================================================================

print("\n2. Creating Substitutions File")
print("-" * 70)

# Group variants by gene
df_col1a1 = df[df['Name'].str.contains('COL1A1', na=False)].copy()
df_col1a2 = df[df['Name'].str.contains('COL1A2', na=False)].copy()

print(f"COL1A1 variants: {len(df_col1a1)}")
print(f"COL1A2 variants: {len(df_col1a2)}")

# Extract position and amino acid changes from protein change
def parse_protein_change(change):
    """
    Parse protein change like 'F1463L' into (position, ref_aa, alt_aa)
    Returns: (1463, 'F', 'L')
    """
    import re
    if pd.isna(change):
        return None, None, None

    change = str(change).strip()
    # Match pattern: Letter + Number + Letter
    match = re.match(r'([A-Z])(\d+)([A-Z])', change)
    if match:
        ref_aa = match.group(1)
        pos = int(match.group(2))
        alt_aa = match.group(3)
        return pos, ref_aa, alt_aa
    return None, None, None

# Create SIFT substitution file for COL1A1
print("\nCreating COL1A1 substitutions file...")
sift_col1a1_subs = []
for idx, row in df_col1a1.iterrows():
    pos, ref_aa, alt_aa = parse_protein_change(row['Protein change'])
    if pos is not None:
        # SIFT format: position,reference_aa,substitution_aa
        sift_col1a1_subs.append({
            'position': pos,
            'reference': ref_aa,
            'substitution': alt_aa,
            'VariationID': row['VariationID'],
            'label': row['label']
        })

df_sift_col1a1 = pd.DataFrame(sift_col1a1_subs)

# Create SIFT substitution file for COL1A2
print("Creating COL1A2 substitutions file...")
sift_col1a2_subs = []
for idx, row in df_col1a2.iterrows():
    pos, ref_aa, alt_aa = parse_protein_change(row['Protein change'])
    if pos is not None:
        sift_col1a2_subs.append({
            'position': pos,
            'reference': ref_aa,
            'substitution': alt_aa,
            'VariationID': row['VariationID'],
            'label': row['label']
        })

df_sift_col1a2 = pd.DataFrame(sift_col1a2_subs)

# ===================================================================
# PART 3: Save SIFT Substitutions Files
# ===================================================================

print("\n3. Saving SIFT Substitutions Files")
print("-" * 70)

# SIFT format: position,reference_aa,new_aa
# One substitution per line

# COL1A1 substitutions
sift_col1a1_file = "sift_COL1A1_substitutions.txt"
with open(sift_col1a1_file, 'w') as f:
    for idx, row in df_sift_col1a1.iterrows():
        f.write(f"{row['position']},{row['reference']},{row['substitution']}\n")

print(f"[OK] Saved: {sift_col1a1_file}")
print(f"    Variants: {len(df_sift_col1a1)}")
print(f"    Format: position,reference_aa,substitution_aa")
print(f"    Example: {df_sift_col1a1.iloc[0]['position']},{df_sift_col1a1.iloc[0]['reference']},{df_sift_col1a1.iloc[0]['substitution']}")

# COL1A2 substitutions
sift_col1a2_file = "sift_COL1A2_substitutions.txt"
with open(sift_col1a2_file, 'w') as f:
    for idx, row in df_sift_col1a2.iterrows():
        f.write(f"{row['position']},{row['reference']},{row['substitution']}\n")

print(f"\n[OK] Saved: {sift_col1a2_file}")
print(f"    Variants: {len(df_sift_col1a2)}")

# Also save with VariationID mapping for later analysis
df_sift_col1a1.to_csv("sift_COL1A1_substitutions_with_ids.tsv", sep='\t', index=False)
df_sift_col1a2.to_csv("sift_COL1A2_substitutions_with_ids.tsv", sep='\t', index=False)

print(f"\n[OK] Saved mapping files with VariationIDs and labels")

# ===================================================================
# PART 4: Create Protein Sequence Files (Download from UniProt)
# ===================================================================

print("\n4. Creating Protein Sequence Download Instructions")
print("-" * 70)

# Create script to download sequences
download_script = """#!/bin/bash
# Download COL1A1 and COL1A2 protein sequences from UniProt

echo "Downloading COL1A1 protein sequence..."
curl -o COL1A1_protein.fasta "https://www.uniprot.org/uniprot/P02452.fasta"

echo "Downloading COL1A2 protein sequence..."
curl -o COL1A2_protein.fasta "https://www.uniprot.org/uniprot/P08123.fasta"

echo "Download complete!"
echo "Files created:"
echo "  - COL1A1_protein.fasta"
echo "  - COL1A2_protein.fasta"
"""

with open("download_protein_sequences.sh", 'w') as f:
    f.write(download_script)

print("[OK] Saved: download_protein_sequences.sh")
print("    Run with: bash download_protein_sequences.sh")

# Create Windows batch version
download_bat = """@echo off
REM Download COL1A1 and COL1A2 protein sequences from UniProt

echo Downloading COL1A1 protein sequence...
curl -o COL1A1_protein.fasta "https://www.uniprot.org/uniprot/P02452.fasta"

echo Downloading COL1A2 protein sequence...
curl -o COL1A2_protein.fasta "https://www.uniprot.org/uniprot/P08123.fasta"

echo Download complete!
echo Files created:
echo   - COL1A1_protein.fasta
echo   - COL1A2_protein.fasta
pause
"""

with open("download_protein_sequences.bat", 'w') as f:
    f.write(download_bat)

print("[OK] Saved: download_protein_sequences.bat")
print("    Run with: download_protein_sequences.bat")

# ===================================================================
# PART 5: Create SIFT Submission Guide
# ===================================================================

print("\n5. Creating SIFT Submission Guide")
print("-" * 70)

guide = f"""
SIFT INPUT FILES - SUBMISSION GUIDE
{"="*70}

FILES CREATED:

1. sift_COL1A1_substitutions.txt ({len(df_sift_col1a1)} variants)
   Format: position,reference_aa,substitution_aa
   Example: 1463,F,L

2. sift_COL1A2_substitutions.txt ({len(df_sift_col1a2)} variants)
   Format: position,reference_aa,substitution_aa

3. sift_COL1A1_substitutions_with_ids.tsv
   Contains VariationID and true labels for analysis

4. sift_COL1A2_substitutions_with_ids.tsv
   Contains VariationID and true labels for analysis

PROTEIN SEQUENCES NEEDED:

You need to download protein sequences from UniProt:

Option 1: Use download script
   Windows: download_protein_sequences.bat
   Linux/Mac: bash download_protein_sequences.sh

Option 2: Manual download
   COL1A1: https://www.uniprot.org/uniprot/P02452.fasta
   COL1A2: https://www.uniprot.org/uniprot/P08123.fasta

{"="*70}
HOW TO USE SIFT WEB INTERFACE
{"="*70}

SIFT Website: http://sift.bii.a-star.edu.sg/

Method 1: SIFT Sequences (Submit with Protein Sequence)
---------------------------------------------------------

1. Go to: http://sift.bii.a-star.edu.sg/www/SIFT_seq_submit2.html

2. Upload Files:
   - Protein sequence: COL1A1_protein.fasta
   - Substitutions: sift_COL1A1_substitutions.txt

3. Settings:
   - Database: UniRef90 (recommended)
   - Median sequence conservation: 3.00

4. Submit and wait for results (may take 15-30 minutes)

5. Repeat for COL1A2:
   - Protein sequence: COL1A2_protein.fasta
   - Substitutions: sift_COL1A2_substitutions.txt

Method 2: SIFT BLink (Faster, if sequences in database)
--------------------------------------------------------

1. Go to: http://sift.bii.a-star.edu.sg/www/SIFT_BLink_submit.html

2. Enter UniProt ID:
   For COL1A1: P02452
   For COL1A2: P08123

3. Upload substitutions file:
   sift_COL1A1_substitutions.txt (or COL1A2)

4. Submit

{"="*70}
INTERPRETING SIFT RESULTS
{"="*70}

SIFT Output Format:
-------------------
Position  Ref  Sub  SIFT_Score  Prediction  Median_Info
1463      F    L    0.00        DELETERIOUS 3.00

Prediction Thresholds:
----------------------
SIFT Score <= 0.05: DELETERIOUS (damaging, pathogenic)
SIFT Score > 0.05:  TOLERATED (benign)

Lower scores = more deleterious

{"="*70}
AFTER GETTING SIFT RESULTS
{"="*70}

1. Download SIFT results file

2. Parse results and match with VariationIDs:

   import pandas as pd

   # Load SIFT results
   df_sift = pd.read_csv("sift_results.txt", sep='\\t')

   # Load mapping with VariationIDs
   df_mapping = pd.read_csv("sift_COL1A1_substitutions_with_ids.tsv", sep='\\t')

   # Merge
   df_merged = df_mapping.merge(df_sift,
                                  left_on=['position', 'reference', 'substitution'],
                                  right_on=['Position', 'Ref', 'Sub'])

   # Compare predictions with true labels
   df_merged['SIFT_pred'] = (df_merged['SIFT_Score'] <= 0.05).astype(int)

   # Calculate accuracy
   from sklearn.metrics import accuracy_score
   accuracy = accuracy_score(df_merged['label'], df_merged['SIFT_pred'])
   print(f"SIFT Accuracy: {{accuracy:.2%}}")

3. Compare with your ML model performance

{"="*70}
ALTERNATIVE: PROVEAN (SIFT Alternative)
{"="*70}

If SIFT is slow or unavailable, try PROVEAN:
http://provean.jcvi.org/seq_submit.php

PROVEAN uses similar input format and provides similar predictions.

{"="*70}
SUMMARY
{"="*70}

Ready to submit:
1. Download protein sequences (run download script)
2. Upload to SIFT web interface:
   - COL1A1_protein.fasta + sift_COL1A1_substitutions.txt
   - COL1A2_protein.fasta + sift_COL1A2_substitutions.txt
3. Wait for results (15-30 min each)
4. Parse results and compare with your ML model

Total variants to test: {len(df_sift_col1a1) + len(df_sift_col1a2)}
"""

with open("SIFT_SUBMISSION_GUIDE.txt", 'w') as f:
    f.write(guide)

print(guide)

print("\n[OK] Guide saved to: SIFT_SUBMISSION_GUIDE.txt")

# ===================================================================
# Summary
# ===================================================================

print("\n" + "="*70)
print("SIFT INPUT FILES CREATED")
print("="*70)
print(f"\nSubstitution files:")
print(f"  1. sift_COL1A1_substitutions.txt ({len(df_sift_col1a1)} variants)")
print(f"  2. sift_COL1A2_substitutions.txt ({len(df_sift_col1a2)} variants)")
print(f"\nMapping files (with IDs and labels):")
print(f"  3. sift_COL1A1_substitutions_with_ids.tsv")
print(f"  4. sift_COL1A2_substitutions_with_ids.tsv")
print(f"\nDownload scripts:")
print(f"  5. download_protein_sequences.sh (Linux/Mac)")
print(f"  6. download_protein_sequences.bat (Windows)")
print(f"\nGuide:")
print(f"  7. SIFT_SUBMISSION_GUIDE.txt")
print(f"\nNext steps:")
print(f"  1. Run: download_protein_sequences.bat")
print(f"  2. Go to: http://sift.bii.a-star.edu.sg/")
print(f"  3. Upload protein sequences + substitution files")
