"""
Create CORRECT SIFT Format
SIFT expects: X#Y format (e.g., F1463L)
Where X = original AA, # = position, Y = new AA
"""

import pandas as pd

print("="*70)
print("CREATING CORRECT SIFT FORMAT (X#Y)")
print("="*70)

# Load mapping files
df_col1a1 = pd.read_csv("sift_COL1A1_substitutions_with_ids.tsv", sep='\t')
df_col1a2 = pd.read_csv("sift_COL1A2_substitutions_with_ids.tsv", sep='\t')

print(f"\nCOL1A1 variants: {len(df_col1a1)}")
print(f"COL1A2 variants: {len(df_col1a2)}")

# Create SIFT format: X#Y (e.g., F1463L)
# Where: X = reference AA, # = position, Y = substitution AA

print("\n1. Creating COL1A1 substitutions (X#Y format)")
print("-" * 70)

with open("sift_COL1A1_substitutions_correct.txt", 'w') as f:
    for idx, row in df_col1a1.iterrows():
        ref = row['reference']
        pos = int(row['position'])
        alt = row['substitution']
        # Write in X#Y format: F1463L
        f.write(f"{ref}{pos}{alt}\n")

print(f"[OK] Created: sift_COL1A1_substitutions_correct.txt")

print("\n2. Creating COL1A2 substitutions (X#Y format)")
print("-" * 70)

with open("sift_COL1A2_substitutions_correct.txt", 'w') as f:
    for idx, row in df_col1a2.iterrows():
        ref = row['reference']
        pos = int(row['position'])
        alt = row['substitution']
        f.write(f"{ref}{pos}{alt}\n")

print(f"[OK] Created: sift_COL1A2_substitutions_correct.txt")

# Show samples
print("\n3. Sample Output")
print("=" * 70)

print("\nCOL1A1 (first 10 substitutions):")
with open("sift_COL1A1_substitutions_correct.txt", 'r') as f:
    for i, line in enumerate(f):
        if i < 10:
            print(f"  {line.strip()}")
        else:
            break

print("\nCOL1A2 (first 10 substitutions):")
with open("sift_COL1A2_substitutions_correct.txt", 'r') as f:
    for i, line in enumerate(f):
        if i < 10:
            print(f"  {line.strip()}")
        else:
            break

print("\n" + "=" * 70)
print("CORRECT FORMAT FILES CREATED")
print("=" * 70)
print("\nFiles ready for SIFT:")
print("  1. sift_COL1A1_substitutions_correct.txt (88 variants)")
print("  2. sift_COL1A2_substitutions_correct.txt (88 variants)")
print("\nFormat: X#Y (e.g., F1463L)")
print("  X = original amino acid")
print("  # = position")
print("  Y = new amino acid")
print("\nUpload these with:")
print("  - COL1A1_protein.fasta")
print("  - COL1A2_protein.fasta")
