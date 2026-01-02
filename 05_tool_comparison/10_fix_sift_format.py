"""
Fix SIFT Substitutions File Format
Ensure proper format with no extra characters
"""

import pandas as pd

print("Creating properly formatted SIFT files...")

# Load the mapping files with IDs
df_col1a1 = pd.read_csv("sift_COL1A1_substitutions_with_ids.tsv", sep='\t')
df_col1a2 = pd.read_csv("sift_COL1A2_substitutions_with_ids.tsv", sep='\t')

print(f"COL1A1 variants: {len(df_col1a1)}")
print(f"COL1A2 variants: {len(df_col1a2)}")

# Create clean SIFT files with exact format
# Format: position,reference,substitution (no spaces, no header)

# COL1A1
with open("sift_COL1A1_substitutions_clean.txt", 'w', newline='') as f:
    for idx, row in df_col1a1.iterrows():
        # Ensure single letter amino acids
        ref = str(row['reference']).strip()
        sub = str(row['substitution']).strip()
        pos = int(row['position'])

        # Write in exact format
        f.write(f"{pos},{ref},{sub}\n")

print("[OK] Created: sift_COL1A1_substitutions_clean.txt")

# COL1A2
with open("sift_COL1A2_substitutions_clean.txt", 'w', newline='') as f:
    for idx, row in df_col1a2.iterrows():
        ref = str(row['reference']).strip()
        sub = str(row['substitution']).strip()
        pos = int(row['position'])
        f.write(f"{pos},{ref},{sub}\n")

print("[OK] Created: sift_COL1A2_substitutions_clean.txt")

# Also create space-separated format (alternative SIFT format)
print("\nCreating space-separated format...")

with open("sift_COL1A1_substitutions_spaced.txt", 'w') as f:
    for idx, row in df_col1a1.iterrows():
        f.write(f"{int(row['position'])} {row['reference']} {row['substitution']}\n")

with open("sift_COL1A2_substitutions_spaced.txt", 'w') as f:
    for idx, row in df_col1a2.iterrows():
        f.write(f"{int(row['position'])} {row['reference']} {row['substitution']}\n")

print("[OK] Created space-separated versions")

# Show samples
print("\n=== Sample COL1A1 (comma-separated) ===")
with open("sift_COL1A1_substitutions_clean.txt", 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(line.strip())
        else:
            break

print("\n=== Sample COL1A1 (space-separated) ===")
with open("sift_COL1A1_substitutions_spaced.txt", 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(line.strip())
        else:
            break

print("\n" + "="*70)
print("FILES CREATED:")
print("="*70)
print("Comma-separated (standard):")
print("  - sift_COL1A1_substitutions_clean.txt")
print("  - sift_COL1A2_substitutions_clean.txt")
print("\nSpace-separated (alternative):")
print("  - sift_COL1A1_substitutions_spaced.txt")
print("  - sift_COL1A2_substitutions_spaced.txt")
print("\nTry uploading the 'clean' versions first.")
print("If error persists, try 'spaced' versions.")
