"""
Create Properly Formatted VEP Input File
Based on Ensembl VEP format specifications
"""

import pandas as pd

print("="*70)
print("CREATING VEP-FORMATTED INPUT FILE")
print("="*70)

# Load test variants
df = pd.read_csv("test_variants_for_revel.tsv", sep='\t')
print(f"\nLoaded {len(df)} test variants")

# ===================================================================
# Format 1: Default VEP Format
# ===================================================================

print("\n1. Creating Default VEP Format File")
print("-" * 70)

# Default VEP format: chromosome start end allele strand [identifier]
# For SNVs: start = end = position
# Alleles: ref/alt

vep_default = []
for idx, row in df.iterrows():
    if pd.notna(row['chr']) and pd.notna(row['pos']) and pd.notna(row['ref']) and pd.notna(row['alt']):
        # Format: chr start end ref/alt strand identifier
        chrom = str(row['chr'])
        pos = str(row['pos'])
        allele = f"{row['ref']}/{row['alt']}"
        strand = "+"  # Assume positive strand (can be adjusted if known)
        identifier = f"var_{row['VariationID']}"

        vep_default.append({
            'chromosome': chrom,
            'start': pos,
            'end': pos,  # For SNVs, start = end
            'allele': allele,
            'strand': strand,
            'identifier': identifier
        })

df_vep_default = pd.DataFrame(vep_default)

# Save without header as per VEP specification
vep_default_file = "test_variants_vep_default.txt"
df_vep_default.to_csv(vep_default_file, sep='\t', header=False, index=False)
print(f"[OK] Saved: {vep_default_file}")
print(f"    Format: chromosome start end allele strand identifier")
print(f"    Variants: {len(df_vep_default)}")
print(f"    Example: {df_vep_default.iloc[0].tolist()}")

# ===================================================================
# Format 2: VCF Format (Standard)
# ===================================================================

print("\n2. Creating Standard VCF Format File")
print("-" * 70)

vcf_file = "test_variants_vep.vcf"
with open(vcf_file, 'w') as f:
    # VCF header
    f.write("##fileformat=VCFv4.2\n")
    f.write("##reference=GRCh38\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

    # Variant lines
    for idx, row in df.iterrows():
        if pd.notna(row['chr']) and pd.notna(row['pos']) and pd.notna(row['ref']) and pd.notna(row['alt']):
            chrom = str(row['chr'])
            pos = str(row['pos'])
            var_id = f"var_{row['VariationID']}"
            ref = str(row['ref'])
            alt = str(row['alt'])

            f.write(f"{chrom}\t{pos}\t{var_id}\t{ref}\t{alt}\t.\t.\t.\n")

print(f"[OK] Saved: {vcf_file}")
print(f"    Format: Standard VCF")
print(f"    Variants: {len(df)}")

# ===================================================================
# Format 3: HGVS Format (using variant names from ClinVar)
# ===================================================================

print("\n3. Creating HGVS Format File")
print("-" * 70)

hgvs_file = "test_variants_vep_hgvs.txt"
with open(hgvs_file, 'w') as f:
    for idx, row in df.iterrows():
        # Extract HGVS notation from ClinVar name
        # Example: NM_000088.4(COL1A1):c.4387T>C (p.Phe1463Leu)
        name = str(row['Name'])
        if ':' in name and 'c.' in name:
            # Extract just the transcript:c.change part
            parts = name.split('(')
            if len(parts) >= 2:
                transcript_part = parts[0]  # e.g., NM_000088.4
                change_part = parts[1].split(')')[1] if ')' in parts[1] else ''

                # Extract c. change
                if ':c.' in name:
                    hgvs = name.split('(p.')[0].strip()  # Remove protein change
                    f.write(f"{hgvs}\n")

print(f"[OK] Saved: {hgvs_file}")
print(f"    Format: HGVS notation")

# ===================================================================
# Format 4: rsID Format (if available)
# ===================================================================

print("\n4. Creating rsID Format File (if available)")
print("-" * 70)

# Load full ClinVar data to get dbSNP IDs
df_full = pd.read_csv("missense_test_set.tsv", sep='\t')

rsids = []
for idx, row in df_full.iterrows():
    if 'dbSNP ID' in row and pd.notna(row['dbSNP ID']):
        dbsnp = str(row['dbSNP ID'])
        if dbsnp.startswith('rs'):
            rsids.append(dbsnp)

if len(rsids) > 0:
    rsid_file = "test_variants_vep_rsid.txt"
    with open(rsid_file, 'w') as f:
        for rsid in rsids:
            f.write(f"{rsid}\n")

    print(f"[OK] Saved: {rsid_file}")
    print(f"    Format: rsID")
    print(f"    Variants with rsIDs: {len(rsids)}")
else:
    print("[NOTE] No rsIDs found in test set")

# ===================================================================
# Summary
# ===================================================================

print("\n5. Summary")
print("="*70)

summary = f"""
VEP INPUT FILES CREATED

Files for Ensembl VEP submission:

1. test_variants_vep_default.txt ({len(df_vep_default)} variants)
   Format: chromosome start end allele strand identifier
   Use: Default VEP format (RECOMMENDED)

2. test_variants_vep.vcf ({len(df)} variants)
   Format: Standard VCF
   Use: VCF format (most compatible)

3. test_variants_vep_hgvs.txt
   Format: HGVS notation (NM_000088.4:c.4387T>C)
   Use: HGVS format (if you have transcript coordinates)

4. test_variants_vep_rsid.txt ({len(rsids) if len(rsids) > 0 else 0} variants)
   Format: rsID (rs699)
   Use: rsID lookup (if variants have dbSNP IDs)

HOW TO USE WITH ENSEMBL VEP WEB INTERFACE:

1. Go to: https://www.ensembl.org/Tools/VEP

2. Upload one of these files:
   - Recommended: test_variants_vep.vcf (most compatible)
   - Alternative: test_variants_vep_default.txt

3. Settings:
   - Species: Human
   - Reference: GRCh38
   - Additional annotations:
     [x] SIFT
     [x] PolyPhen
     [x] dbNSFP (for CADD, REVEL scores)

4. Run and download results

5. Compare VEP predictions with:
   - Ground truth labels in test_variants_for_revel.tsv
   - Your ML model predictions on same test set

COMMAND LINE VEP (if installed):

vep --input_file test_variants_vep.vcf \\
    --output_file vep_results.txt \\
    --format vcf \\
    --species homo_sapiens \\
    --assembly GRCh38 \\
    --sift b \\
    --polyphen b \\
    --plugin dbNSFP,REVEL_score,CADD_phred \\
    --offline

NEXT STEPS:

1. Submit test_variants_vep.vcf to Ensembl VEP
2. Download results (will include SIFT, PolyPhen scores)
3. Parse VEP output to extract predictions
4. Compare with ground truth in test_variants_for_revel.tsv
5. Calculate accuracy, precision, recall
6. Compare with your ML model performance
"""

print(summary)

with open("VEP_SUBMISSION_GUIDE.txt", 'w') as f:
    f.write(summary)

print("\n[OK] Guide saved to: VEP_SUBMISSION_GUIDE.txt")

print("\n" + "="*70)
print("VEP INPUT FILE CREATION COMPLETE")
print("="*70)
print("\nReady to submit to Ensembl VEP:")
print(f"  1. test_variants_vep.vcf ({len(df)} variants)")
print(f"  2. test_variants_vep_default.txt ({len(df_vep_default)} variants)")
print("\nWeb interface: https://www.ensembl.org/Tools/VEP")
