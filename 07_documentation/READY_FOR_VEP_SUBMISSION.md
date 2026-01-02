# Ready for VEP Submission - Test Files

**Date**: December 24, 2024
**Purpose**: Test your ML models against SIFT, PolyPhen-2, CADD, REVEL on the same test set

---

## Files Created ✓

All files are properly formatted for Ensembl VEP submission:

### 1. **test_variants_vep.vcf** (RECOMMENDED)
- **Format**: Standard VCF v4.2
- **Variants**: 176 missense variants
- **Reference**: GRCh38
- **Use**: Most compatible format for VEP web interface

### 2. **test_variants_vep_default.txt**
- **Format**: Default VEP format (chr start end allele strand id)
- **Variants**: 175 variants
- **Use**: Alternative format if VCF doesn't work

### 3. **test_variants_vep_hgvs.txt**
- **Format**: HGVS notation (e.g., NM_000088.4(COL1A1):c.4387T>C)
- **Variants**: 176 variants
- **Use**: If you prefer transcript-based notation

### 4. **test_variants_vep_rsid.txt**
- **Format**: dbSNP rsIDs (e.g., rs577626107)
- **Variants**: 150 variants with rsIDs
- **Use**: Fastest lookup method (if variants have rsIDs)

### 5. **test_variants_for_revel.tsv** (Ground Truth)
- **Contains**: VariationID, chr, pos, ref, alt, Name, Protein change, **label**
- **Use**: This has the TRUE LABELS (1=pathogenic, 0=benign)
- **Purpose**: Compare VEP predictions against this ground truth

---

## How to Submit to Ensembl VEP

### Option A: Web Interface (Easiest)

1. **Go to**: https://www.ensembl.org/Tools/VEP

2. **Upload file**: Choose `test_variants_vep.vcf`

3. **Configure settings**:
   - Species: **Human**
   - Assembly: **GRCh38** (very important!)
   - Input format: **VCF**

4. **Select additional annotations**:
   - [x] **SIFT predictions**
   - [x] **PolyPhen predictions**
   - [x] **dbNSFP plugin** (for CADD and REVEL scores)
     - In "Plugins" section, enable dbNSFP
     - Select: REVEL_score, CADD_phred

5. **Run** and wait for results (may take 5-15 minutes for 176 variants)

6. **Download** the results file (usually `vep_results.txt` or similar)

### Option B: Command Line VEP (If Installed)

```bash
vep --input_file test_variants_vep.vcf \
    --output_file vep_results.txt \
    --format vcf \
    --species homo_sapiens \
    --assembly GRCh38 \
    --sift b \
    --polyphen b \
    --plugin dbNSFP,REVEL_score,CADD_phred \
    --cache
```

---

## Test Set Details

**Source**: Fold 1 from 5-fold cross-validation (same folds used to train/test your ML models)

**Total test set**:
- 621 variants total
- 336 pathogenic (54.1%)
- 285 benign (45.9%)

**Missense test set** (for VEP):
- 176 missense variants
- 160 pathogenic (90.9%)
- 16 benign (9.1%)
- 175 with complete genomic coordinates (99.4%)

**Why missense-focused?**
- SIFT, PolyPhen-2, and REVEL are designed specifically for missense variants
- Your ML model handles ALL variant types (frameshift, nonsense, splice, etc.)
- This comparison focuses on the overlapping scope

---

## After Getting VEP Results

### Step 1: Parse VEP Output

VEP output will be a tab-separated file with columns like:
- `#Uploaded_variation`: Your variant ID (var_1166697)
- `Location`: Genomic position
- `SIFT`: Prediction (deleterious/tolerated) and score
- `PolyPhen`: Prediction (probably_damaging/possibly_damaging/benign) and score
- `REVEL`: Score (0-1, higher = more pathogenic)
- `CADD_phred`: CADD phred score (>20 = pathogenic)

### Step 2: Match with Ground Truth

```python
import pandas as pd

# Load VEP results
df_vep = pd.read_csv("vep_results.txt", sep='\t', comment='##')

# Load ground truth
df_truth = pd.read_csv("test_variants_for_revel.tsv", sep='\t')

# Merge on variant ID
df_comparison = df_vep.merge(df_truth,
                               left_on='#Uploaded_variation',
                               right_on='VariationID')
```

### Step 3: Calculate Tool Performance

```python
# Convert REVEL scores to predictions (threshold = 0.5)
df_comparison['REVEL_pred'] = (df_comparison['REVEL'] > 0.5).astype(int)

# Convert SIFT predictions
df_comparison['SIFT_pred'] = (df_comparison['SIFT'].str.contains('deleterious', case=False, na=False)).astype(int)

# Convert PolyPhen predictions
df_comparison['PolyPhen_pred'] = (df_comparison['PolyPhen'].str.contains('damaging', case=False, na=False)).astype(int)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

for tool in ['REVEL_pred', 'SIFT_pred', 'PolyPhen_pred']:
    acc = accuracy_score(df_comparison['label'], df_comparison[tool])
    prec = precision_score(df_comparison['label'], df_comparison[tool])
    rec = recall_score(df_comparison['label'], df_comparison[tool])

    print(f"{tool}: Accuracy={acc:.2%}, Precision={prec:.2%}, Recall={rec:.2%}")
```

### Step 4: Compare with Your ML Model

Your ML model's expected performance on this test set:
- **Accuracy**: ~97%
- **Precision**: ~98%
- **Recall**: ~96%

Generic tools (expected on missense variants):
- **SIFT**: ~75-80% accuracy
- **PolyPhen-2**: ~80-85% accuracy
- **REVEL**: ~85-90% accuracy
- **CADD**: ~85-90% accuracy

---

## Expected Outcomes

### Your Advantage

Your ML model should outperform generic tools because:

1. **Disease-specific training**
   - Trained exclusively on OI-related COL1A1/COL1A2 variants
   - Generic tools trained on variants across all genes

2. **Collagen-specific features**
   - **Glycine substitution feature** (6.1% importance)
   - Captures critical Gly-X-Y repeat requirement
   - Generic tools can't explicitly model this

3. **Comprehensive feature set**
   - 25 engineered features
   - Multiple information levels (DNA, RNA, protein, biochemical)

### Fair Comparison Note

- **Your model**: Tested on ALL variant types (frameshift, nonsense, splice, missense)
- **Generic tools**: Work best on missense variants only
- **This test**: Focuses on 176 missense variants (overlap)
- **Expected**: Your model still better, but gap may be smaller than overall comparison

---

## Quick Start Guide

1. **Upload**: `test_variants_vep.vcf` to https://www.ensembl.org/Tools/VEP

2. **Settings**:
   - Assembly: GRCh38
   - Enable: SIFT, PolyPhen, dbNSFP plugin

3. **Download results** when complete

4. **Parse and compare** with `test_variants_for_revel.tsv`

5. **Calculate metrics** and create comparison table

6. **Update your report** with direct head-to-head comparison

---

## Files Checklist

- [x] test_variants_vep.vcf (VCF format for VEP)
- [x] test_variants_vep_default.txt (Default VEP format)
- [x] test_variants_vep_hgvs.txt (HGVS format)
- [x] test_variants_vep_rsid.txt (rsID format)
- [x] test_variants_for_revel.tsv (Ground truth labels)
- [x] VEP_SUBMISSION_GUIDE.txt (Detailed instructions)
- [x] complete_test_set.tsv (Full test set, all variants)
- [x] missense_test_set.tsv (Missense variants with all data)

---

## Alternative: Use Pre-Existing REVEL Scores

If VEP is slow or doesn't work, you can look up pre-computed REVEL scores:

1. **REVEL website**: https://sites.google.com/site/revelgenomics/downloads
   - Download pre-computed scores
   - Look up chr:pos:ref:alt from your test file

2. **dbNSFP database**:
   - Contains pre-computed SIFT, PolyPhen, CADD, REVEL
   - See `TOOL_PREDICTIONS_STATUS.md` for download instructions

---

## Questions?

- **VEP documentation**: https://www.ensembl.org/info/docs/tools/vep/index.html
- **VEP formats**: https://www.ensembl.org/info/docs/tools/vep/vep_formats.html
- **dbNSFP info**: See `06_get_existing_tool_predictions.md`

---

**Ready to submit!** 🚀

Upload `test_variants_vep.vcf` to Ensembl VEP and get your direct comparison results.
