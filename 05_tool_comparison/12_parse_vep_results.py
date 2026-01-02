"""
Parse VEP Results and Compare with ML Model
Extract SIFT and PolyPhen predictions from VEP output
Compare with ground truth and ML model performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("PARSING VEP RESULTS AND COMPARING WITH ML MODEL")
print("="*70)

# ===================================================================
# PART 1: Load VEP Results
# ===================================================================

print("\n1. Loading VEP Results")
print("-" * 70)

# Read VEP output (tab-separated, skip comment lines)
df_vep = pd.read_csv("1eXf8ZzkWueyMwc6.txt", sep='\t', comment='##')

print(f"Total VEP records: {len(df_vep)}")
print(f"Columns: {list(df_vep.columns[:10])}...")  # Show first 10 columns

# Filter for missense variants only (most relevant)
df_vep_missense = df_vep[df_vep['Consequence'] == 'missense_variant'].copy()
print(f"Missense variant records: {len(df_vep_missense)}")

# Keep only MANE_Select transcript (canonical/primary transcript)
df_vep_canonical = df_vep_missense[df_vep_missense['MANE'].str.contains('MANE_Select', na=False)].copy()
print(f"Canonical transcript (MANE_Select) records: {len(df_vep_canonical)}")

# ===================================================================
# PART 2: Extract SIFT and PolyPhen Scores
# ===================================================================

print("\n2. Extracting SIFT and PolyPhen Predictions")
print("-" * 70)

def parse_sift(sift_str):
    """
    Parse SIFT column: 'deleterious(0)' or 'tolerated(0.5)'
    Returns: (prediction, score)
    """
    if pd.isna(sift_str) or sift_str == '-':
        return 'unknown', np.nan

    # Extract prediction and score
    if 'deleterious' in sift_str.lower():
        pred = 'deleterious'
    elif 'tolerated' in sift_str.lower():
        pred = 'tolerated'
    else:
        pred = 'unknown'

    # Extract score
    import re
    match = re.search(r'\(([\d.]+)\)', sift_str)
    if match:
        score = float(match.group(1))
    else:
        score = np.nan

    return pred, score

def parse_polyphen(polyphen_str):
    """
    Parse PolyPhen column: 'probably_damaging(0.99)' or 'benign(0.1)'
    Returns: (prediction, score)
    """
    if pd.isna(polyphen_str) or polyphen_str == '-':
        return 'unknown', np.nan

    # Extract prediction and score
    polyphen_lower = polyphen_str.lower()
    if 'probably' in polyphen_lower and 'damaging' in polyphen_lower:
        pred = 'probably_damaging'
    elif 'possibly' in polyphen_lower and 'damaging' in polyphen_lower:
        pred = 'possibly_damaging'
    elif 'benign' in polyphen_lower:
        pred = 'benign'
    else:
        pred = 'unknown'

    # Extract score
    import re
    match = re.search(r'\(([\d.]+)\)', polyphen_str)
    if match:
        score = float(match.group(1))
    else:
        score = np.nan

    return pred, score

# Parse SIFT and PolyPhen
print("Parsing SIFT predictions...")
df_vep_canonical[['SIFT_pred', 'SIFT_score']] = df_vep_canonical['SIFT'].apply(
    lambda x: pd.Series(parse_sift(x))
)

print("Parsing PolyPhen predictions...")
df_vep_canonical[['PolyPhen_pred', 'PolyPhen_score']] = df_vep_canonical['PolyPhen'].apply(
    lambda x: pd.Series(parse_polyphen(x))
)

# Show summary
print(f"\nSIFT predictions:")
print(df_vep_canonical['SIFT_pred'].value_counts())
print(f"\nSIFT scores available: {df_vep_canonical['SIFT_score'].notna().sum()}/{len(df_vep_canonical)}")

print(f"\nPolyPhen predictions:")
print(df_vep_canonical['PolyPhen_pred'].value_counts())
print(f"\nPolyPhen scores available: {df_vep_canonical['PolyPhen_score'].notna().sum()}/{len(df_vep_canonical)}")

# ===================================================================
# PART 3: Match with Ground Truth
# ===================================================================

print("\n3. Matching with Ground Truth Labels")
print("-" * 70)

# Load ground truth
df_truth = pd.read_csv("test_variants_for_revel.tsv", sep='\t')
print(f"Ground truth variants: {len(df_truth)}")

# Extract variant ID from VEP uploaded_variation column
# Format: var_1166697 or rsID
df_vep_canonical['VariationID'] = df_vep_canonical['#Uploaded_variation'].str.extract(r'var_(\d+)|rs(\d+)')[0].fillna(
    df_vep_canonical['#Uploaded_variation'].str.extract(r'var_(\d+)|rs(\d+)')[1]
)

# Try matching by rsID if VariationID match fails
# Also try matching by location
df_vep_canonical['Location_match'] = df_vep_canonical['Location']

# Merge with ground truth
print("\nMatching by VariationID...")
df_merged = pd.merge(
    df_vep_canonical[['#Uploaded_variation', 'VariationID', 'Location', 'Amino_acids',
                       'SIFT_pred', 'SIFT_score', 'PolyPhen_pred', 'PolyPhen_score']],
    df_truth[['VariationID', 'Name', 'Protein change', 'label']],
    left_on='#Uploaded_variation',
    right_on='VariationID',
    how='left',
    suffixes=('_vep', '_truth')
)

# For those that didn't match, try matching by extracting variation ID differently
if df_merged['label'].isna().sum() > 0:
    print(f"Could not match {df_merged['label'].isna().sum()} variants by uploaded_variation")
    print("Trying alternative matching...")

    # Create a mapping from uploaded variation to VariationID
    # The uploaded variation might be like 'var_1166697'
    for idx, row in df_merged[df_merged['label'].isna()].iterrows():
        uploaded_var = row['#Uploaded_variation']

        # Try extracting just the ID
        if 'var_' in str(uploaded_var):
            var_id = uploaded_var.replace('var_', '')
            match = df_truth[df_truth['VariationID'].astype(str) == var_id]
            if len(match) > 0:
                df_merged.loc[idx, 'label'] = match.iloc[0]['label']
                df_merged.loc[idx, 'Name'] = match.iloc[0]['Name']
                df_merged.loc[idx, 'Protein change'] = match.iloc[0]['Protein change']

matched = df_merged['label'].notna().sum()
print(f"\nSuccessfully matched: {matched}/{len(df_merged)} variants")

# Filter to only matched variants
df_final = df_merged[df_merged['label'].notna()].copy()
print(f"Final dataset for analysis: {len(df_final)} variants")

# ===================================================================
# PART 4: Convert Predictions to Binary
# ===================================================================

print("\n4. Converting Predictions to Binary (Pathogenic=1, Benign=0)")
print("-" * 70)

# SIFT: deleterious = pathogenic (1), tolerated = benign (0)
df_final['SIFT_binary'] = df_final['SIFT_pred'].map({
    'deleterious': 1,
    'tolerated': 0,
    'unknown': np.nan
})

# PolyPhen: probably_damaging or possibly_damaging = pathogenic (1), benign = benign (0)
df_final['PolyPhen_binary'] = df_final['PolyPhen_pred'].map({
    'probably_damaging': 1,
    'possibly_damaging': 1,
    'benign': 0,
    'unknown': np.nan
})

# Show distribution
print(f"\nSIFT binary predictions:")
print(df_final['SIFT_binary'].value_counts(dropna=False))

print(f"\nPolyPhen binary predictions:")
print(df_final['PolyPhen_binary'].value_counts(dropna=False))

print(f"\nTrue labels:")
print(df_final['label'].value_counts())

# ===================================================================
# PART 5: Calculate Performance Metrics
# ===================================================================

print("\n5. Calculating Performance Metrics")
print("=" * 70)

# Filter to variants with predictions
df_sift_eval = df_final[df_final['SIFT_binary'].notna()].copy()
df_polyphen_eval = df_final[df_final['PolyPhen_binary'].notna()].copy()

results = {}

# SIFT Performance
if len(df_sift_eval) > 0:
    sift_acc = accuracy_score(df_sift_eval['label'], df_sift_eval['SIFT_binary'])
    sift_prec = precision_score(df_sift_eval['label'], df_sift_eval['SIFT_binary'], zero_division=0)
    sift_rec = recall_score(df_sift_eval['label'], df_sift_eval['SIFT_binary'], zero_division=0)
    sift_f1 = f1_score(df_sift_eval['label'], df_sift_eval['SIFT_binary'], zero_division=0)

    # Calculate specificity
    tn = ((df_sift_eval['label'] == 0) & (df_sift_eval['SIFT_binary'] == 0)).sum()
    fp = ((df_sift_eval['label'] == 0) & (df_sift_eval['SIFT_binary'] == 1)).sum()
    sift_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    results['SIFT'] = {
        'Accuracy': sift_acc,
        'Precision': sift_prec,
        'Recall': sift_rec,
        'F1-Score': sift_f1,
        'Specificity': sift_spec,
        'N': len(df_sift_eval)
    }

# PolyPhen Performance
if len(df_polyphen_eval) > 0:
    poly_acc = accuracy_score(df_polyphen_eval['label'], df_polyphen_eval['PolyPhen_binary'])
    poly_prec = precision_score(df_polyphen_eval['label'], df_polyphen_eval['PolyPhen_binary'], zero_division=0)
    poly_rec = recall_score(df_polyphen_eval['label'], df_polyphen_eval['PolyPhen_binary'], zero_division=0)
    poly_f1 = f1_score(df_polyphen_eval['label'], df_polyphen_eval['PolyPhen_binary'], zero_division=0)

    # Calculate specificity
    tn = ((df_polyphen_eval['label'] == 0) & (df_polyphen_eval['PolyPhen_binary'] == 0)).sum()
    fp = ((df_polyphen_eval['label'] == 0) & (df_polyphen_eval['PolyPhen_binary'] == 1)).sum()
    poly_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    results['PolyPhen-2'] = {
        'Accuracy': poly_acc,
        'Precision': poly_prec,
        'Recall': poly_rec,
        'F1-Score': poly_f1,
        'Specificity': poly_spec,
        'N': len(df_polyphen_eval)
    }

# Add ML model results (from previous analysis)
results['Our Model (Gradient Boosting)'] = {
    'Accuracy': 0.9700,
    'Precision': 0.9801,
    'Recall': 0.9643,
    'F1-Score': 0.9721,
    'Specificity': 0.9979,
    'N': 3105  # Full dataset
}

results['Our Model (Random Forest)'] = {
    'Accuracy': 0.9726,
    'Precision': 0.9836,
    'Recall': 0.9655,
    'F1-Score': 0.9744,
    'Specificity': 0.9993,
    'N': 3105
}

# Create results dataframe
df_results = pd.DataFrame(results).T
df_results = df_results.sort_values('Accuracy', ascending=False)

print("\nPERFORMANCE COMPARISON")
print("=" * 70)
print(df_results.to_string())

# Save results
df_results.to_csv("vep_vs_ml_comparison.csv")
print("\n[OK] Saved results to: vep_vs_ml_comparison.csv")

# ===================================================================
# PART 6: Create Visualization
# ===================================================================

print("\n6. Creating Comparison Visualization")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
tools = list(df_results.index)
accuracies = df_results['Accuracy'] * 100
colors = ['#2ecc71' if 'Our Model' in tool else '#3498db' for tool in tools]
bars = ax1.barh(tools, accuracies, color=colors)
ax1.set_xlabel('Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, accuracies)):
    ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')

# Plot 2: Precision vs Recall
ax2 = axes[0, 1]
for tool in tools:
    prec = df_results.loc[tool, 'Precision'] * 100
    rec = df_results.loc[tool, 'Recall'] * 100
    color = '#2ecc71' if 'Our Model' in tool else '#3498db'
    marker = 'o' if 'Our Model' in tool else 's'
    size = 200 if 'Our Model' in tool else 100
    ax2.scatter(rec, prec, s=size, c=color, marker=marker, alpha=0.7, edgecolors='black', linewidth=2)
    ax2.annotate(tool, (rec, prec), xytext=(5, 5), textcoords='offset points', fontsize=9)
ax2.set_xlabel('Recall (%)', fontsize=12)
ax2.set_ylabel('Precision (%)', fontsize=12)
ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xlim(80, 100)
ax2.set_ylim(80, 100)

# Plot 3: All Metrics Radar Chart
ax3 = axes[1, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

ax3 = plt.subplot(2, 2, 3, projection='polar')
for tool in tools[:3]:  # Show top 3
    values = df_results.loc[tool, metrics].values.tolist()
    values += values[:1]
    color = '#2ecc71' if 'Our Model' in tool else '#3498db'
    ax3.plot(angles, values, 'o-', linewidth=2, label=tool, color=color)
    ax3.fill(angles, values, alpha=0.15, color=color)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metrics, size=10)
ax3.set_ylim(0, 1)
ax3.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax3.grid(True)

# Plot 4: Specificity Comparison
ax4 = axes[1, 1]
specificities = df_results['Specificity'] * 100
bars = ax4.barh(tools, specificities, color=colors)
ax4.set_xlabel('Specificity (%)', fontsize=12)
ax4.set_title('Specificity Comparison', fontsize=14, fontweight='bold')
ax4.set_xlim(0, 100)
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, specificities)):
    ax4.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('vep_vs_ml_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved visualization to: vep_vs_ml_comparison.png")

# ===================================================================
# PART 7: Summary
# ===================================================================

print("\n7. Summary")
print("=" * 70)

summary = f"""
VEP RESULTS VS ML MODEL COMPARISON

Test Set:
  - Total variants analyzed: {len(df_final)}
  - Pathogenic: {(df_final['label']==1).sum()}
  - Benign: {(df_final['label']==0).sum()}

SIFT Performance:
  - Variants evaluated: {results['SIFT']['N']}
  - Accuracy: {results['SIFT']['Accuracy']:.2%}
  - Precision: {results['SIFT']['Precision']:.2%}
  - Recall: {results['SIFT']['Recall']:.2%}
  - Specificity: {results['SIFT']['Specificity']:.2%}

PolyPhen-2 Performance:
  - Variants evaluated: {results['PolyPhen-2']['N']}
  - Accuracy: {results['PolyPhen-2']['Accuracy']:.2%}
  - Precision: {results['PolyPhen-2']['Precision']:.2%}
  - Recall: {results['PolyPhen-2']['Recall']:.2%}
  - Specificity: {results['PolyPhen-2']['Specificity']:.2%}

Our Best Model (Random Forest):
  - Accuracy: {results['Our Model (Random Forest)']['Accuracy']:.2%}
  - Precision: {results['Our Model (Random Forest)']['Precision']:.2%}
  - Recall: {results['Our Model (Random Forest)']['Recall']:.2%}
  - Specificity: {results['Our Model (Random Forest)']['Specificity']:.2%}

IMPROVEMENT OVER SIFT:
  - Accuracy: +{(results['Our Model (Random Forest)']['Accuracy'] - results['SIFT']['Accuracy'])*100:.1f} percentage points
  - Precision: +{(results['Our Model (Random Forest)']['Precision'] - results['SIFT']['Precision'])*100:.1f} percentage points
  - Specificity: +{(results['Our Model (Random Forest)']['Specificity'] - results['SIFT']['Specificity'])*100:.1f} percentage points

IMPROVEMENT OVER POLYPHEN-2:
  - Accuracy: +{(results['Our Model (Random Forest)']['Accuracy'] - results['PolyPhen-2']['Accuracy'])*100:.1f} percentage points
  - Precision: +{(results['Our Model (Random Forest)']['Precision'] - results['PolyPhen-2']['Precision'])*100:.1f} percentage points
  - Specificity: +{(results['Our Model (Random Forest)']['Specificity'] - results['PolyPhen-2']['Specificity'])*100:.1f} percentage points

FILES GENERATED:
  1. vep_vs_ml_comparison.csv - Detailed metrics
  2. vep_vs_ml_comparison.png - Visualization
"""

print(summary)

with open("VEP_COMPARISON_SUMMARY.txt", 'w') as f:
    f.write(summary)

print("\n[OK] Summary saved to: VEP_COMPARISON_SUMMARY.txt")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
