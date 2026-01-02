"""
Data Exploration for COL1A1/COL1A2 Osteogenesis Imperfecta Variants
This script explores the cleaned variant dataset to understand:
- Class distribution (pathogenic vs benign)
- Variant types distribution
- Molecular consequences
- Missing data patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the cleaned data
data_path = Path("data") / "cleaned_COL1_variants.csv"
df = pd.read_csv(data_path)

print("="*60)
print("OSTEOGENESIS IMPERFECTA VARIANT DATASET EXPLORATION")
print("="*60)

# Basic dataset info
print(f"\n1. Dataset Overview")
print(f"   Total variants: {len(df)}")
print(f"   Features: {df.shape[1]}")
print(f"   Columns: {', '.join(df.columns)}")

# Class distribution
print(f"\n2. Class Distribution (Label)")
class_counts = df['label'].value_counts()
print(f"   Benign (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
print(f"   Pathogenic (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

# Gene distribution
print(f"\n3. Gene Distribution")
gene_counts = df['Gene(s)'].value_counts()
for gene, count in gene_counts.items():
    print(f"   {gene}: {count} ({count/len(df)*100:.1f}%)")

# Variant type distribution
print(f"\n4. Variant Type Distribution")
variant_types = df['Variant type'].value_counts()
for vtype, count in variant_types.head(10).items():
    print(f"   {vtype}: {count}")

# Molecular consequence distribution
print(f"\n5. Molecular Consequence Distribution")
mol_conseq = df['Molecular consequence'].value_counts()
for conseq, count in mol_conseq.head(10).items():
    print(f"   {conseq}: {count}")

# Cross-tabulation: Variant type vs Label
print(f"\n6. Variant Type vs Pathogenicity")
variant_label_crosstab = pd.crosstab(df['Variant type'], df['label'], margins=True)
print(variant_label_crosstab)

# Cross-tabulation: Molecular consequence vs Label
print(f"\n7. Molecular Consequence vs Pathogenicity (Top 10)")
top_consequences = df['Molecular consequence'].value_counts().head(10).index
df_top_conseq = df[df['Molecular consequence'].isin(top_consequences)]
conseq_label_crosstab = pd.crosstab(df_top_conseq['Molecular consequence'],
                                     df_top_conseq['label'],
                                     margins=True)
print(conseq_label_crosstab)

# Missing data analysis
print(f"\n8. Missing Data Analysis")
missing = df.isnull().sum()
if missing.any():
    print(missing[missing > 0])
else:
    print("   No missing values found!")

# Protein change analysis
print(f"\n9. Protein Change Information")
has_protein_change = df['Protein change'].notna().sum()
print(f"   Variants with protein change info: {has_protein_change} ({has_protein_change/len(df)*100:.1f}%)")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Class distribution
axes[0, 0].bar(['Benign', 'Pathogenic'],
               [class_counts.get(0, 0), class_counts.get(1, 0)],
               color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate([class_counts.get(0, 0), class_counts.get(1, 0)]):
    axes[0, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Plot 2: Top 10 Variant Types
top_variants = df['Variant type'].value_counts().head(10)
axes[0, 1].barh(range(len(top_variants)), top_variants.values)
axes[0, 1].set_yticks(range(len(top_variants)))
axes[0, 1].set_yticklabels(top_variants.index)
axes[0, 1].set_title('Top 10 Variant Types', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Count')
axes[0, 1].invert_yaxis()

# Plot 3: Top 10 Molecular Consequences
top_conseq = df['Molecular consequence'].value_counts().head(10)
axes[1, 0].barh(range(len(top_conseq)), top_conseq.values, color='teal')
axes[1, 0].set_yticks(range(len(top_conseq)))
axes[1, 0].set_yticklabels(top_conseq.index)
axes[1, 0].set_title('Top 10 Molecular Consequences', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Count')
axes[1, 0].invert_yaxis()

# Plot 4: Pathogenicity by Variant Type (top 5)
top_5_variants = df['Variant type'].value_counts().head(5).index
df_top5 = df[df['Variant type'].isin(top_5_variants)]
variant_path_data = pd.crosstab(df_top5['Variant type'], df_top5['label'])
variant_path_data.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Pathogenicity by Variant Type (Top 5)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Variant Type')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(['Benign', 'Pathogenic'])
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data_exploration_plots.png', dpi=300, bbox_inches='tight')
print(f"\n10. Visualizations saved to 'data_exploration_plots.png'")

print("\n" + "="*60)
print("EXPLORATION COMPLETE")
print("="*60)
