import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the file without the comment='#' argument
# This ensures the header line is read.
df = pd.read_csv('1eXf8ZzkWueyMwc6.vep.txt', sep='\t')

# 2. Clean the column names (removes the '#' from '#Uploaded_variation')
df.columns = [col.lstrip('#') for col in df.columns]

# DEBUG: Print columns to verify 'Consequence' and 'REVEL' (or equivalent) exist
print("Detected Columns:", df.columns.tolist())

# 3. Filter for Missense variants
if 'Consequence' in df.columns:
    df_missense = df[df['Consequence'].str.contains('missense_variant', na=False)].copy()
else:
    print("Error: 'Consequence' column not found. Check file formatting.")
    exit()

# 4. Handle Pathogenicity Labels (ClinVar)
# If CLIN_SIG is empty or '-', you may need to map your known labels from your CSV
pathogenic_terms = ['pathogenic', 'likely_pathogenic']
df_missense['actual'] = df_missense['CLIN_SIG'].apply(
    lambda x: 1 if any(term in str(x).lower() for term in pathogenic_terms) else 0
)

# 5. REVEL Score Logic
# NOTE: Check your 'Detected Columns' output above. 
# If the column isn't named 'REVEL', change the name below (e.g., to 'REVEL_score')
revel_col = 'REVEL' 

if revel_col in df_missense.columns:
    # Convert to numeric (handles cases where scores might be string or '-')
    df_missense[revel_col] = pd.to_numeric(df_missense[revel_col], errors='coerce')
    df_missense = df_missense.dropna(subset=[revel_col]) # Remove variants without a REVEL score
    
    threshold = 0.5
    df_missense['predicted'] = (df_missense[revel_col] >= threshold).astype(int)

    # 6. Generate Matrix
    cm = confusion_matrix(df_missense['actual'], df_missense['predicted'])

    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu_r', 
                xticklabels=['Pred. Benign', 'Pred. Pathogenic'],
                yticklabels=['Actual Benign', 'Actual Pathogenic'])
    plt.ylabel('Actual (ClinVar)')
    plt.xlabel('Predicted (REVEL)')
    plt.title(f'REVEL Performance on COL1 Variants\n(Threshold = {threshold})')
    plt.show()

    print("\nClassification Report for REVEL:")
    print(classification_report(df_missense['actual'], df_missense['predicted']))
else:
    print(f"Error: Column '{revel_col}' not found. Please check the 'Detected Columns' list above.")