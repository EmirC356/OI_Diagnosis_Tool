import pandas as pd


def clean(df):
    # Keep relevant OI records already filtered
    df = df.copy()
    
    # Map labels
    label_map = {
        "Pathogenic": 1,
        "Likely pathogenic": 1,
        "Pathogenic/Likely pathogenic": 1,
        "Benign": 0,
        "Likely benign": 0,
        "Benign/Likely benign": 0
    }
    
    df = df[df["Germline classification"].isin(label_map.keys())]
    df["label"] = df["Germline classification"].map(label_map)

    # Keep meaningful columns
    keep = [
        "Name", "Gene(s)", "Protein change", "Condition(s)",
        "VariationID", "Variant type", "Molecular consequence", "label"
    ]
    df = df[keep].drop_duplicates()
    
    return df

col1a1 = pd.read_csv("COL1A1_All.txt", sep="\t", dtype=str)
col1a2 = pd.read_csv("COL1A2_All.txt", sep="\t", dtype=str)

# Keep only OI-related entries
col1a1 = col1a1[col1a1["Condition(s)"].str.contains("osteogenesis", case=False, na=False)]
col1a2 = col1a2[col1a2["Condition(s)"].str.contains("osteogenesis", case=False, na=False)]

clean1 = clean(col1a1)
clean2 = clean(col1a2)

combined = pd.concat([clean1, clean2], ignore_index=True)
combined.to_csv("cleaned_COL1_variants.csv", index=False)

print("Final dataset shape:", combined.shape)
print(combined["label"].value_counts())