import pandas as pd

df = pd.read_csv("COL1A2_All.txt", sep="\t", dtype=str, low_memory=False)

print("Total rows:", len(df))
print("\nClinical significance counts:")
print(df["Germline classification"].value_counts(dropna=False))

print("\nMolecular consequence counts:")
print(df["Molecular consequence"].value_counts(dropna=False))

print("\nNumber of OI-related rows:")
print(df["Condition(s)"].str.contains("osteogenesis", case=False, na=False).sum())