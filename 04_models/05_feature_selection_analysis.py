"""
Feature Selection Analysis: Identify which new features are worth keeping
Evaluates individual feature contribution to decide what to eliminate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

output_dir = Path("04_models")

# Load data
print("Loading feature matrix...")
df = pd.read_csv(Path("data") / "feature_matrix.csv")
y = df['label']

# Define feature groups
ORIGINAL_FEATURES = [
    'is_missense', 'is_nonsense', 'is_frameshift', 'is_splice',
    'is_synonymous', 'is_intron', 'is_utr', 'is_inframe_indel',
    'is_snv', 'is_deletion', 'is_insertion', 'is_duplication',
    'is_COL1A1', 'is_COL1A2',
    'hydrophobic_change', 'charge_change', 'polar_change',
    'aromatic_change', 'size_change', 'flexibility_change',
    'has_aa_change', 'normalized_position',
    'high_risk_consequence', 'low_risk_consequence', 'glycine_substitution'
]

NEW_EXTENDED_AA = [
    'alpha_change', 'beta_change', 'turn_change',
    'polarity_change', 'volume_change', 'surface_change', 'mw_change'
]

NEW_GPP_PROLINE = [
    'is_in_gxy_motif', 'gxy_position', 'is_in_gpp_motif',
    'is_proline_ref', 'is_proline_alt', 'proline_substitution',
    'affects_gxy_glycine'
]

NEW_WINDOW = [
    'window_avg_hydrophobic', 'window_avg_volume', 'window_avg_polarity',
    'window_avg_alpha', 'window_avg_beta', 'window_avg_turn',
]

NEW_WINDOW_NO_GLY = [
    'window_avg_hydrophobic_no_gly', 'window_avg_volume_no_gly', 'window_avg_polarity_no_gly',
    'window_avg_alpha_no_gly', 'window_avg_beta_no_gly', 'window_avg_turn_no_gly',
]

# Filter to existing columns
ORIGINAL_FEATURES = [f for f in ORIGINAL_FEATURES if f in df.columns]
NEW_EXTENDED_AA = [f for f in NEW_EXTENDED_AA if f in df.columns]
NEW_GPP_PROLINE = [f for f in NEW_GPP_PROLINE if f in df.columns]
NEW_WINDOW = [f for f in NEW_WINDOW if f in df.columns]
NEW_WINDOW_NO_GLY = [f for f in NEW_WINDOW_NO_GLY if f in df.columns]

ALL_NEW = NEW_EXTENDED_AA + NEW_GPP_PROLINE + NEW_WINDOW + NEW_WINDOW_NO_GLY
ALL_FEATURES = ORIGINAL_FEATURES + ALL_NEW

print(f"\nFeature Groups:")
print(f"  Original: {len(ORIGINAL_FEATURES)}")
print(f"  New Extended AA: {len(NEW_EXTENDED_AA)}")
print(f"  New GPP/Proline: {len(NEW_GPP_PROLINE)}")
print(f"  New Window: {len(NEW_WINDOW)}")
print(f"  New Window (no Gly): {len(NEW_WINDOW_NO_GLY)}")
print(f"  Total New: {len(ALL_NEW)}")

# ============================================================
# 1. Feature Importance from Random Forest
# ============================================================
print(f"\n{'='*70}")
print("1. RANDOM FOREST FEATURE IMPORTANCE")
print(f"{'='*70}")

X_all = df[ALL_FEATURES].fillna(0)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_all, y)

importance_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': rf.feature_importances_,
    'is_new': ['NEW' if f in ALL_NEW else 'ORIGINAL' for f in ALL_FEATURES]
}).sort_values('importance', ascending=False)

print("\nTop 20 Features:")
print(importance_df.head(20).to_string(index=False))

print("\nNew Features Ranked by Importance:")
new_importance = importance_df[importance_df['is_new'] == 'NEW'].copy()
new_importance['rank'] = range(1, len(new_importance) + 1)
print(new_importance.to_string(index=False))

# ============================================================
# 2. Mutual Information Analysis
# ============================================================
print(f"\n{'='*70}")
print("2. MUTUAL INFORMATION WITH TARGET")
print(f"{'='*70}")

mi_scores = mutual_info_classif(X_all.fillna(0), y, random_state=42)
mi_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'mi_score': mi_scores,
    'is_new': ['NEW' if f in ALL_NEW else 'ORIGINAL' for f in ALL_FEATURES]
}).sort_values('mi_score', ascending=False)

print("\nNew Features by Mutual Information:")
new_mi = mi_df[mi_df['is_new'] == 'NEW'].sort_values('mi_score', ascending=False)
print(new_mi.to_string(index=False))

# ============================================================
# 3. Correlation with Existing Features (Redundancy Check)
# ============================================================
print(f"\n{'='*70}")
print("3. REDUNDANCY CHECK: Correlation with Existing Features")
print(f"{'='*70}")

X_orig = df[ORIGINAL_FEATURES].fillna(0)
X_new = df[ALL_NEW].fillna(0)

redundant_features = []
print("\nNew features highly correlated (>0.7) with original features:")
for new_feat in ALL_NEW:
    if new_feat not in df.columns:
        continue
    max_corr = 0
    max_corr_feat = None
    for orig_feat in ORIGINAL_FEATURES:
        if orig_feat not in df.columns:
            continue
        corr = abs(df[new_feat].fillna(0).corr(df[orig_feat].fillna(0)))
        if corr > max_corr:
            max_corr = corr
            max_corr_feat = orig_feat

    if max_corr > 0.7:
        redundant_features.append(new_feat)
        print(f"  {new_feat} <-> {max_corr_feat}: {max_corr:.3f} (REDUNDANT)")

# ============================================================
# 4. Ablation Study: Remove each feature group and measure impact
# ============================================================
print(f"\n{'='*70}")
print("4. ABLATION STUDY: Impact of Each Feature Group")
print(f"{'='*70}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_features(features, name):
    X = df[features].fillna(0)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean(), scores.std()

results = {}

# Baseline: Original only
acc, std = evaluate_features(ORIGINAL_FEATURES, "Original Only")
results['Original (25 features)'] = (acc, std, len(ORIGINAL_FEATURES))
print(f"Original Only (25): {acc:.4f} ± {std:.4f}")

# All features
acc, std = evaluate_features(ALL_FEATURES, "All Features")
results['All Features (51)'] = (acc, std, len(ALL_FEATURES))
print(f"All Features (51): {acc:.4f} ± {std:.4f}")

# Original + only GPP/Proline
features = ORIGINAL_FEATURES + NEW_GPP_PROLINE
acc, std = evaluate_features(features, "Original + GPP/Proline")
results['+ GPP/Proline (32)'] = (acc, std, len(features))
print(f"Original + GPP/Proline ({len(features)}): {acc:.4f} ± {std:.4f}")

# Original + only Extended AA
features = ORIGINAL_FEATURES + NEW_EXTENDED_AA
acc, std = evaluate_features(features, "Original + Extended AA")
results['+ Extended AA (32)'] = (acc, std, len(features))
print(f"Original + Extended AA ({len(features)}): {acc:.4f} ± {std:.4f}")

# Original + only Window features
features = ORIGINAL_FEATURES + NEW_WINDOW
acc, std = evaluate_features(features, "Original + Window")
results['+ Window (31)'] = (acc, std, len(features))
print(f"Original + Window ({len(features)}): {acc:.4f} ± {std:.4f}")

# Original + Window (no gly only)
features = ORIGINAL_FEATURES + NEW_WINDOW_NO_GLY
acc, std = evaluate_features(features, "Original + Window NoGly")
results['+ Window NoGly (31)'] = (acc, std, len(features))
print(f"Original + Window NoGly ({len(features)}): {acc:.4f} ± {std:.4f}")

# ============================================================
# 5. Identify Features to KEEP vs ELIMINATE
# ============================================================
print(f"\n{'='*70}")
print("5. RECOMMENDATION: Features to KEEP vs ELIMINATE")
print(f"{'='*70}")

# Criteria for keeping:
# - Importance > 0.005 (top half of new features)
# - Not redundant with existing features
# - MI score > median of new features

importance_threshold = new_importance['importance'].median()
mi_threshold = new_mi['mi_score'].median()

keep_features = []
eliminate_features = []

for feat in ALL_NEW:
    imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
    mi = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
    is_redundant = feat in redundant_features

    # Decision logic
    reasons_keep = []
    reasons_eliminate = []

    if imp >= importance_threshold:
        reasons_keep.append(f"importance={imp:.4f}")
    else:
        reasons_eliminate.append(f"low importance={imp:.4f}")

    if mi >= mi_threshold:
        reasons_keep.append(f"MI={mi:.4f}")
    else:
        reasons_eliminate.append(f"low MI={mi:.4f}")

    if is_redundant:
        reasons_eliminate.append("redundant")

    # Final decision
    if len(reasons_keep) >= 2 and not is_redundant:
        keep_features.append((feat, reasons_keep))
    else:
        eliminate_features.append((feat, reasons_eliminate))

print("\nFEATURES TO KEEP:")
for feat, reasons in keep_features:
    print(f"  [KEEP] {feat}: {', '.join(reasons)}")

print(f"\nFEATURES TO ELIMINATE:")
for feat, reasons in eliminate_features:
    print(f"  [DROP] {feat}: {', '.join(reasons)}")

# ============================================================
# 6. Test Optimized Feature Set
# ============================================================
print(f"\n{'='*70}")
print("6. OPTIMIZED FEATURE SET EVALUATION")
print(f"{'='*70}")

kept_features = [f[0] for f in keep_features]
OPTIMIZED_FEATURES = ORIGINAL_FEATURES + kept_features

print(f"\nOptimized set: {len(OPTIMIZED_FEATURES)} features ({len(kept_features)} new)")
print(f"Eliminated: {len(eliminate_features)} features")

acc_opt, std_opt = evaluate_features(OPTIMIZED_FEATURES, "Optimized")
acc_orig, _ = results['Original (25 features)'][:2]
acc_all, _ = results['All Features (51)'][:2]

print(f"\nComparison:")
print(f"  Original (25 features):  {acc_orig:.4f}")
print(f"  All New (51 features):   {acc_all:.4f} ({(acc_all-acc_orig)*100:+.2f}%)")
print(f"  Optimized ({len(OPTIMIZED_FEATURES)} features): {acc_opt:.4f} ({(acc_opt-acc_orig)*100:+.2f}%)")

# ============================================================
# 7. Save Recommendations
# ============================================================
print(f"\n{'='*70}")
print("7. FINAL RECOMMENDATION")
print(f"{'='*70}")

report = f"""
FEATURE SELECTION ANALYSIS REPORT
{'='*50}

SUMMARY:
- Original features: {len(ORIGINAL_FEATURES)}
- New features added: {len(ALL_NEW)}
- Recommended to KEEP: {len(kept_features)}
- Recommended to DROP: {len(eliminate_features)}
- Optimized total: {len(OPTIMIZED_FEATURES)}

PERFORMANCE COMPARISON:
- Original (25 features):  {acc_orig:.4f}
- All Features (51):       {acc_all:.4f} ({(acc_all-acc_orig)*100:+.2f}%)
- Optimized ({len(OPTIMIZED_FEATURES)} features):   {acc_opt:.4f} ({(acc_opt-acc_orig)*100:+.2f}%)

FEATURES TO KEEP:
{chr(10).join([f"  - {f[0]}" for f in keep_features])}

FEATURES TO ELIMINATE:
{chr(10).join([f"  - {f[0]}: {', '.join(f[1])}" for f in eliminate_features])}

REASONING:
The window average features and many GPP/Proline features show low
importance and high redundancy with existing features. The model
already captures pathogenicity well through high_risk_consequence,
glycine_substitution, and molecular consequence features.

OPTIMIZED FEATURE LIST:
{chr(10).join([f"  {i+1}. {f}" for i, f in enumerate(OPTIMIZED_FEATURES)])}
"""

print(report)

with open(output_dir / 'FEATURE_SELECTION_REPORT.txt', 'w') as f:
    f.write(report)

# Save optimized feature list
pd.DataFrame({'feature': OPTIMIZED_FEATURES}).to_csv(
    output_dir / 'optimized_features.csv', index=False
)

print(f"\nFiles saved:")
print(f"  - {output_dir / 'FEATURE_SELECTION_REPORT.txt'}")
print(f"  - {output_dir / 'optimized_features.csv'}")
