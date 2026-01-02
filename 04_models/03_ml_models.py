"""
Machine Learning Models for Variant Pathogenicity Prediction
Implements and evaluates multiple ML classifiers:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Uses 5-fold cross-validation and comprehensive evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load feature matrix
print("Loading feature matrix...")
df = pd.read_csv(Path("data") / "feature_matrix.csv")

# Define features
FEATURE_COLS = [
    'is_missense', 'is_nonsense', 'is_frameshift', 'is_splice',
    'is_synonymous', 'is_intron', 'is_utr', 'is_inframe_indel',
    'is_snv', 'is_deletion', 'is_insertion', 'is_duplication',
    'is_COL1A1', 'is_COL1A2',
    'hydrophobic_change', 'charge_change', 'polar_change',
    'aromatic_change', 'size_change', 'flexibility_change',
    'has_aa_change', 'normalized_position',
    'high_risk_consequence', 'low_risk_consequence', 'glycine_substitution'
]

# Prepare data
X = df[FEATURE_COLS].fillna(0)
y = df['label']

print(f"Dataset: {len(X)} samples, {len(FEATURE_COLS)} features")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Standardize features (important for SVM and Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,
                                                    learning_rate=0.1,
                                                    max_depth=5,
                                                    random_state=42)
}

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Perform cross-validation
print(f"\n{'='*70}")
print("CROSS-VALIDATION RESULTS (5-Fold)")
print(f"{'='*70}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Use scaled data for LogReg and SVM, original for tree-based
    X_train = X_scaled_df if name in ['Logistic Regression', 'SVM'] else X

    cv_results = cross_validate(model, X_train, y, cv=cv, scoring=scoring,
                                return_train_score=True, n_jobs=-1)

    results[name] = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'precision': cv_results['test_precision'].mean(),
        'recall': cv_results['test_recall'].mean(),
        'f1': cv_results['test_f1'].mean(),
        'roc_auc': cv_results['test_roc_auc'].mean(),
        'train_accuracy': cv_results['train_accuracy'].mean()
    }

    print(f"  Accuracy:  {results[name]['accuracy']:.4f} ± {results[name]['accuracy_std']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  F1-Score:  {results[name]['f1']:.4f}")
    print(f"  ROC-AUC:   {results[name]['roc_auc']:.4f}")
    print(f"  Train Acc: {results[name]['train_accuracy']:.4f}")

# Create results DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print(f"\n{'='*70}")
print("MODEL COMPARISON")
print(f"{'='*70}")
print(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])

# Save results
results_df.to_csv('model_comparison.csv')
print("\nResults saved to 'model_comparison.csv'")

# Train final models on full dataset for feature importance and predictions
print(f"\n{'='*70}")
print("TRAINING FINAL MODELS ON FULL DATASET")
print(f"{'='*70}")

final_models = {}
for name, model in models.items():
    print(f"Training final {name}...")
    X_train = X_scaled_df if name in ['Logistic Regression', 'SVM'] else X
    model.fit(X_train, y)
    final_models[name] = model

    # Get predictions
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]

    # Calculate MCC
    mcc = matthews_corrcoef(y, y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"  MCC: {mcc:.4f}")
    print(f"  Sensitivity (TPR): {sensitivity:.4f}")
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

# Feature importance for Random Forest
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE (Random Forest)")
print(f"{'='*70}")

rf_model = final_models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15))
feature_importance.to_csv('feature_importance.csv', index=False)

# Visualizations
print(f"\nCreating visualizations...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Model comparison
ax1 = plt.subplot(2, 3, 1)
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(models))
width = 0.15

for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in models.keys()]
    ax1.bar(x + i*width, values, width, label=metric.upper())

ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison', fontweight='bold')
ax1.set_xticks(x + width*2)
ax1.set_xticklabels(models.keys(), rotation=15, ha='right')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.05])

# Plot 2: Feature importance
ax2 = plt.subplot(2, 3, 2)
top_features = feature_importance.head(15)
ax2.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'].values, fontsize=9)
ax2.set_xlabel('Importance')
ax2.set_title('Top 15 Feature Importances (RF)', fontweight='bold')
ax2.invert_yaxis()

# Plot 3-6: ROC curves for each model
for idx, (name, model) in enumerate(final_models.items(), start=3):
    ax = plt.subplot(2, 3, idx)
    X_train = X_scaled_df if name in ['Logistic Regression', 'SVM'] else X
    y_proba = model.predict_proba(X_train)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {name}', fontweight='bold', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("Visualizations saved to 'model_evaluation.png'")

# Create confusion matrices
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(final_models.items()):
    X_train = X_scaled_df if name in ['Logistic Regression', 'SVM'] else X
    y_pred = model.predict(X_train)
    cm = confusion_matrix(y, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Benign', 'Pathogenic'],
                yticklabels=['Benign', 'Pathogenic'])
    axes[idx].set_title(f'Confusion Matrix - {name}', fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Confusion matrices saved to 'confusion_matrices.png'")

print(f"\n{'='*70}")
print("MODEL TRAINING COMPLETE")
print(f"{'='*70}")
print("\nFiles generated:")
print("  - model_comparison.csv")
print("  - feature_importance.csv")
print("  - model_evaluation.png")
print("  - confusion_matrices.png")
