"""
Model Training Script for OI-Pred
=================================
Trains and evaluates Random Forest classifier for OI variant pathogenicity.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURE_MATRIX, TRAINED_MODEL, FEATURE_LIST,
    FEATURE_COLS, RANDOM_STATE, CV_FOLDS, RF_PARAMS
)


def load_data():
    """Load and prepare training data."""
    print("Loading feature matrix...")
    df = pd.read_csv(FEATURE_MATRIX)

    X = df[FEATURE_COLS].fillna(0)
    y = df['label']

    print(f"Dataset: {len(X)} samples, {len(FEATURE_COLS)} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y


def train_model(X, y):
    """Train Random Forest model with cross-validation."""
    print(f"\n{'='*60}")
    print("TRAINING RANDOM FOREST MODEL")
    print(f"{'='*60}")

    # Initialize model
    model = RandomForestClassifier(**RF_PARAMS)

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    print(f"\nPerforming {CV_FOLDS}-fold cross-validation...")
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                return_train_score=True, n_jobs=-1)

    # Print results
    print(f"\nCross-Validation Results:")
    print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} +/- {cv_results['test_precision'].std():.4f}")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f} +/- {cv_results['test_recall'].std():.4f}")
    print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std():.4f}")
    print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}")

    # Train final model on full dataset
    print(f"\nTraining final model on full dataset...")
    model.fit(X, y)

    # Calculate additional metrics on training set
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    mcc = matthews_corrcoef(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"\nFinal Model Performance (Full Dataset):")
    print(f"  MCC:         {mcc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn}, FP={fp}")
    print(f"    FN={fn}, TP={tp}")

    return model, cv_results


def get_feature_importance(model, feature_names):
    """Extract and display feature importance."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top 15)")
    print(f"{'='*60}")
    print(importance_df.head(15).to_string(index=False))

    return importance_df


def save_model(model, feature_names):
    """Save trained model and feature list."""
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")

    # Save model
    joblib.dump(model, TRAINED_MODEL)
    print(f"Model saved to: {TRAINED_MODEL}")

    # Save feature list
    with open(FEATURE_LIST, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature list saved to: {FEATURE_LIST}")


def main():
    """Main training pipeline."""
    # Load data
    X, y = load_data()

    # Train model
    model, cv_results = train_model(X, y)

    # Get feature importance
    importance_df = get_feature_importance(model, FEATURE_COLS)

    # Save model
    save_model(model, FEATURE_COLS)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    return model, cv_results, importance_df


if __name__ == "__main__":
    model, cv_results, importance_df = main()
