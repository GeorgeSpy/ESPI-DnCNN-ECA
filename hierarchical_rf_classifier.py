#!/usr/bin/env python3
"""
Hierarchical Random Forest Classifier for ESPI mode classification.
Uses LOW vs Rest approach to improve minority class performance.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from joblib import dump
from pathlib import Path

class HierarchicalRFClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.level1_classifier = None  # LOW vs Rest
        self.level2_classifier = None  # MID/HIGH/UHIGH classification
        self.class_names = None
        
    def prepare_hierarchical_labels(self, labels):
        """Convert multi-class labels to hierarchical structure."""
        # Level 1: LOW vs Rest
        level1_labels = ['LOW' if label == 'LOW' else 'REST' for label in labels]
        
        # Level 2: Only for REST samples (MID, HIGH, UHIGH)
        level2_labels = [label if label != 'LOW' else None for label in labels]
        
        return level1_labels, level2_labels
    
    def fit(self, X, y, groups):
        """Train hierarchical classifiers."""
        print("Training Hierarchical RF Classifier...")
        
        # Prepare hierarchical labels
        level1_y, level2_y = self.prepare_hierarchical_labels(y)
        
        # Level 1: LOW vs Rest
        print("Training Level 1: LOW vs REST")
        self.level1_classifier = RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state
        )
        self.level1_classifier.fit(X, level1_y)
        
        # Level 2: REST classification (only non-LOW samples)
        rest_mask = np.array(level2_y) != None
        if rest_mask.sum() > 0:
            print(f"Training Level 2: REST classification ({rest_mask.sum()} samples)")
            X_rest = X[rest_mask]
            y_rest = np.array(level2_y)[rest_mask]
            groups_rest = np.array(groups)[rest_mask]
            
            self.level2_classifier = RandomForestClassifier(
                n_estimators=800,
                max_features="sqrt", 
                min_samples_leaf=1,  # More aggressive for minority classes
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state
            )
            self.level2_classifier.fit(X_rest, y_rest)
        else:
            print("Warning: No REST samples found for Level 2 training")
            self.level2_classifier = None
        
        # Store class names
        self.class_names = sorted(set(y))
        
    def predict(self, X):
        """Make hierarchical predictions."""
        # Level 1: LOW vs REST
        level1_pred = self.level1_classifier.predict(X)
        
        # Level 2: REST classification
        final_pred = level1_pred.copy()
        
        if self.level2_classifier is not None:
            rest_mask = level1_pred == 'REST'
            if rest_mask.sum() > 0:
                X_rest = X[rest_mask]
                level2_pred = self.level2_classifier.predict(X_rest)
                final_pred[rest_mask] = level2_pred
        
        return final_pred
    
    def predict_proba(self, X):
        """Get prediction probabilities (simplified)."""
        # For simplicity, return level 1 probabilities
        return self.level1_classifier.predict_proba(X)

def evaluate_hierarchical_rf(X, y, groups, n_splits=3):
    """Evaluate hierarchical RF with cross-validation."""
    print("Evaluating Hierarchical RF Classifier...")
    
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Standard RF for comparison
    print("\nTraining Standard RF for comparison...")
    standard_f1s = []
    standard_yt = []
    standard_yp = []
    
    for tr, te in cv.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=1200,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict(X.iloc[te])
        standard_f1s.append(f1_score(y.iloc[te], p, average="macro"))
        standard_yt += y.iloc[te].tolist()
        standard_yp += p.tolist()
    
    # Hierarchical RF
    print("\nTraining Hierarchical RF...")
    hierarchical_f1s = []
    hierarchical_yt = []
    hierarchical_yp = []
    
    for tr, te in cv.split(X, y, groups):
        hrf = HierarchicalRFClassifier(random_state=42)
        hrf.fit(X.iloc[tr], y.iloc[tr], groups.iloc[tr])
        p = hrf.predict(X.iloc[te])
        hierarchical_f1s.append(f1_score(y.iloc[te], p, average="macro"))
        hierarchical_yt += y.iloc[te].tolist()
        hierarchical_yp += p.tolist()
    
    # Results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS COMPARISON")
    print("="*60)
    
    print(f"Standard RF:")
    print(f"  Macro-F1: {np.mean(standard_f1s):.3f} ± {np.std(standard_f1s):.3f}")
    print(f"  Per-class F1: {f1_score(standard_yt, standard_yp, average=None)}")
    
    print(f"\nHierarchical RF:")
    print(f"  Macro-F1: {np.mean(hierarchical_f1s):.3f} ± {np.std(hierarchical_f1s):.3f}")
    print(f"  Per-class F1: {f1_score(hierarchical_yt, hierarchical_yp, average=None)}")
    
    print(f"\nImprovement: {np.mean(hierarchical_f1s) - np.mean(standard_f1s):+.3f}")
    
    # Detailed classification reports
    print(f"\nStandard RF Classification Report:")
    print(classification_report(standard_yt, standard_yp))
    
    print(f"\nHierarchical RF Classification Report:")
    print(classification_report(hierarchical_yt, hierarchical_yp))
    
    return {
        'standard': {
            'macro_f1': np.mean(standard_f1s),
            'std': np.std(standard_f1s),
            'per_class_f1': f1_score(standard_yt, standard_yp, average=None),
            'predictions': standard_yp,
            'true': standard_yt
        },
        'hierarchical': {
            'macro_f1': np.mean(hierarchical_f1s),
            'std': np.std(hierarchical_f1s),
            'per_class_f1': f1_score(hierarchical_yt, hierarchical_yp, average=None),
            'predictions': hierarchical_yp,
            'true': hierarchical_yt
        }
    }

def main():
    """Main function to run hierarchical RF evaluation."""
    # Load data
    features_file = Path("C:/ESPI_TEMP/features/features_numeric_only.csv")
    labels_file = Path("C:/ESPI_TEMP/features/labels_modes_grouped.csv")
    
    if not features_file.exists() or not labels_file.exists():
        print("❌ Required files not found!")
        print(f"Features: {features_file}")
        print(f"Labels: {labels_file}")
        return
    
    # Load and merge data
    X = pd.read_csv(features_file)
    y_df = pd.read_csv(labels_file)
    
    df = X.merge(y_df, on=["set", "basename"], how="inner")
    
    # Prepare features and labels
    features = df.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    labels = df["label"].astype(str)
    groups = df["set"]
    
    print(f"Training data shape: {features.shape}")
    print(f"Label distribution:")
    print(labels.value_counts())
    print(f"Groups: {groups.value_counts()}")
    
    # Evaluate both approaches
    results = evaluate_hierarchical_rf(features, labels, groups)
    
    # Save results
    output_dir = Path("C:/ESPI_TEMP/features/hierarchical_rf_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save confusion matrices
    cm_standard = confusion_matrix(results['standard']['true'], results['standard']['predictions'], normalize="true")
    cm_hierarchical = confusion_matrix(results['hierarchical']['true'], results['hierarchical']['predictions'], normalize="true")
    
    pd.DataFrame(cm_standard, 
                index=sorted(set(labels)), 
                columns=sorted(set(labels))).to_csv(output_dir / "confusion_standard.csv")
    
    pd.DataFrame(cm_hierarchical,
                index=sorted(set(labels)),
                columns=sorted(set(labels))).to_csv(output_dir / "confusion_hierarchical.csv")
    
    print(f"\n📄 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

