#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

def main():
    # Load labeled features
    df = pd.read_csv("C:\\ESPI_TEMP\\features\\labels_fixed_bins.csv")
    
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution:")
    print(df['class_name'].value_counts())
    
    # Feature columns (exclude metadata)
    feature_cols = ['valid_px', 'zero_frac', 'chg_h', 'chg_v', 'chg_d1', 'chg_d2', 
                   'grad_mean', 'grad_std', 'lap_mad', 'phase_std']
    
    # Check if we have enough features
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Available features: {available_features}")
    
    if len(available_features) < 3:
        print("Not enough features for training. Need more data.")
        return
    
    # Prepare features and labels
    X = df[available_features].values
    y = df['class_id'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Check if we have enough samples per class
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Classes: {unique_classes}")
    print(f"Counts: {counts}")
    
    if len(unique_classes) < 2:
        print("Need at least 2 classes for classification")
        return
    
    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== DEMO RF RESULTS ===")
    print(f"Accuracy: {accuracy:.3f}")
    
    print(f"\nClassification Report:")
    # Get actual class names from the data
    class_id_to_name = dict(zip(df['class_id'], df['class_name']))
    actual_class_names = [class_id_to_name[class_id] for class_id in sorted(unique_classes)]
    print(classification_report(y_test, y_pred, target_names=actual_class_names))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = rf.feature_importances_
    print(f"\nTop 5 Feature Importances:")
    for i, (feature, importance) in enumerate(zip(available_features, feature_importance)):
        if i < 5:
            print(f"  {feature}: {importance:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Save results
    results = {
        "n_samples": len(df),
        "n_features": len(available_features),
        "classes": unique_classes.tolist(),
        "class_counts": counts.tolist(),
        "accuracy": float(accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feature_importance": dict(zip(available_features, feature_importance.tolist()))
    }
    
    with open("C:\\ESPI_TEMP\\demo_rf_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: C:\\ESPI_TEMP\\demo_rf_results.json")

if __name__ == "__main__":
    main()
