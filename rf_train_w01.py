#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from joblib import dump
from pathlib import Path

def main():
    # Load features and labels
    X = pd.read_csv(r"C:\ESPI_TEMP\features\features_QCpass_W01.csv")
    y = pd.read_csv(r"C:\ESPI_TEMP\features\labels_W01.csv")
    
    # Merge features with labels
    df = X.merge(y[["id", "label"]], on="id", how="inner")
    
    # Select numeric features only
    features = df.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    labels = df["label"].astype(str)
    
    print(f"Training data shape: {features.shape}")
    print(f"Label distribution:")
    print(labels.value_counts())
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    yt = []
    yp = []
    
    for tr, te in cv.split(features, labels):
        clf = RandomForestClassifier(
            n_estimators=1000, 
            max_features="sqrt", 
            min_samples_leaf=2,
            class_weight="balanced", 
            n_jobs=-1, 
            random_state=42
        )
        clf.fit(features.iloc[tr], labels.iloc[tr])
        p = clf.predict(features.iloc[te])
        f1s.append(f1_score(labels.iloc[te], p, average="macro"))
        yt += labels.iloc[te].tolist()
        yp += p.tolist()
    
    print(f"\nMacro-F1 (W01 only): {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print("\nClassification Report:")
    print(classification_report(yt, yp))
    
    # Train final model
    clf = RandomForestClassifier(
        n_estimators=1000, 
        max_features="sqrt", 
        min_samples_leaf=2,
        class_weight="balanced", 
        n_jobs=-1, 
        random_state=42
    )
    clf.fit(features, labels)
    
    # Save model
    model_dir = Path(r"C:\ESPI_TEMP\features\rf_model_w01")
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(clf, model_dir / "rf.joblib")
    print(f"\n[OK] Model saved -> {model_dir / 'rf.joblib'}")

if __name__ == "__main__":
    main()

