#!/usr/bin/env python3
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from joblib import dump
from pathlib import Path

# Load data
X = pd.read_csv(r"C:\ESPI_TEMP\features\features_numeric_only.csv")
y = pd.read_csv(r"C:\ESPI_TEMP\features\labels_modes_grouped.csv")

# Merge features with labels
df = X.merge(y, on=["set", "basename"], how="inner")

# Prepare features and labels
features = df.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
labels = df["label"].astype(str)
groups = df["set"]

print(f"Training data shape: {features.shape}")
print(f"Label distribution:")
print(labels.value_counts())
print(f"Groups: {groups.value_counts()}")

# StratifiedGroupKFold with group=set (use 3-fold due to small classes)
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
f1s = []
yt = []
yp = []

for tr, te in cv.split(features, labels, groups):
    clf = RandomForestClassifier(
        n_estimators=1200,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(features.iloc[tr], labels.iloc[tr])
    p = clf.predict(features.iloc[te])
    yt += labels.iloc[te].tolist()
    yp += p.tolist()
    f1s.append(f1_score(labels.iloc[te], p, average="macro"))

print(f"\nMacro-F1 (3-fold): {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print("\nClassification Report:")
print(classification_report(yt, yp))

# Confusion matrix
cm = confusion_matrix(yt, yp, normalize="true")
cm_df = pd.DataFrame(cm, index=sorted(set(labels)), columns=sorted(set(labels)))

# Save results
out_dir = Path(r"C:\ESPI_TEMP\features\rf_model_5to7")
out_dir.mkdir(parents=True, exist_ok=True)

cm_df.to_csv(out_dir / "confusion.csv")
print(f"\nConfusion matrix saved to: {out_dir / 'confusion.csv'}")

# Train final model on all data
final = RandomForestClassifier(
    n_estimators=1200,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
final.fit(features, labels)

# Save final model
dump(final, out_dir / "modes_rf_pipeline.joblib")
print(f"Final model saved to: {out_dir / 'modes_rf_pipeline.joblib'}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features.columns,
    'importance': final.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)
print(f"Feature importance saved to: {out_dir / 'feature_importance.csv'}")
print("\nTop 10 most important features:")
print(feature_importance.head(10))
