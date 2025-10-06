import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json

# Load data
df = pd.read_csv(r"C:\ESPI_TEMP\features\labels_no_qc_gate.csv")

print(f"Loaded {len(df)} samples")
print("\nClass distribution:")
print(df["class_name"].value_counts())

# Prepare features - use all available feature columns
exclude_cols = ['id', 'dataset', 'source_file', 'class_id', 'class_name', 'material', 'freq_hz.1']
available_features = [c for c in df.columns if c not in exclude_cols]
print(f"\nAvailable features: {available_features}")

X = df[available_features].values
y = df['class_id'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")
print(f"Classes: {np.unique(y)}")
print(f"Counts: {[np.sum(y==c) for c in np.unique(y)]}")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RF
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, 
                            class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Get actual class names from the data
class_id_to_name = dict(zip(df['class_id'], df['class_name']))
unique_classes = sorted(df['class_id'].unique())
actual_class_names = [class_id_to_name[class_id] for class_id in unique_classes]

# Results
print("\n=== RF RESULTS (COMPLETE DATA) ===")
print(f"Accuracy: {rf.score(X_test, y_test):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=actual_class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = sorted(zip(available_features, rf.feature_importances_), 
                           key=lambda x: x[1], reverse=True)
print("\nTop 5 Feature Importances:")
for feat, imp in feature_importance[:5]:
    print(f"  {feat}: {imp:.3f}")

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Save results
results = {
    "total_samples": int(len(df)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "accuracy": float(rf.score(X_test, y_test)),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "class_distribution": df["class_name"].value_counts().to_dict(),
    "feature_importance": {feat: float(imp) for feat, imp in feature_importance},
    "confusion_matrix": cm.tolist()
}

with open(r"C:\ESPI_TEMP\rf_results_complete.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: C:\\ESPI_TEMP\\rf_results_complete.json")

