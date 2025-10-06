#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load deduplicated features
df = pd.read_csv(r"C:\ESPI_TEMP\features\features_dedup.csv")

# Select only numeric columns (exclude set, basename, id)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Keep set and basename for merging with labels
keep_cols = ['set', 'basename'] + numeric_cols
df_numeric = df[keep_cols].copy()

# Save numeric-only features
df_numeric.to_csv(r"C:\ESPI_TEMP\features\features_numeric_only.csv", index=False)

print(f"[OK] Numeric features: {df_numeric.shape}")
print(f"Columns: {df_numeric.columns.tolist()}")

