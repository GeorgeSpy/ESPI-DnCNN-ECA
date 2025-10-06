#!/usr/bin/env python3
import pandas as pd

# Load the QC-passed features
df = pd.read_csv(r"C:\ESPI_TEMP\features\all_features_QCpass.csv")

# Remove duplicates based on basename (keep first occurrence)
df_dedup = df.drop_duplicates(subset=['basename'], keep='first')

# Save deduplicated data
df_dedup.to_csv(r"C:\ESPI_TEMP\features\features_dedup.csv", index=False)

print(f"[OK] Dedup: {len(df)} -> {len(df_dedup)} files")
print(f"Removed {len(df) - len(df_dedup)} duplicates")

