import pandas as pd

# Load features
df = pd.read_csv(r'C:\ESPI_TEMP\features\features_QCpass_W01.csv')
print('Features loaded:', df.shape)
print('Columns:', df.columns.tolist())

# Create labels based on frequency
labels = []
for _, row in df.iterrows():
    freq = row['freq_hz']
    if freq <= 45:
        label = 'low'
    elif freq <= 55:
        label = 'mid'
    else:
        label = 'high'
    labels.append({'id': row['id'], 'label': label})

# Save labels
label_df = pd.DataFrame(labels)
label_df.to_csv(r'C:\ESPI_TEMP\features\labels_W01.csv', index=False)
print('Labels created:', len(labels))
print('Label distribution:')
print(label_df['label'].value_counts())

