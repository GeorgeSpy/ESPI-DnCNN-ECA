#!/usr/bin/env python3
import pandas as pd

def main():
    df = pd.read_csv(r'C:\ESPI_TEMP\features\labels_fixed_bins.csv')
    print('N =', len(df))
    print('Class distribution:')
    print(df['class_name'].value_counts(dropna=False))
    print()
    print('Frequency distribution by class:')
    for class_name in df['class_name'].unique():
        class_data = df[df['class_name'] == class_name]
        if len(class_data) > 0:
            freq_stats = class_data['freq_hz'].agg(['min','median','max'])
            print(f'{class_name}: {freq_stats["min"]:.0f}-{freq_stats["max"]:.0f} Hz (median: {freq_stats["median"]:.0f} Hz, n={len(class_data)})')

if __name__ == "__main__":
    main()
