import pandas as pd

def main():
    df = pd.read_csv(r"C:\ESPI_TEMP\features\labels_fixed_bins_complete.csv")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df["class_name"].value_counts(dropna=False))

    print("\nDataset distribution:")
    print(df["dataset"].value_counts())

    print("\nFrequency distribution by class:")
    freq_stats_by_class = df.groupby("class_name")["freq_hz"].agg(["min", "median", "max", "count"]).round(1)
    for class_name, row in freq_stats_by_class.iterrows():
        print(f'{class_name}: {row["min"]:.0f}-{row["max"]:.0f} Hz (median: {row["median"]:.0f} Hz, n={int(row["count"])})')

if __name__ == "__main__":
    main()

