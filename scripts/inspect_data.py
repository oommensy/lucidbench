import pandas as pd

PATH = "data/raw/dreamviews-posts.tsv"
df = pd.read_csv(PATH, sep="\t", low_memory=False)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
for col in df.columns:
    nunique = df[col].nunique(dropna=True)
    if nunique <= 20:
        print(f"\n{col}:\n", df[col].value_counts(dropna=False).head(20))
