import pandas as pd

df = pd.read_parquet("data/processed/dataset_final.parquet")

# tomar una pequeña muestra
df.sample(
    n=100,
    random_state=42,
).to_parquet(
    "tests/data/dataset_final.parquet",
    index=False,
)
