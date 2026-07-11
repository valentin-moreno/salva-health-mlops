import pandas as pd

# 1. Leer el archivo parquet
df = pd.read_parquet('data/processed/X_train.parquet')

# 2. Ver el total de filas y columnas
print(f"Dimensiones: {df.shape[0]} filas y {df.shape[1]} columnas.\n")

# 3. Mostrar la lista de columnas
print("--- Columnas del Dataset ---")
print(df.columns.tolist())

# 4. Darle un vistazo a los primeros datos
display(df.head())
