import pandas as pd
import pyarrow

# Especifica la ruta del archivo Parquet
file_path = '/home/oscar/data/abandoned_carts.parquet'

# Lee el archivo Parquet en un DataFrame de pandas
df = pd.read_parquet(file_path, engine='auto')

# Muestra las primeras filas del DataFrame
print(df.head())
