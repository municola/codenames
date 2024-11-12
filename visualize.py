import pandas as pd
import numpy as np

# Load the processed data
df = pd.read_csv("data/processed/top100german.csv")

# Convert the string representation of embeddings back to numpy arrays
df["embedding"] = df["embedding"].apply(eval)

print(df)
