from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json

datasetname = "top10000german"

with open(".secrets.json", "r") as f:
    secrets = json.load(f)
client = OpenAI(api_key=secrets["openai_api_key"])


def get_embeddings_batch(texts, model="text-embedding-3-small", batch_size=2048):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i : i + batch_size]
        batch = [text.replace("\n", " ") for text in batch]
        response = client.embeddings.create(input=batch, model=model)
        embeddings.extend([data.embedding for data in response.data])
    return embeddings


# Read in German words from top100german.txt, ensuring proper encoding
with open(f"data/raw/{datasetname}.txt", "r", encoding="utf-8") as f:
    german_words = [word.lower() for word in f.read().splitlines()]

# Create DataFrame
df = pd.DataFrame({"word": german_words})

# Get embeddings in batches
embeddings = get_embeddings_batch(df.word.tolist())
df["embedding"] = embeddings

# Save with UTF-8 encoding to preserve special characters
df.to_csv(f"data/processed/{datasetname}.csv", index=False, encoding="utf-8")
