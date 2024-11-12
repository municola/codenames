import json
import time
import numpy as np
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from openai import OpenAI


class WordEmbeddingManager:
    """
    A manager class for word embeddings using numpy storage.
    Includes common search operations and utility functions.
    """

    def __init__(self):
        with open(".secrets.json", "r") as f:
            secrets = json.load(f)
        self.client = OpenAI(api_key=secrets["openai_api_key"])

        self.embeddings: np.ndarray
        self.words: List[str]
        self.word_to_idx: Dict[str, int]

    def load(self, filepath: str, n_rows: int | None = None):
        """
        Load embeddings and metadata from CSV file.
        """
        import pandas as pd

        df = pd.read_csv(filepath, nrows=n_rows)
        self.embeddings = np.array([eval(emb) for emb in df["embedding"]])
        self.words = df["word"].tolist()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding for a specific word.
        If word is not in vocabulary, generates a new embedding.
        """
        if word not in self.word_to_idx:
            return self.generate_embedding(word)
        return self.embeddings[self.word_to_idx[word]]

    def generate_embedding(self, word: str) -> np.ndarray:
        """
        Generate an embedding for a word.
        """
        embedding = self.client.embeddings.create(input=[word], model="text-embedding-3-small").data[0].embedding
        return np.array(embedding)

    def find_most_similar(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        exclude_words: List[str] | None = None,
        allow_input_words: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar words to a query.
        Query can be either a word or an embedding vector.

        Args:
            query: Input word or embedding vector
            top_k: Number of similar words to return
            exclude_words: List of words to exclude from the results
            allow_input_words: If False, excludes words in exclude_words from results

        Returns:
            List of (word, similarity_score) tuples
        """
        if isinstance(query, str):
            query_vector = self.get_embedding(query)
        else:
            query_vector = query

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_vector)

        # Create a mask for excluded words if needed
        if exclude_words and not allow_input_words:
            exclude_indices = [self.word_to_idx[word] for word in exclude_words if word in self.word_to_idx]
            similarities[exclude_indices] = -float("inf")  # Set similarity to negative infinity for excluded words

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(self.words[idx], similarities[idx]) for idx in top_indices]

    def find_analogies(self, word1: str, word2: str, word3: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find word analogies (e.g., king - man + woman = queen).
        """
        # Get embeddings for the words
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        emb3 = self.get_embedding(word3)

        # Calculate target vector
        target = emb1 - emb2 + emb3

        return self.find_most_similar(target, top_k)

    def find_between(self, word1: str, word2: str, ratio: float = 0.5, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find words that lie between two given words in the embedding space.
        ratio determines how close to word1 vs word2 (0.5 = halfway between)
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)

        # Calculate intermediate point
        target = (1 - ratio) * emb1 + ratio * emb2

        return self.find_most_similar(target, top_k)

    def get_vocabulary_size(self) -> int:
        """
        Return the number of words in the vocabulary.
        """
        return len(self.words) if self.words is not None else 0


def main():
    manager = WordEmbeddingManager()
    start_time = time.time()
    manager.load("data/processed/top10000german.csv", n_rows=10000)
    print(f"Loading time: {time.time() - start_time:.3f} seconds")

    # Example 1
    # blue_words = ["blume", "wasser", "katze", "brot", "himmel", "haus", "musik"]
    # red_words = ["feuer", "auto", "telefon", "buch", "tisch", "kaffee", "apfel", "schuh"]
    # neutral_words = ["fenster", "tasche", "mond", "brief", "bank", "garten", "lampe", "uhr", "tee", "bild"]

    blue_words = ["katze", "hund"]

    # Find a word in the vocabulary that is most similar to the average of the blue words
    # For this we need to average the embeddings of the blue words and then find the word most similar to this average vector
    print("finding blue average")
    start_time = time.time()
    blue_embeddings = np.array([manager.get_embedding(word) for word in blue_words])
    blue_avg = np.mean(blue_embeddings, axis=0)
    print(f"Time taken: {time.time() - start_time:.3f} seconds")

    print("\nWithout input words:")
    most_similar = manager.find_most_similar(blue_avg, top_k=3, exclude_words=blue_words, allow_input_words=False)
    print(most_similar)


if __name__ == "__main__":
    main()
