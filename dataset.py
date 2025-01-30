import requests
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def normalize_embedding(embedding):
    """Normalize embeddings to unit length (L2 normalization)."""
    embedding = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def get_llm_embedding(text):
    """Fetch embeddings from an LLM API and ensure they are of fixed size."""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "cas/llama-3.2-3b-instruct:latest", "prompt": text},
            timeout=10
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", None)
        
        if embedding is None:
            raise ValueError("No embedding received.")
        
        
        return embedding  # Return the full embedding (likely 3072)
    except requests.RequestException as e:
        print(f"Error fetching embeddings: {e}")
        return [0] * 3072  # Return a fallback vector of correct size



class EntityMatchingDataset(Dataset):
    """Dataset for entity matching using LLM embeddings."""
    
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        embedding1 = normalize_embedding(get_llm_embedding(text1))
        embedding2 = normalize_embedding(get_llm_embedding(text2))
        embedding1 = np.array(embedding1, dtype=np.float32)
        embedding2 = np.array(embedding2, dtype=np.float32)
        
        if embedding1.shape[0] != 3072 or embedding2.shape[0] != 3072:
            raise ValueError(f"Embedding size mismatch: {embedding1.shape}, {embedding2.shape}")


        input_vector = np.concatenate((embedding1, embedding2))   # Concatenating embeddings
        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
