from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if self._model is None:
            print("Loading local SentenceTransformer model 'all-MiniLM-L6-v2' (384 dimensions) into memory...")
            try:
                self._model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            except Exception:
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully.")

    def get_embedding(self, text: str) -> list:
        """
        Generates a 384-dimensional vector embedding for a single text input.
        """
        if not text.strip():
            # Return zero vector if empty string is supplied
            return [0.0] * 384
        vector = self._model.encode(text)
        return vector.tolist()

    def get_embeddings(self, texts: list) -> list:
        """
        Generates vector embeddings for a list of text inputs.
        """
        if not texts:
            return []
        vectors = self._model.encode(texts)
        return vectors.tolist()

# Global singleton instance to import across application
embedding_service = EmbeddingService()
