"""
Vector index wrapper using FAISS.

Provides approximate nearest neighbor search for code embeddings.
"""

from pathlib import Path

import faiss
import numpy as np
from rich.console import Console

console = Console()


class VectorIndex:
    """FAISS-based vector index for semantic search."""

    def __init__(self, dim: int, index_path: Path):
        """
        Initialize vector index.

        Args:
            dim: Dimension of the embedding vectors
            index_path: Path to save/load the index file
        """
        self.dim = dim
        self.index_path = index_path
        self.index: faiss.IndexFlatIP | None = None
        self.next_id = 0

    def initialize(self) -> None:
        """Initialize a new empty index."""
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.next_id = 0

    def load(self) -> None:
        """
        Load index from disk.

        Raises:
            FileNotFoundError: If index file doesn't exist
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found at {self.index_path}")

        self.index = faiss.read_index(str(self.index_path))
        # Set next_id to the current number of vectors
        self.next_id = self.index.ntotal

    def save(self) -> None:
        """Save index to disk."""
        if self.index is None:
            raise RuntimeError("Index not initialized")

        # Ensure parent directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            ids: List of integer IDs for the vectors
            vectors: 2D numpy array of shape (n, dim)

        Raises:
            ValueError: If vectors shape is incorrect or IDs don't match
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")

        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dim {self.dim}")

        if len(ids) != len(vectors):
            raise ValueError(f"Number of IDs ({len(ids)}) doesn't match number of vectors ({len(vectors)})")

        # Ensure vectors are float32 and contiguous
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms

        # Add to index
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[list[int], list[float]]:
        """
        Search for top-k most similar vectors.

        Args:
            query: Query vector of shape (dim,)
            k: Number of results to return

        Returns:
            Tuple of (ids, scores) where ids are the vector IDs and scores are similarity scores
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")

        # Ensure query is the right shape and type
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = np.ascontiguousarray(query, dtype=np.float32)

        # Normalize query vector
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Ensure we don't search for more results than exist
        k = min(k, self.index.ntotal)

        if k == 0:
            return [], []

        # Search
        scores, indices = self.index.search(query, k)

        # Convert to lists
        ids = indices[0].tolist()
        scores_list = scores[0].tolist()

        return ids, scores_list

    def delete(self, ids: list[int]) -> None:
        """
        Delete vectors by ID.

        Note: FAISS IndexFlatIP doesn't support deletion directly.
        This is a placeholder for future implementation with a different index type
        or rebuilding the index without deleted items.

        Args:
            ids: List of vector IDs to delete
        """
        # FAISS IndexFlatIP doesn't support deletion
        # For simplicity in v0, we'll just mark this as not supported
        # A production version could use IndexIDMap or rebuild the index
        raise NotImplementedError(
            "Deletion not supported in current implementation. "
            "Re-index the repository to remove deleted files."
        )

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
