"""
Embedder for generating semantic embeddings from code using ONNX.

Uses ONNX Runtime and transformers for efficient on-device inference.
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from rich.console import Console
from transformers import AutoTokenizer

console = Console()


class Embedder:
    """Wrapper for ONNX-based embedding model with MRL support."""

    # Supported MRL dimensions for EmbeddingGemma
    SUPPORTED_DIMS = [768, 512, 256, 128]

    def __init__(
        self, model_name: str, device: str | None = None, embedding_dim: int = 768
    ):
        """
        Initialize embedder with ONNX model.

        Args:
            model_name: HuggingFace model name (e.g., "onnx-community/embeddinggemma-300m-ONNX")
            device: Device to use ("cuda" or "cpu", None for auto-detect)
            embedding_dim: Dimension to use for embeddings (768, 512, 256, or 128)
                          Lower dimensions are faster but may have lower quality.
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Validate embedding dimension
        if embedding_dim not in self.SUPPORTED_DIMS:
            raise ValueError(
                f"embedding_dim must be one of {self.SUPPORTED_DIMS}, got {embedding_dim}"
            )

        console.print(
            f"Loading ONNX model [cyan]{model_name}[/cyan] with dimension [cyan]{embedding_dim}[/cyan]..."
        )

        try:
            # Download model files from HuggingFace Hub
            model_path = hf_hub_download(model_name, subfolder="onnx", filename="model.onnx")
            hf_hub_download(model_name, subfolder="onnx", filename="model.onnx_data")

            providers = ["CPUExecutionProvider"]

            # Create ONNX Runtime session
            self.session = ort.InferenceSession(model_path, providers=providers)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Get actual provider being used
            actual_provider = self.session.get_providers()[0]
            console.print(
                f"✓ Model loaded successfully (provider: {actual_provider}, dimension: {embedding_dim})",
                style="green",
            )

        except Exception as e:
            console.print(f"✗ Failed to load model: {e}", style="red")
            console.print(
                "\n[yellow]Make sure you've set a valid ONNX embedding model in config.json.[/yellow]"
            )
            console.print(
                "[yellow]For example: 'onnx-community/embeddinggemma-300m-ONNX'[/yellow]"
            )
            raise

    def _embed_batch(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """
        Internal method to embed a batch of texts.

        Args:
            texts: List of text strings to embed
            prefix: Prefix to add to each text (e.g., query or document prefix)

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) with L2-normalized embeddings
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        # Add prefix to all texts
        if prefix:
            texts = [prefix + text for text in texts]

        # Tokenize with truncation to handle long chunks
        # EmbeddingGemma has a max sequence length of 2048 tokens
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="np"
        )

        # Run inference
        # ONNX model returns (token_embeddings, sentence_embeddings)
        _, sentence_embeddings = self.session.run(None, inputs.data)

        # Truncate to desired dimension (MRL support)
        sentence_embeddings = sentence_embeddings[:, : self.embedding_dim]

        # L2 normalize for cosine similarity
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        sentence_embeddings = sentence_embeddings / norms

        return sentence_embeddings

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of code chunks.

        Args:
            texts: List of text strings to embed

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) with L2-normalized embeddings
        """
        # Use document prefix for code chunks
        document_prefix = "title: none | text: "
        return self._embed_batch(texts, prefix=document_prefix)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single search query.

        Args:
            query: Query string

        Returns:
            1D numpy array of shape (embedding_dim,) with L2-normalized embedding
        """
        # Use query prefix for search queries
        query_prefix = "task: search result | query: "
        embeddings = self._embed_batch([query], prefix=query_prefix)
        return embeddings[0]

    @property
    def dim(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dim
