"""
Configuration management for loot.

Handles loading and saving config.json, and managing the .loot/ directory.
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class LootConfig:
    """Configuration for loot semantic search."""

    # Default configuration values
    DEFAULT_MODEL_NAME = "onnx-community/embeddinggemma-300m-ONNX"
    DEFAULT_EMBEDDING_DIM = 768
    DEFAULT_INCLUDE_EXTENSIONS = [
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".scala",
        ".cs",
        ".php",
        ".rb",
        ".sh",
        ".sql",
        ".html",
        ".css",
        ".json",
        ".yaml",
        ".yml",
    ]
    DEFAULT_CHUNK_SIZE_LINES = 64
    DEFAULT_CHUNK_OVERLAP_LINES = 16

    def __init__(
        self,
        model_name: str | None = None,
        embedding_dim: int | None = None,
        include_extensions: list[str] | None = None,
        chunk_size_lines: int | None = None,
        chunk_overlap_lines: int | None = None,
    ):
        """Initialize configuration with provided or default values."""
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.embedding_dim = embedding_dim or self.DEFAULT_EMBEDDING_DIM
        self.include_extensions = include_extensions or self.DEFAULT_INCLUDE_EXTENSIONS
        self.chunk_size_lines = chunk_size_lines or self.DEFAULT_CHUNK_SIZE_LINES
        self.chunk_overlap_lines = chunk_overlap_lines or self.DEFAULT_CHUNK_OVERLAP_LINES

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "include_extensions": self.include_extensions,
            "chunk_size_lines": self.chunk_size_lines,
            "chunk_overlap_lines": self.chunk_overlap_lines,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LootConfig":
        """Create configuration from dictionary."""
        return cls(
            model_name=data.get("model_name"),
            embedding_dim=data.get("embedding_dim"),
            include_extensions=data.get("include_extensions"),
            chunk_size_lines=data.get("chunk_size_lines"),
            chunk_overlap_lines=data.get("chunk_overlap_lines"),
        )


def get_loot_dir(root_dir: Path | None = None) -> Path:
    """
    Get the .loot directory path.

    Args:
        root_dir: Root directory to use. Defaults to current directory.

    Returns:
        Path to .loot directory
    """
    if root_dir is None:
        root_dir = Path.cwd()
    return root_dir / ".loot"


def ensure_loot_dir(root_dir: Path | None = None) -> Path:
    """
    Ensure .loot directory exists.

    Args:
        root_dir: Root directory to use. Defaults to current directory.

    Returns:
        Path to .loot directory

    Raises:
        OSError: If directory cannot be created
    """
    loot_dir = get_loot_dir(root_dir)
    loot_dir.mkdir(exist_ok=True)
    return loot_dir


def get_config_path(root_dir: Path | None = None) -> Path:
    """Get path to config.json file."""
    return get_loot_dir(root_dir) / "config.json"


def get_db_path(root_dir: Path | None = None) -> Path:
    """Get path to index.db file."""
    return get_loot_dir(root_dir) / "index.db"


def get_vector_index_path(root_dir: Path | None = None) -> Path:
    """Get path to vectors.faiss file."""
    return get_loot_dir(root_dir) / "vectors.faiss"


def load_config(root_dir: Path | None = None) -> LootConfig:
    """
    Load configuration from config.json.

    Args:
        root_dir: Root directory to use. Defaults to current directory.

    Returns:
        LootConfig object

    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid
    """
    config_path = get_config_path(root_dir)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Run 'loot init' first."
        )

    with open(config_path, "r") as f:
        data = json.load(f)

    return LootConfig.from_dict(data)


def save_config(config: LootConfig, root_dir: Path | None = None) -> None:
    """
    Save configuration to config.json.

    Args:
        config: Configuration to save
        root_dir: Root directory to use. Defaults to current directory.
    """
    config_path = get_config_path(root_dir)
    ensure_loot_dir(root_dir)

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    console.print(f"✓ Configuration saved to {config_path}", style="green")


def is_initialized(root_dir: Path | None = None) -> bool:
    """
    Check if loot has been initialized in the directory.

    Args:
        root_dir: Root directory to check. Defaults to current directory.

    Returns:
        True if .loot directory and config.json exist
    """
    loot_dir = get_loot_dir(root_dir)
    config_path = get_config_path(root_dir)

    return loot_dir.exists() and config_path.exists()
