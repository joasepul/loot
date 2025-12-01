"""
Semantic search functionality for loot.

Enables searching code by meaning rather than exact text matches.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .config import get_db_path, get_vector_index_path, load_config
from .embedder import Embedder
from .storage import LootStorage
from .vector_index import VectorIndex

console = Console()


@dataclass
class SearchResult:
    """Result from a semantic search."""

    file_path: str
    start_line: int
    end_line: int
    score: float
    content: str
    language: str | None = None


def get_file_lines(file_path: Path, start_line: int, end_line: int) -> str:
    """
    Get specific lines from a file.

    Args:
        file_path: Path to the file
        start_line: Starting line (1-indexed)
        end_line: Ending line (1-indexed, inclusive)

    Returns:
        Content of the specified lines
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Convert to 0-indexed
        start_idx = start_line - 1
        end_idx = end_line

        return "".join(lines[start_idx:end_idx])
    except Exception as e:
        return f"[Error reading file: {e}]"


def semantic_search(
    query: str, top_k: int = 10, root_dir: Path | None = None
) -> list[SearchResult]:
    """
    Perform semantic search on the indexed code.

    Args:
        query: Search query string
        top_k: Number of results to return
        root_dir: Root directory (defaults to current directory)

    Returns:
        List of SearchResult objects
    """
    if root_dir is None:
        root_dir = Path.cwd()

    # Load configuration
    config = load_config(root_dir)

    # Initialize storage
    db_path = get_db_path(root_dir)
    storage = LootStorage(db_path)
    storage.connect()

    # Initialize embedder
    embedder = Embedder(config.model_name, embedding_dim=config.embedding_dim)

    # Load vector index
    vector_index_path = get_vector_index_path(root_dir)
    if not vector_index_path.exists():
        console.print("[red]✗ Vector index not found. Run 'loot index' first.[/red]")
        storage.close()
        return []

    vector_index = VectorIndex(embedder.dim, vector_index_path)
    vector_index.load()

    # Embed the query
    query_embedding = embedder.embed_query(query)

    # Search for similar vectors
    vector_ids, scores = vector_index.search(query_embedding, k=top_k)

    # Retrieve chunk metadata and file content
    results = []
    for vector_id, score in zip(vector_ids, scores):
        # Get chunk metadata
        chunk = storage.get_chunk_by_vector_id(vector_id)
        if chunk is None:
            continue

        # Get file metadata
        file_record = storage.get_file_by_id(chunk.file_id)
        if file_record is None:
            continue

        # Read file content for this chunk
        file_path = root_dir / file_record.path
        content = get_file_lines(file_path, chunk.start_line, chunk.end_line)

        # Create result
        result = SearchResult(
            file_path=file_record.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            score=score,
            content=content,
            language=file_record.language,
        )
        results.append(result)

    # Close storage
    storage.close()

    return results


def format_search_results(
    results: list[SearchResult], max_lines: int | None = None, as_json: bool = False
) -> str:
    """
    Format search results for display.

    Args:
        results: List of SearchResult objects
        max_lines: Maximum number of lines to show around each result (None = show all)
        as_json: Output as JSON instead of formatted text

    Returns:
        Formatted string
    """
    if as_json:
        # JSON output
        results_dict = [
            {
                "file_path": r.file_path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "score": round(r.score, 4),
                "language": r.language,
                "content": r.content,
            }
            for r in results
        ]
        return json.dumps(results_dict, indent=2)

    # Formatted text output (ripgrep-like)
    output_lines = []

    for result in results:
        # Header line
        header = f"{result.file_path}:{result.start_line}-{result.end_line}  score={result.score:.2f}"
        output_lines.append(header)

        # Content (potentially truncated)
        content = result.content
        if max_lines is not None:
            lines = content.split("\n")
            if len(lines) > max_lines:
                # Show first max_lines/2 and last max_lines/2
                half = max_lines // 2
                lines = lines[:half] + ["    ..."] + lines[-half:]
            content = "\n".join(lines)

        # Indent content
        for line in content.split("\n"):
            output_lines.append(f"    {line}")

        output_lines.append("")  # Empty line between results

    return "\n".join(output_lines)
