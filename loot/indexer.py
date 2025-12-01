"""
Incremental indexing for loot.

Handles indexing files with mtime-based change detection.
"""

import os
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from .config import LootConfig, get_db_path, get_vector_index_path
from .embedder import Embedder
from .file_filters import chunk_file_by_lines, get_language_from_extension, iter_code_files
from .storage import LootStorage
from .vector_index import VectorIndex

console = Console()


class Indexer:
    """Incremental indexing for code files."""

    def __init__(
        self,
        config: LootConfig,
        storage: LootStorage,
        embedder: Embedder,
        vector_index: VectorIndex,
    ):
        """
        Initialize indexer.

        Args:
            config: Loot configuration
            storage: Storage layer
            embedder: Embedding model
            vector_index: Vector index
        """
        self.config = config
        self.storage = storage
        self.embedder = embedder
        self.vector_index = vector_index

    def index_repository(self, root_dir: Path, batch_size: int = 32) -> dict[str, int]:
        """
        Index all code files in the repository.

        Uses mtime-based change detection for incremental indexing.
        Batches chunking and embedding for better performance.

        Args:
            root_dir: Root directory to index
            batch_size: Number of chunks to embed per batch (default: 32)

        Returns:
            Dictionary with stats: new_files, updated_files, skipped_files, total_chunks
        """
        stats = {
            "new_files": 0,
            "updated_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
        }

        # Get all code files
        console.print(f"\n[cyan]Scanning for code files...[/cyan]")
        code_files = iter_code_files(root_dir, self.config.include_extensions)
        console.print(f"Found {len(code_files)} code files")

        # Build a set of current file paths for cleanup
        current_file_paths = {str(f.relative_to(root_dir)) for f in code_files}

        # Phase 1: Determine which files need indexing and chunk them
        console.print("[cyan]Analyzing files and chunking...[/cyan]")
        files_to_index = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking files", total=len(code_files))

            for file_path in code_files:
                try:
                    relative_path = str(file_path.relative_to(root_dir))
                    progress.update(task, description=f"[bold blue]Chunking[/bold blue] [cyan]{relative_path}[/cyan]")
                    
                    file_info = self._prepare_file_for_indexing(file_path, relative_path)
                    
                    if file_info:
                        files_to_index.append(file_info)
                        if file_info["action"] == "new":
                            stats["new_files"] += 1
                        elif file_info["action"] == "updated":
                            stats["updated_files"] += 1
                    else:
                        stats["skipped_files"] += 1

                except Exception as e:
                    console.print(f"[red]✗ Error preparing {file_path}: {e}[/red]")

                progress.advance(task)

        if not files_to_index:
            console.print("[yellow]No files to index[/yellow]")
            stats["deleted_files"] = self._cleanup_deleted_files(current_file_paths)
            return stats

        # Phase 2: Batch embed all chunks
        console.print(f"\n[cyan]Embedding {len(files_to_index)} files...[/cyan]")
        all_chunk_texts = []
        chunk_file_mapping = []  # Track which file each chunk belongs to
        
        for file_info in files_to_index:
            for chunk_text in file_info["chunk_texts"]:
                all_chunk_texts.append(chunk_text)
                chunk_file_mapping.append(file_info)
                stats["total_chunks"] += 1

        # Embed in batches with progress bar
        console.print(f"Generating embeddings for {len(all_chunk_texts)} chunks (batch size: {batch_size})...")
        all_embeddings = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding chunks", total=len(all_chunk_texts))
            
            for i in range(0, len(all_chunk_texts), batch_size):
                batch = all_chunk_texts[i:i + batch_size]
                batch_embeddings = self.embedder.embed(batch)
                all_embeddings.append(batch_embeddings)
                progress.advance(task, advance=len(batch))
        
        # Concatenate all batch embeddings
        import numpy as np
        all_embeddings = np.vstack(all_embeddings)

        # Phase 3: Store everything
        console.print("[cyan]Storing indexed data...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Storing chunks", total=len(files_to_index))
            
            chunk_idx = 0
            for file_info in files_to_index:
                try:
                    relative_path = file_info["relative_path"]
                    progress.update(task, description=f"[bold blue]Storing[/bold blue] [cyan]{relative_path}[/cyan]")
                    
                    # Store file and its chunks
                    num_chunks = len(file_info["chunks"])
                    file_embeddings = all_embeddings[chunk_idx:chunk_idx + num_chunks]
                    
                    self._store_file_and_chunks(file_info, file_embeddings)
                    
                    # Show status
                    action = file_info["action"]
                    status_color = "green" if action == "new" else "yellow"
                    progress.console.print(
                        f"  [{status_color}]{action:8}[/{status_color}] "
                        f"{relative_path} ({num_chunks} chunks)",
                        highlight=False
                    )
                    
                    chunk_idx += num_chunks

                except Exception as e:
                    console.print(f"[red]✗ Error storing {file_info['relative_path']}: {e}[/red]")
                    chunk_idx += len(file_info["chunks"])

                progress.advance(task)

        # Cleanup deleted files
        stats["deleted_files"] = self._cleanup_deleted_files(current_file_paths)

        return stats

    def _prepare_file_for_indexing(self, file_path: Path, relative_path: str) -> dict | None:
        """
        Prepare a file for indexing by checking if it needs indexing and chunking it.

        Args:
            file_path: Absolute path to the file
            relative_path: Path relative to repository root

        Returns:
            Dictionary with file info and chunks, or None if file should be skipped
        """
        # Get file mtime
        mtime = os.path.getmtime(file_path)

        # Check if file exists in database
        file_record = self.storage.get_file_by_path(relative_path)

        # Determine action
        if file_record is None:
            action = "new"
            file_id = None
        elif abs(file_record.mtime - mtime) < 0.001:  # Account for float precision
            # File hasn't changed
            return None
        else:
            action = "updated"
            file_id = file_record.id
            # Delete old chunks (we'll add new ones later)
            self.storage.delete_chunks_by_file_id(file_record.id)

        # Get language
        language = get_language_from_extension(file_path.suffix)

        # Chunk the file
        chunks = chunk_file_by_lines(
            file_path, self.config.chunk_size_lines, self.config.chunk_overlap_lines
        )

        if not chunks:
            return None

        # Extract text content
        chunk_texts = [chunk[2] for chunk in chunks]

        return {
            "file_path": file_path,
            "relative_path": relative_path,
            "mtime": mtime,
            "language": language,
            "action": action,
            "file_id": file_id,
            "chunks": chunks,
            "chunk_texts": chunk_texts,
        }

    def _store_file_and_chunks(self, file_info: dict, embeddings: any) -> None:
        """
        Store file metadata and chunk embeddings.

        Args:
            file_info: File information dictionary from _prepare_file_for_indexing
            embeddings: Numpy array of embeddings for this file's chunks
        """
        # Upsert file record
        file_id = self.storage.upsert_file(
            file_info["relative_path"],
            file_info["mtime"],
            file_info["language"]
        )

        # Add chunks to vector index and database
        for i, (start_line, end_line, text) in enumerate(file_info["chunks"]):
            # Vector ID is the current index in the FAISS index
            vector_id = self.vector_index.next_id
            self.vector_index.next_id += 1

            # Add to vector index
            self.vector_index.add([vector_id], embeddings[i : i + 1])

            # Add to database
            self.storage.insert_chunk(file_id, start_line, end_line, vector_id)

    def _cleanup_deleted_files(self, current_file_paths: set[str]) -> int:
        """
        Remove files from the index that no longer exist.

        Args:
            current_file_paths: Set of current file paths

        Returns:
            Number of deleted files
        """
        all_files = self.storage.get_all_files()
        deleted_count = 0

        for file_record in all_files:
            if file_record.path not in current_file_paths:
                # File was deleted, remove from index
                self.storage.delete_chunks_by_file_id(file_record.id)
                self.storage.delete_file(file_record.id)
                deleted_count += 1

        if deleted_count > 0:
            console.print(f"[yellow]Cleaned up {deleted_count} deleted files[/yellow]")

        return deleted_count


def index_repo(root_dir: Path | None = None, batch_size: int = 32) -> dict[str, int]:
    """
    Index a repository.

    Args:
        root_dir: Root directory to index (defaults to current directory)
        batch_size: Number of chunks to embed per batch (default: 32)

    Returns:
        Indexing statistics
    """
    if root_dir is None:
        root_dir = Path.cwd()

    # Load configuration
    from .config import load_config

    config = load_config(root_dir)

    # Initialize storage
    db_path = get_db_path(root_dir)
    storage = LootStorage(db_path)
    storage.connect()
    storage.initialize_schema()

    # Initialize embedder
    embedder = Embedder(config.model_name, embedding_dim=config.embedding_dim)

    # Initialize or load vector index
    vector_index_path = get_vector_index_path(root_dir)
    vector_index = VectorIndex(embedder.dim, vector_index_path)

    if vector_index_path.exists():
        console.print("[cyan]Loading existing vector index...[/cyan]")
        vector_index.load()
    else:
        console.print("[cyan]Creating new vector index...[/cyan]")
        vector_index.initialize()

    # Create indexer and run
    indexer = Indexer(config, storage, embedder, vector_index)
    stats = indexer.index_repository(root_dir, batch_size=batch_size)

    # Save vector index
    console.print("[cyan]Saving vector index...[/cyan]")
    vector_index.save()

    # Close storage
    storage.close()

    return stats
