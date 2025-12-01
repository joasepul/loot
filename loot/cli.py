"""
Command-line interface for loot.

Provides commands: init, index, search, stats.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import (
    LootConfig,
    ensure_loot_dir,
    get_config_path,
    get_db_path,
    is_initialized,
    save_config,
)
from .indexer import index_repo
from .search import format_search_results, semantic_search
from .storage import LootStorage

app = typer.Typer(
    name="loot",
    help="Local semantic code search tool using on-device embeddings",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"loot version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """loot - Local semantic code search"""
    pass


@app.command()
def init(
    embedding_dim: int = typer.Option(
        768,
        "--dim",
        "-d",
        help="Embedding dimension (768, 512, 256, or 128 for EmbeddingGemma MRL)",
    ),
):
    """
    Initialize loot in the current directory.

    Creates .loot/ directory with configuration and empty index.
    Uses onnx-community/embeddinggemma-300m-ONNX as the default model.
    """
    root_dir = Path.cwd()

    if is_initialized(root_dir):
        console.print(
            "[yellow]⚠ loot is already initialized in this directory[/yellow]"
        )
        overwrite = typer.confirm("Do you want to overwrite the existing configuration?")
        if not overwrite:
            console.print("[cyan]Initialization cancelled[/cyan]")
            raise typer.Exit()

    console.print(f"[cyan]Initializing loot in {root_dir}[/cyan]\n")

    # Create .loot directory
    loot_dir = ensure_loot_dir(root_dir)
    console.print(f"✓ Created directory: {loot_dir}", style="green")

    # Create configuration with defaults
    config = LootConfig(embedding_dim=embedding_dim)
    save_config(config, root_dir)

    # Initialize empty database
    db_path = get_db_path(root_dir)
    with LootStorage(db_path) as storage:
        storage.initialize_schema()
        console.print(f"✓ Created database: {db_path}", style="green")

    console.print(f"\n[green]✓ loot initialized successfully![/green]")
    console.print(f"\nConfiguration:")
    console.print(f"  Model: [cyan]{config.model_name}[/cyan]")
    console.print(f"  Embedding dimension: [cyan]{config.embedding_dim}[/cyan]")
    console.print("\nNext steps:")
    console.print("  1. Run [cyan]loot index[/cyan] to index your code")
    console.print("  2. Run [cyan]loot search 'your query'[/cyan] to search")
    console.print("\nNote: Edit .loot/config.json to customize settings if needed")


@app.command()
def index(
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Number of chunks to embed in each batch (default: 32)"
    ),
):
    """
    Index code files in the current directory.

    Uses incremental indexing - only re-indexes changed files.
    """
    root_dir = Path.cwd()

    if not is_initialized(root_dir):
        console.print(
            "[red]✗ loot is not initialized. Run 'loot init' first.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Indexing code in {root_dir}[/cyan]")

    try:
        stats = index_repo(root_dir, batch_size=batch_size)

        # Print statistics
        console.print("\n[green]✓ Indexing complete![/green]\n")

        table = Table(title="Indexing Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("New files", str(stats["new_files"]))
        table.add_row("Updated files", str(stats["updated_files"]))
        table.add_row("Skipped files", str(stats["skipped_files"]))
        table.add_row("Deleted files", str(stats["deleted_files"]))
        table.add_row("Total chunks indexed", str(stats["total_chunks"]))

        console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error during indexing: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return"),
    max_lines: int = typer.Option(
        None,
        "--max-lines",
        "-l",
        help="Maximum lines to show per result (None = show all)",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON"
    ),
):
    """
    Search for code using semantic similarity.

    Example:
        loot search "authentication logic"
        loot search "error handling" -k 5
        loot search "database queries" --json
    """
    root_dir = Path.cwd()

    if not is_initialized(root_dir):
        console.print(
            "[red]✗ loot is not initialized. Run 'loot init' first.[/red]"
        )
        raise typer.Exit(1)

    try:
        console.print(f'[cyan]Searching for:[/cyan] "{query}"\n')

        results = semantic_search(query, top_k=top_k, root_dir=root_dir)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            raise typer.Exit(0)

        # Format and display results
        formatted = format_search_results(
            results, max_lines=max_lines, as_json=json_output
        )

        if json_output:
            console.print(formatted)
        else:
            console.print(f"[green]Found {len(results)} results:[/green]\n")
            console.print(formatted)

    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error during search: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats():
    """
    Show statistics about the indexed code.
    """
    root_dir = Path.cwd()

    if not is_initialized(root_dir):
        console.print(
            "[red]✗ loot is not initialized. Run 'loot init' first.[/red]"
        )
        raise typer.Exit(1)

    try:
        from .config import get_vector_index_path, load_config
        from .vector_index import VectorIndex

        # Load configuration
        config = load_config(root_dir)

        # Get database stats
        db_path = get_db_path(root_dir)
        with LootStorage(db_path) as storage:
            db_stats = storage.get_stats()

        # Get vector index stats
        vector_index_path = get_vector_index_path(root_dir)
        vector_size = 0
        if vector_index_path.exists():
            vector_index = VectorIndex(config.embedding_dim, vector_index_path)
            vector_index.load()
            vector_size = vector_index.size

        # Display statistics
        table = Table(title="Loot Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Root directory", str(root_dir))
        table.add_row("Model", config.model_name)
        table.add_row("Embedding dimension", str(config.embedding_dim))
        table.add_row("Indexed files", str(db_stats["file_count"]))
        table.add_row("Total chunks", str(db_stats["chunk_count"]))
        table.add_row("Vectors in index", str(vector_size))
        table.add_row("Chunk size", f"{config.chunk_size_lines} lines")
        table.add_row("Chunk overlap", f"{config.chunk_overlap_lines} lines")

        console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error getting stats: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
