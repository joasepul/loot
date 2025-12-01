"""
File filtering utilities for loot.

Determines which files should be indexed based on extensions.
"""

from pathlib import Path


def get_language_from_extension(extension: str) -> str | None:
    """
    Infer programming language from file extension.

    Args:
        extension: File extension (e.g., ".py")

    Returns:
        Language name or None
    """
    language_map = {
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".scala": "scala",
        ".cs": "csharp",
        ".php": "php",
        ".rb": "ruby",
        ".sh": "shell",
        ".bash": "shell",
        ".sql": "sql",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".md": "markdown",
        ".txt": "text",
    }

    return language_map.get(extension.lower())


def should_include_file(file_path: Path, include_extensions: list[str]) -> bool:
    """
    Check if a file should be included based on its extension.

    Args:
        file_path: Path to the file
        include_extensions: List of extensions to include (e.g., [".py", ".js"])

    Returns:
        True if file should be included
    """
    return file_path.suffix.lower() in [ext.lower() for ext in include_extensions]


def iter_code_files(
    root_dir: Path, include_extensions: list[str], exclude_dirs: list[str] | None = None
) -> list[Path]:
    """
    Recursively find all code files in a directory.

    Args:
        root_dir: Root directory to search
        include_extensions: List of file extensions to include
        exclude_dirs: List of directory names to exclude (e.g., [".git", "node_modules"])

    Returns:
        List of file paths
    """
    if exclude_dirs is None:
        exclude_dirs = [
            ".git",
            ".loot",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".env",
            "dist",
            "build",
            ".next",
            ".nuxt",
            "target",
            "bin",
            "obj",
        ]

    code_files = []

    for path in root_dir.rglob("*"):
        # Skip if it's a directory
        if path.is_dir():
            continue

        # Skip if parent directory is in exclude list
        skip = False
        for parent in path.parents:
            if parent.name in exclude_dirs:
                skip = True
                break

        if skip:
            continue

        # Check if file should be included
        if should_include_file(path, include_extensions):
            code_files.append(path)

    return sorted(code_files)


def chunk_file_by_lines(
    file_path: Path, chunk_size: int, overlap: int
) -> list[tuple[int, int, str]]:
    """
    Split a file into overlapping chunks by line numbers.

    Args:
        file_path: Path to the file
        chunk_size: Number of lines per chunk
        overlap: Number of overlapping lines between chunks

    Returns:
        List of tuples: (start_line, end_line, text_content)
        Line numbers are 1-indexed.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        # Skip files that can't be read
        return []

    if not lines:
        return []

    chunks = []
    num_lines = len(lines)
    start = 0

    while start < num_lines:
        end = min(start + chunk_size, num_lines)

        # Get chunk text
        chunk_lines = lines[start:end]
        chunk_text = "".join(chunk_lines)

        # Store 1-indexed line numbers
        chunks.append((start + 1, end, chunk_text))

        # Move to next chunk with overlap
        start += chunk_size - overlap

        # Avoid infinite loop on small files
        if chunk_size <= overlap:
            break

    return chunks
