"""
Tests for file chunking functionality.
"""

from pathlib import Path
import tempfile
import pytest

from loot.file_filters import chunk_file_by_lines


def test_chunk_file_basic():
    """Test basic file chunking."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        # Write 100 lines
        for i in range(100):
            f.write(f"line {i+1}\n")
        temp_path = Path(f.name)

    try:
        # Chunk with size=20, overlap=5
        chunks = chunk_file_by_lines(temp_path, chunk_size=20, overlap=5)

        # Should have multiple chunks
        assert len(chunks) > 0

        # First chunk should start at line 1
        assert chunks[0][0] == 1

        # First chunk should have 20 lines
        assert chunks[0][1] == 20

        # Second chunk should start at line 16 (20 - 5 + 1)
        if len(chunks) > 1:
            assert chunks[1][0] == 16

    finally:
        # Cleanup
        temp_path.unlink()


def test_chunk_file_small():
    """Test chunking a file smaller than chunk size."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        # Write only 10 lines
        for i in range(10):
            f.write(f"line {i+1}\n")
        temp_path = Path(f.name)

    try:
        # Chunk with size=20, overlap=5
        chunks = chunk_file_by_lines(temp_path, chunk_size=20, overlap=5)

        # Should have exactly 1 chunk
        assert len(chunks) == 1

        # Should contain all 10 lines
        assert chunks[0][0] == 1
        assert chunks[0][1] == 10

    finally:
        temp_path.unlink()


def test_chunk_file_empty():
    """Test chunking an empty file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        temp_path = Path(f.name)

    try:
        chunks = chunk_file_by_lines(temp_path, chunk_size=20, overlap=5)

        # Should have no chunks
        assert len(chunks) == 0

    finally:
        temp_path.unlink()


def test_chunk_content():
    """Test that chunk content is correctly extracted."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write("def foo():\n")
        f.write("    return 42\n")
        f.write("\n")
        f.write("def bar():\n")
        f.write("    return 'hello'\n")
        temp_path = Path(f.name)

    try:
        chunks = chunk_file_by_lines(temp_path, chunk_size=3, overlap=1)

        # Should have 2 chunks
        assert len(chunks) == 2

        # First chunk content
        start, end, content = chunks[0]
        assert start == 1
        assert end == 3
        assert "def foo():" in content
        assert "return 42" in content

        # Second chunk content
        start, end, content = chunks[1]
        assert "def bar():" in content

    finally:
        temp_path.unlink()
