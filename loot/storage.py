"""
SQLite storage layer for loot.

Manages files and chunks metadata in the database.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class FileRecord:
    """Represents a file in the database."""

    id: int | None
    path: str
    mtime: float
    language: str | None = None


@dataclass
class ChunkRecord:
    """Represents a code chunk in the database."""

    id: int | None
    file_id: int
    start_line: int
    end_line: int
    vector_id: int


class LootStorage:
    """SQLite storage manager for loot."""

    def __init__(self, db_path: Path):
        """
        Initialize storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_schema(self) -> None:
        """Create database tables if they don't exist."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()

        # Files table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                mtime REAL NOT NULL,
                language TEXT
            )
            """
        )

        # Chunks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                vector_id INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id)
            )
            """
        )

        # Create indices for faster lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id)"
        )

        self.conn.commit()

    def upsert_file(self, path: str, mtime: float, language: str | None = None) -> int:
        """
        Insert or update a file record.

        Args:
            path: File path
            mtime: Modification timestamp
            language: Programming language

        Returns:
            File ID
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO files (path, mtime, language)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                mtime = excluded.mtime,
                language = excluded.language
            """,
            (path, mtime, language),
        )
        self.conn.commit()

        # Get the file ID
        cursor.execute("SELECT id FROM files WHERE path = ?", (path,))
        result = cursor.fetchone()
        return result[0] if result else cursor.lastrowid

    def get_file_by_path(self, path: str) -> FileRecord | None:
        """
        Get file record by path.

        Args:
            path: File path

        Returns:
            FileRecord or None if not found
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM files WHERE path = ?", (path,))
        row = cursor.fetchone()

        if not row:
            return None

        return FileRecord(
            id=row["id"], path=row["path"], mtime=row["mtime"], language=row["language"]
        )

    def get_file_by_id(self, file_id: int) -> FileRecord | None:
        """
        Get file record by ID.

        Args:
            file_id: File ID

        Returns:
            FileRecord or None if not found
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM files WHERE id = ?", (file_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return FileRecord(
            id=row["id"], path=row["path"], mtime=row["mtime"], language=row["language"]
        )

    def insert_chunk(
        self, file_id: int, start_line: int, end_line: int, vector_id: int
    ) -> int:
        """
        Insert a chunk record.

        Args:
            file_id: File ID
            start_line: Starting line number
            end_line: Ending line number
            vector_id: Vector ID in the ANN index

        Returns:
            Chunk ID
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (file_id, start_line, end_line, vector_id)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, start_line, end_line, vector_id),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_chunks_by_file_id(self, file_id: int) -> list[ChunkRecord]:
        """
        Get all chunks for a file.

        Args:
            file_id: File ID

        Returns:
            List of ChunkRecords
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE file_id = ?", (file_id,))
        rows = cursor.fetchall()

        return [
            ChunkRecord(
                id=row["id"],
                file_id=row["file_id"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                vector_id=row["vector_id"],
            )
            for row in rows
        ]

    def get_chunk_by_vector_id(self, vector_id: int) -> ChunkRecord | None:
        """
        Get chunk by vector ID.

        Args:
            vector_id: Vector ID

        Returns:
            ChunkRecord or None if not found
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE vector_id = ?", (vector_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return ChunkRecord(
            id=row["id"],
            file_id=row["file_id"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            vector_id=row["vector_id"],
        )

    def delete_chunks_by_file_id(self, file_id: int) -> list[int]:
        """
        Delete all chunks for a file.

        Args:
            file_id: File ID

        Returns:
            List of deleted vector IDs
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        # Get vector IDs before deleting
        chunks = self.get_chunks_by_file_id(file_id)
        vector_ids = [chunk.vector_id for chunk in chunks]

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self.conn.commit()

        return vector_ids

    def delete_file(self, file_id: int) -> None:
        """
        Delete a file record (chunks must be deleted first).

        Args:
            file_id: File ID
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self.conn.commit()

    def get_all_files(self) -> list[FileRecord]:
        """
        Get all file records.

        Returns:
            List of FileRecords
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM files")
        rows = cursor.fetchall()

        return [
            FileRecord(
                id=row["id"], path=row["path"], mtime=row["mtime"], language=row["language"]
            )
            for row in rows
        ]

    def get_stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with stats
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM files")
        file_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        return {"file_count": file_count, "chunk_count": chunk_count}
