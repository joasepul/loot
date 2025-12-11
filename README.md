# loot 🔍

**Local semantic code search** - A grep-like tool that uses embeddingGemma embeddings for semantic code search, completely on device.

Instead of matching exact text patterns, `loot` understands the *meaning* of your code. Search for "authentication logic" or "error handling" and find relevant code even if it doesn't contain those exact words.

## Features

- 🧠 **Semantic search** - Find code by meaning, not just keywords
- ⚡ **Incremental indexing** - Only re-indexes changed files
- 💻 **Fully local** - Uses on-device ONNX models, no cloud APIs
- 🎯 **MRL support** - Configurable embedding dimensions (768, 512, 256, 128)
- 🚀 **Fast** - FAISS-powered approximate nearest neighbor search
- 📦 **Simple** - Works like grep/ripgrep with a familiar CLI

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Install from source

```bash
# Clone or download the repository
cd loot

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize loot in your project

```bash
cd your-project
loot init
```

This creates a `.loot/` directory with configuration and an empty index.

### 2. Index your code

```bash
loot index
```

The first indexing will take a few minutes as it downloads the embedding model and processes all code files. Subsequent runs are fast thanks to incremental indexing.

### 3. Search semantically

```bash
# Find authentication-related code
loot search "user authentication logic"

# Find error handling
loot search "error handling and exceptions" -k 5

# Output as JSON
loot search "database queries" --json
```

## Usage

### Commands

#### `loot init`

Initialize loot in the current directory.

```bash
loot init                # Use default 768-dim embeddings
loot init --dim 512      # Use 512-dim embeddings (faster, slightly lower quality)
loot init --dim 256      # Use 256-dim embeddings (even faster)
```

Options:
- `--dim, -d`: Embedding dimension (768, 512, 256, or 128). Default: 768

#### `loot index`

Index code files in the current directory.

```bash
loot index
```

Loot will:
- Scan for code files recursively
- Skip files that haven't changed (using mtime)
- Re-index modified files
- Remove deleted files from the index

#### `loot search`

Search for code using semantic similarity.

```bash
loot search "query string"
loot search "authentication" --top-k 5
loot search "error handling" --max-lines 10
loot search "api endpoints" --json
```

Options:
- `--top-k, -k`: Number of results to return (default: 10)
- `--max-lines, -l`: Max lines to show per result (default: show all)
- `--json`: Output results as JSON

#### `loot stats`

Show statistics about the indexed code.

```bash
loot stats
```

Displays:
- Number of indexed files
- Total chunks
- Model configuration
- Embedding dimension
- Chunk size settings

## Configuration

Configuration is stored in `.loot/config.json`:

```json
{
  "model_name": "onnx-community/embeddinggemma-300m-ONNX",
  "embedding_dim": 768,
  "include_extensions": [
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".rs", ".go", ".java", ".c", ".cpp",
    ".rb", ".php", ".sh", ".sql", ".css",
    ".html", ".json", ".yaml", ".yml"
  ],
  "chunk_size_lines": 64,
  "chunk_overlap_lines": 16
}
```

### Configuration Options

- **model_name**: HuggingFace ONNX model to use for embeddings
- **embedding_dim**: Dimension of embeddings (768, 512, 256, or 128 for MRL)
- **include_extensions**: File extensions to index
- **chunk_size_lines**: Lines per chunk (default: 64)
- **chunk_overlap_lines**: Overlapping lines between chunks (default: 16)

## How It Works

1. **Chunking**: Files are split into overlapping chunks (default: 64 lines with 16 line overlap)
2. **Embedding**: Each chunk is converted to a semantic vector using EmbeddingGemma
3. **Indexing**: Vectors are stored in a FAISS index for fast similarity search
4. **Search**: Your query is embedded and matched against indexed chunks using cosine similarity

## Performance Tips

### Embedding Dimension

Lower dimensions = faster indexing and search, slightly lower quality:

- **768** (default), Best
- **512**: Good balance
- **256**: Fast, good for large codebases
- **128**: Fastest, still surprisingly effective

### Excluded Directories

By default, loot excludes common directories:
- `.git`, `.loot`, `node_modules`, `__pycache__`
- `.venv`, `venv`, `dist`, `build`, `target`
- `.next`, `.nuxt`, `bin`, `obj`

## Project Structure

```
your-project/
├── .loot/
│   ├── config.json      # Configuration
│   ├── index.db         # SQLite metadata
│   └── vectors.faiss    # Vector index
├── your-code-files...
```

## Examples

### Find authentication code
```bash
loot search "user login and authentication"
```

### Find error handling
```bash
loot search "exception handling and error recovery" -k 5
```

### Find API endpoints
```bash
loot search "REST API endpoint handlers"
```

### Find database operations
```bash
loot search "database queries and transactions"
```

## Limitations (v0)

- No deletion support in vector index (re-index to clean up)
- No multi-repo orchestration
- Single directory operation only
- FAISS CPU-only (no GPU acceleration yet)

## Development

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black loot/
ruff check loot/
```

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- [EmbeddingGemma](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) - Google's embedding model
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [ONNX Runtime](https://onnxruntime.ai/) - Efficient inference
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output
