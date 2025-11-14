# BoEmbedder

A Python library for generating text embeddings using Google's Gemini API with batching, retry logic, and progress tracking.

## Installation

### From GitHub

```bash
pip install git+https://github.com/OpenPecha/BoEmbedder.git
```

### From Source

```bash
# Clone the repository
git clone https://github.com/OpenPecha/BoEmbedder.git
cd BoEmbedder

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Basic Example

```python
from BoEmbedder.gemini import embed_texts_batch

# Your texts to embed
texts = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one."
]

# Generate embeddings
embeddings = embed_texts_batch(
    texts=texts,
    api_key="your-gemini-api-key",
    batch_size=10,
    model="gemini-embedding-001",
    output_dimensionality=768
)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has dimension: {len(embeddings[0])}")
```

### Using Environment Variables

```python
import os
from BoEmbedder.gemini import embed_texts_batch

# Set your API key as an environment variable
os.environ["GOOGLE_GEMINI_KEY"] = "your-gemini-api-key"

# Embed texts (API key will be read from environment)
embeddings = embed_texts_batch(
    texts=texts,
    api_key=os.getenv("GOOGLE_GEMINI_KEY")
)
```

### Advanced Usage with Custom Parameters

```python
from BoEmbedder.gemini import embed_texts_batch

embeddings = embed_texts_batch(
    texts=texts,
    api_key="your-gemini-api-key",
    batch_size=5,              # Process 5 texts per batch
    max_texts=100,             # Limit to first 100 texts
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",  # Or "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"
    output_dimensionality=768,  # Embedding dimension
    max_retries=3,             # Retry failed batches up to 3 times
    show_progress=True         # Display progress bar
)
```

### Configuration Only

```python
from BoEmbedder.gemini import configure_gemini

# Configure Gemini API separately
configure_gemini(api_key="your-gemini-api-key")

# Or use environment variable
configure_gemini()  # Reads from GOOGLE_GEMINI_KEY
```

### Features

- **Batch Processing**: Process multiple texts efficiently in configurable batch sizes
- **Automatic Retries**: Built-in retry logic with exponential backoff and jitter
- **Progress Tracking**: Optional progress bar using tqdm
- **Flexible Configuration**: Support for different models, task types, and embedding dimensions
- **Error Handling**: Robust error handling with detailed error messages
- **Environment Variable Support**: API keys can be provided directly or via environment variables

### API Reference

#### `embed_texts_batch()`

Generate embeddings for a list of texts.

**Parameters:**
- `texts` (List[str]): List of text strings to embed
- `api_key` (str): Gemini API key
- `batch_size` (int, optional): Number of texts per batch (default: 10)
- `max_texts` (int, optional): Maximum number of texts to process (default: None, processes all)
- `model` (str, optional): Gemini model name (default: "gemini-embedding-001")
- `task_type` (str, optional): Task type for embeddings (default: "RETRIEVAL_DOCUMENT")
- `output_dimensionality` (int, optional): Embedding dimension (default: 768)
- `max_retries` (int, optional): Maximum retry attempts per batch (default: 3)
- `show_progress` (bool, optional): Show progress bar (default: True)

**Returns:**
- `List[List[float]]`: List of embeddings, where each embedding is a list of floats

#### `configure_gemini()`

Configure the Gemini API client.

**Parameters:**
- `api_key` (str, optional): API key. If not provided, uses `GOOGLE_GEMINI_KEY` from environment

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests if available (`pytest`)
5. Commit your changes (`git commit -am 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Create a new Pull Request

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=BoEmbedder
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2023 OpenPecha

---

**Maintained by:** [OpenPecha](https://github.com/OpenPecha)

**Questions or Issues?** Please file an issue on our [GitHub Issues](https://github.com/OpenPecha/BoEmbedder/issues) page.
