import google.generativeai as genai
from tqdm.auto import tqdm
import time
import random
import os
from typing import List, Optional, Tuple
from pathlib import Path


def configure_gemini(api_key: Optional[str] = None):
    """
    Configure the Gemini API client.
    
    Args:
        api_key: Optional API key. If not provided, uses GEMINI_API_KEY from environment.
    """
    key = api_key or os.getenv("GOOGLE_GEMINI_KEY")
    if not key:
        raise ValueError("API key must be provided or set in GEMINI_API_KEY environment variable")
    genai.configure(api_key=key)


def embed_texts_batch(
    texts: List[str],
    api_key: str,
    batch_size: int = 10,
    max_texts: Optional[int] = None,
    model: str = "gemini-embedding-001",
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = 768,
    max_retries: int = 3,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Embed a list of texts using Gemini API with batching and retry logic.
    
    Note: You must call configure_gemini() before using this function.
    
    Args:
        texts: List of text strings to embed
        api_key: Gemini api key
        batch_size: Number of texts to process in each batch
        max_texts: Maximum number of texts to process (None for all)
        model: Gemini model to use for embeddings
        task_type: Task type for embeddings
        output_dimensionality: Dimension of output embeddings
        max_retries: Maximum number of retry attempts per batch
        show_progress: Whether to show progress bar
    
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    configure_gemini(api_key)
    # Limit texts if max_texts is specified
    if max_texts is not None:
        texts = texts[:max_texts]
    
    # Calculate batch start indices
    starts = list(range(0, len(texts), batch_size))
    
    def embed_batch(start_idx: int) -> Tuple[int, List[List[float]]]:
        """
        Embed a single batch with retry logic.
        
        Args:
            start_idx: Starting index for the batch
        
        Returns:
            Tuple of (start_idx, list of embeddings)
        """
        batch = texts[start_idx:start_idx + batch_size]
        
        for attempt in range(max_retries):
            try:
                # Process each text in the batch individually
                embeddings = []
                for text in batch:
                    result = genai.embed_content(
                        model=model,
                        content=text,
                        task_type=task_type,
                        output_dimensionality=output_dimensionality
                    )
                    embeddings.append(result['embedding'])
                return start_idx, embeddings
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with jitter
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                time.sleep(2 ** attempt + random.random() * 0.5)
    
    # Process all batches
    results_by_start = {}
    iterator = tqdm(starts, desc="Embedding") if show_progress else starts
    
    for batch_start_idx in iterator:
        start_idx, emb = embed_batch(batch_start_idx)
        results_by_start[start_idx] = emb
    
    # Reconstruct embeddings in original order
    all_embeddings = [e for s in starts for e in results_by_start[s]]
    
    # Verify we got embeddings for all texts
    assert len(all_embeddings) == len(texts), \
        f"Mismatch: {len(all_embeddings)} embeddings vs {len(texts)} texts"
    
    if show_progress:
        print(f"Successfully embedded {len(all_embeddings)} texts.")
    
    return all_embeddings


if __name__ == "__main__":
    texts = Path("./tests/data/texts.txt").read_text().splitlines()
    configure_gemini()
    embeddings = embed_texts_batch(texts, batch_size=3, max_texts=100, show_progress=True)
    print(embeddings)
