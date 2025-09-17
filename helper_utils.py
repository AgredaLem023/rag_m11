# helper_utils.py
import numpy as np
import chromadb
import pandas as pd
from pypdf import PdfReader
import numpy as np


def project_embeddings(embeddings, umap_transform):
    """
    Projects the given embeddings using the provided UMAP transformer.

    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings


def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])


def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file with the modern pypdf API.
    Skips empty pages and returns a single newline-joined string.
    """
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            texts.append(t)
    return "\n".join(texts)


def load_chroma(filename, collection_name, embedding_function):
    """
    Loads a document from a PDF, extracts text, generates embeddings, and stores it in a Chroma collection.

    Args:
        filename (str): The path to the PDF file.
        collection_name (str): The name of the Chroma collection.
        embedding_function (callable): A function that takes a string or list[str] and returns an embedding
            vector or list of vectors.

    Returns:
        chromadb.api.models.Collection.Collection: The Chroma collection with the document embeddings.
    """
    # Extract text from the PDF
    text = extract_text_from_pdf(filename)

    # Split text into paragraphs or chunks and drop empties
    paragraphs = [p.strip() for p in text.split("\n\n") if p and p.strip()]

    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)

    if not paragraphs:
        return collection

    # Compute embeddings (try vectorized call first, then per-paragraph)
    try:
        raw_embeddings = embedding_function(paragraphs)
    except Exception:
        raw_embeddings = [embedding_function(p) for p in paragraphs]

    # Ensure embeddings are lists of floats
    embeddings = []
    for emb in raw_embeddings:
        arr = np.asarray(emb, dtype=float).ravel()
        embeddings.append(arr.tolist())

    # Prepare ids and batch add
    ids = [str(i) for i in range(len(paragraphs))]

    collection.add(
        ids=ids,
        documents=paragraphs,
        embeddings=embeddings,
    )

    return collection
