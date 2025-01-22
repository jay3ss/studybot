import hashlib
import os
import random
import re
from datetime import datetime, timedelta
from typing import Iterable, Literal, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from text.embeddings import EmbeddingModel

from settings import settings


def chunk_document(
    documents: Iterable[Document],
    embeddings_model: Optional[EmbeddingModel] = None,
    chunking_type: Literal["semantic", "recursive"] = "semantic",
    debug: bool = False,
    *args,
    **kwargs,
) -> list[Document]:
    """
    Splits documents into smaller chunks using the specified chunking strategy.

    Args:
        documents (Iterable[Document]): A collection of `Document` objects to split.
        embeddings_model (Optional[EmbeddingModel]): An embedding model instance used for semantic
        chunking.
            If `chunking_type` is "semantic" and no model is provided, defaults to `OpenAIEmbeddings`.
        chunking_type (Literal["semantic", "recursive"], optional): The type of chunking to apply.
            "semantic" uses embeddings for context-aware splitting, while "recursive" splits based on
            character length.
            Defaults to "semantic".
        *args: Additional positional arguments passed to the splitter class.
        **kwargs: Additional keyword arguments passed to the splitter class.

    Returns:
        list[Document]: A list of `Document` objects, each representing a chunk of the original content.

    Raises:
        ValueError: If an unsupported chunking type is specified or no embedding model is provided for
        semantic chunking.
    """

    if chunking_type == "semantic" and not embeddings_model:
        embeddings_model = OpenAIEmbeddings(
            api_key=settings.openai_api_key, show_progress_bar=debug
        )

    if chunking_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(*args, **kwargs)
    elif chunking_type == "semantic":
        splitter = SemanticChunker(embeddings_model, *args, **kwargs)
    else:
        raise ValueError(f"Unknown chunking_type: {chunking_type}")

    chunks = splitter.split_documents(documents)
    return chunks


def get_file_extension(filename: str) -> str:
    """Returns the file extension from the given filename"""
    return os.path.splitext(filename)[1]


def get_deck_id(file_path: str) -> int:
    """Generates a consistent deck ID based on the file path."""
    return int(hashlib.md5(file_path.encode()).hexdigest(), 16) % (10**10)


def clean_text(text: str) -> str:
    """Clean the text by fixing encoding, removing unnecessary whitespace,
    stop words, and normalizing the text."""

    # Ensure the text is in a valid encoding (utf-8)
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Remove extra spaces, newlines, and tabs
    text = re.sub(r"\s+", " ", text)

    # Normalize text: convert to lowercase
    text = text.lower()

    # Additional text normalization (e.g., expanding contractions, etc.)
    text = re.sub(
        r"(\w+)n't", r"\1 not", text
    )  # Expanding contractions like "isn't" to "is not"

    return text


def gen_rand_id() -> int:
    """Generates a random ID

    source: https://darigovresearch.github.io/genanki/build/html/overview.html#models
    """
    return random.randrange(1 << 30, 1 << 31)


def convert_pdf_date_to_rfc3339(pdf_date: str) -> str | None:
    try:
        if not pdf_date.startswith("D:"):
            raise ValueError("Invalid PDF date format")

        # Strip the "D:" prefix
        pdf_date = pdf_date[2:]

        # Extract date and time components
        date_part = pdf_date[:8]  # YYYYMMDD
        time_part = pdf_date[8:14]  # HHMMSS
        timezone_part = pdf_date[14:]  # ±HH'mm'

        # Parse the date and time
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")

        # Parse the timezone offset
        if timezone_part:
            sign = 1 if timezone_part[0] == "+" else -1
            hours_offset = int(timezone_part[1:3])
            minutes_offset = int(timezone_part[4:6])
            offset = sign * timedelta(hours=hours_offset, minutes=minutes_offset)
        else:
            offset = timedelta()  # Default to UTC if no timezone provided

        # Apply the timezone offset to get the correct time
        dt_with_offset = dt - offset

        # Format to RFC3339
        rfc3339 = dt_with_offset.strftime("%Y-%m-%dT%H:%M:%S")
        timezone_str = f"{timezone_part[0]}{timezone_part[1:3]}:{timezone_part[4:6]}"
        return f"{rfc3339}{timezone_str}"
    except Exception as e:
        return None
