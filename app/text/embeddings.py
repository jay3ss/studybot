from typing import Optional

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel

from settings import settings


class EmbeddingModel(BaseModel, Embeddings):
    """A protocol for embedding models that are both Pydantic BaseModel and LangChain Embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents."""
        pass

    def embed_query(self, query: str) -> list[float]:
        """Embeds a single query."""
        pass


def generate_embeddings(
    documents: list[Document],
    embeddings_model: Optional[EmbeddingModel] = OpenAIEmbeddings(
        api_key=settings.openai_api_key
    ),
) -> list:
    """
    Generates embeddings for a list of documents.

    This function takes a list of documents, uses the provided embedding model (which must conform to
    both Pydantic's BaseModel and LangChain's Embeddings), and returns a list of embeddings (each
    represented as a list of floats).

    If no embedding model is provided, the default model used is OpenAI's embedding model.

    Args:
        documents (list[Document]): A list of `Document` objects containing the content to be embedded.
        embeddings_model (EmbeddingModel, optional): The model used to generate embeddings. Defaults to
        OpenAIEmbeddings.

    Returns:
        list[list[float]]: A list of embeddings where each embedding is a list of floats corresponding to
        each document.

    Example:
        >>> documents = [Document(page_content="This is a sample document.")]
        >>> embeddings = generate_embeddings(documents)
        >>> print(embeddings)
    """

    embeddings = embeddings_model.embed_documents(
        [doc.page_content for doc in documents]
    )

    return embeddings
