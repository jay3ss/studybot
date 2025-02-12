from typing import Optional

from langchain.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document


def create_vector_store(
    documents: list[Document],
    embeddings: list[list[float]],
    vector_store_class: Optional[VectorStore] = FAISS,
) -> VectorStore:
    """
    Creates a vector store from a list of documents and their corresponding embeddings.

    This function takes a list of documents and their embeddings, and stores them in a vector store.
    The vector store allows for efficient similarity searches based on the document embeddings.

    Args:
        documents (list[Document]): A list of `Document` objects that contain the content.
        embeddings (list[list[float]]): A list of embeddings corresponding to the documents.
        vector_store_class (VectorStore, optional): The vector store class to use. Defaults to FAISS.

    Returns:
        VectorStore: The vector store containing the documents and embeddings, ready for similarity searches.

    Example:
        >>> documents = [Document(page_content="This is a sample document.")]
        >>> embeddings = generate_embeddings(documents)
        >>> vector_store = create_vector_store(documents, embeddings)
        >>> print(vector_store)
    """

    # Create the vector store using the provided embeddings and documents
    vector_store = vector_store_class.from_documents(documents, embeddings)

    return vector_store
