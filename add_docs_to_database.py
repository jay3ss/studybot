import argparse
import logging
import os
from typing import Generator, List

import weaviate
import weaviate.classes.config as wc
from langchain.globals import set_debug
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from text.loaders import DirectoryMultiFileLoader, DocumentLoader
from text.utils import convert_pdf_date_to_rfc3339
from tqdm import tqdm

from settings import settings

logging.basicConfig(level=logging.INFO)

set_debug(True)

client = weaviate.connect_to_local(
    headers={"X-OpenAI-Api-Key": settings.openai_api_key}
)

if not client.collections.exists("DocumentChunk"):
    client.collections.create(
        name="DocumentChunk",
        properties=[
            wc.Property(
                name="text",
                data_type=wc.DataType.TEXT,
                description="The text content of the document chunk",
            ),
            wc.Property(
                name="metadata",
                data_type=wc.DataType.OBJECT,
                description="Metadata associated with the document chunk",
                nested_properties=[
                    wc.Property(
                        name="source",
                        data_type=wc.DataType.TEXT,
                        description="The name of the document that the text is from",
                    ),
                    wc.Property(
                        name="author",
                        data_type=wc.DataType.TEXT,
                        description="The author of the document chunk",
                    ),
                    wc.Property(
                        name="publish_date",
                        data_type=wc.DataType.DATE,
                        description="The publication date of the document chunk",
                    ),
                    wc.Property(
                        name="index",
                        data_type=wc.DataType.INT,
                        description="The index of the subtitle (SRT) file",
                    ),
                    wc.Property(
                        name="start_time",
                        data_type=wc.DataType.TEXT,
                        description="The starting time of the subtitle",
                    ),
                    wc.Property(
                        name="end_time",
                        data_type=wc.DataType.TEXT,
                        description="The ending time of the subtitle",
                    ),
                ],
            ),
        ],
        vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
        generative_config=wc.Configure.Generative.openai(),
    )


# Load Documents
def load_documents(file_path: str) -> Generator[Document, None, None] | list:
    """Loads documents from a given file path."""
    if os.path.isdir(file_path):
        docs = DirectoryMultiFileLoader.load_documents(file_path, ["srt"])
    else:
        docs = DocumentLoader.load_documents(file_path)
    return docs


# Chunk Text
def chunk_text(documents: Generator[Document, None, None] | list) -> List[str]:
    """Chunks documents into smaller parts."""
    splitter = SemanticChunker(OpenAIEmbeddings(api_key=settings.openai_api_key))
    chunks = []

    for doc in documents:
        chunked_text = splitter.split_documents([doc])
        for chunk in chunked_text:
            chunks.append(
                {
                    "text": chunk.page_content,
                    "metadata": doc.metadata,
                }
            )

    return chunks


def ingest_document_metadata(metadata) -> dict:
    """Ingest a single document chunk's metadata"""
    return {
        "source": metadata.get("source", None),
        "author": metadata.get("author", None),
        "publish_date": convert_pdf_date_to_rfc3339(metadata.get("publish_date")),
        "index": convert_index_to_int(metadata.get("index", None)),
        "start_time": metadata.get("start_time", None),
        "end_time": metadata.get("end_time", None),
    }


def convert_index_to_int(index: str) -> int:
    if index:
        return int(index)
    return None


# Store chunks in database
def store_chunks(chunks: list[dict]) -> None:
    document_chunk = client.collections.get("DocumentChunk")
    with document_chunk.batch.dynamic() as batch:
        for chunk in tqdm(chunks):
            doc_chunk_obj = {
                "text": chunk["text"],
                "metadata": ingest_document_metadata(chunk["metadata"]),
            }
            batch.add_object(properties=doc_chunk_obj)

    if len(document_chunk.batch.failed_objects) > 0:
        logging.error(
            f"Failed to import {len(document_chunk.batch.failed_objects)} objects"
        )


# Main MVP Function
def main():
    parser = argparse.ArgumentParser(
        description="Add text to the databse for generating Anki cards from text documents."
    )

    parser.add_argument("-i", "--input", help="Path to the input text file(s).")

    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        help="Show logging statements",
    )

    args = parser.parse_args()

    logging.info("Loading documents...")
    documents = load_documents(args.input)
    logging.info("Chunking text...")
    chunks = chunk_text(documents)
    logging.info("Storing chunks...")
    store_chunks(chunks)
    logging.info("Done!")
    client.close()


# Example Usage
if __name__ == "__main__":
    main()
