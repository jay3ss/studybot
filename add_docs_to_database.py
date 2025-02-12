import argparse
import logging
import os
from typing import Generator, List

import ollama
import weaviate
import weaviate.classes.config as wc
from langchain.globals import set_debug
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama.embeddings import OllamaEmbeddings
from text.loaders import DirectoryMultiFileLoader, DocumentLoader
from text.utils import convert_pdf_date_to_rfc3339
from tqdm import tqdm

from settings import settings

logging.basicConfig(level=logging.INFO)

set_debug(True)

client = weaviate.connect_to_local()

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
        vectorizer_config=wc.Configure.Vectorizer.text2vec_ollama(
            api_endpoint=settings.api_endpoint, model=settings.embeddings_model
        ),
        generative_config=wc.Configure.Generative.ollama(
            api_endpoint=settings.api_endpoint, model=settings.inference_model
        ),
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
    splitter = SemanticChunker(OllamaEmbeddings(model=settings.embeddings_model))
    chunks = []

    index = 0
    for doc in documents:
        chunked_text = splitter.split_documents([doc])
        for chunk in chunked_text:
            chunks.append(
                {"text": chunk.page_content, "metadata": doc.metadata, "index": index}
            )
            index += 1

    return chunks


def ingest_document_metadata(metadata) -> dict:
    """Ingest a single document chunk's metadata"""
    return {
        "source": metadata.get("source", None),
        "author": metadata.get("author", None),
        "publish_date": convert_pdf_date_to_rfc3339(metadata.get("publish_date")),
        "start_time": metadata.get("start_time", None),
        "end_time": metadata.get("end_time", None),
    }


def convert_index_to_int(index: str) -> int:
    if isinstance(index, str):
        return int(index)
    return index


# Store chunks in database
def store_chunks(chunks: list[dict]) -> None:
    document_chunk = client.collections.get("DocumentChunk")
    with document_chunk.batch.dynamic() as batch:
        for index, chunk in tqdm(enumerate(chunks)):
            doc_chunk_obj = {
                "text": chunk["text"],
                **ingest_document_metadata(chunk["metadata"]),
                "index": chunk["index"],
            }
            # doc_chunk_obj.update(ingest_document_metadata(chunk["metadata"]))
            response = ollama.embeddings(
                model=settings.embeddings_model, prompt=chunk["text"]
            )
            batch.add_object(properties=doc_chunk_obj, vector=response["embedding"])

    if len(document_chunk.batch.failed_objects) > 0:
        logging.error(
            f"Failed to import {len(document_chunk.batch.failed_objects)} objects"
        )


# Main MVP Function
def main():
    # parser = argparse.ArgumentParser(
    #     description="Add text to the databse for generating Anki cards from text documents."
    # )

    # parser.add_argument("-i", "--input", help="Path to the input text file(s).")

    # parser.add_argument(
    #     "-d",
    #     "--debug",
    #     default=False,
    #     help="Show logging statements",
    # )

    # args = parser.parse_args()

    # logging.info("Loading documents...")
    # documents = load_documents(args.input)
    doc_paths = [
        "P1L2 Text Browser Exercise (Analysis) Subtitles",
        "P1L3 Design Concepts Subtitles",
        "P2L1 Review of UML Subtitles",
        "SWEBOKv3_chapter2.pdf",
    ]
    for doc_path in doc_paths:
        logging.info(f"Loading {doc_path}")
        documents = load_documents(f"documents/{doc_path}")
        logging.info("Chunking text...")
        chunks = chunk_text(documents)
        logging.info("Storing chunks...")
        store_chunks(chunks)
        logging.info("Done!")
    client.close()


# Example Usage
if __name__ == "__main__":
    main()
