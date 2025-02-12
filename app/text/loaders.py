import os
from typing import Generator

import pypdf
import srt
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from text.utils import get_file_extension


class DocumentLoader:

    @staticmethod
    def load_documents(file_path: str) -> list:
        """Load the documents using LangChain document loaders.

        Supported extensions:
        - .pdf (processed with PyPDFLoader)
        - .txt (processed with TextLoader)
        - .srt (processed with SRTLoader)
        """
        file_extension = get_file_extension(file_path)

        if file_extension == ".pdf":
            loader = EnhancedPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".srt":
            loader = SRTLoader(file_path)
        else:
            raise ValueError("Unsupported file extension:", file_extension)

        return loader.lazy_load()


class DirectoryMultiFileLoader:

    @staticmethod
    def load_documents(
        file_path: str, file_types: list = None, combine_srts: bool = True
    ) -> Generator[Document, None, None]:
        """Load the documents from a directory using LangChain document loaders.

        Supported extensions:
        - .pdf (processed with EnhancedPDFLoader)
        - .txt (processed with TextLoader)
        - .srt (processed with SRTLoader)

        Args:
            file_path (str): The path to the directory containing the files.
            file_types (list, optional): A list of file types to load. If None, all supported types are loaded.
            combine_srts (bool): If true, the SRT files will be combined into one Document.

        Yields:
            Document: A loaded document.
        """
        loaders: list[BaseLoader] = []

        if file_types is None or "pdf" in file_types:
            loaders.append(
                DirectoryLoader(
                    file_path,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    use_multithreading=True,
                )
            )

        if file_types is None or "txt" in file_types:
            loaders.append(
                DirectoryLoader(
                    file_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    use_multithreading=True,
                )
            )

        if file_types is None or "srt" in file_types:
            loaders.append(
                DirectoryLoader(
                    file_path,
                    glob="**/*.srt",
                    loader_cls=SRTLoader,
                    use_multithreading=True,
                )
            )

        # Load documents from all loaders
        all_documents = [
            # val for sublist in matrix for val in sublist
            document
            for loader in loaders
            for document in loader.lazy_load()
        ]

        if not file_types or any(ft.lower() == "srt" for ft in file_types):
            sorted_documents = sorted(
                all_documents,
                key=lambda doc: int(
                    os.path.basename(doc.metadata["source"]).split(" - ")[0]
                ),
            )

            if not sorted_documents:
                return []

            if combine_srts:
                docs_text = "\n\n".join(doc.page_content for doc in sorted_documents)

                sorted_documents = [Document(page_content=docs_text)]
                sorted_documents[0].metadata["source"] = file_path

            all_documents = sorted_documents

        for document in all_documents:
            yield document


class SRTLoader(BaseLoader):
    """A document loader that loads SRT files."""

    def __init__(self, file_path: str, ignore_errors: bool = False) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
            ignore_errors: Whether to ignore parsing errors in the SRT file.
        """
        self.file_path = file_path
        self.ignore_errors = ignore_errors

    def lazy_load(self) -> Generator[Document, None, None]:
        """Lazily loads the SRT file"""
        try:
            with open(self.file_path, encoding="utf-8") as f:
                subtitles = srt.parse(f.read(), ignore_errors=self.ignore_errors)
            for sub in subtitles:
                yield Document(
                    page_content=sub.content.strip(),
                    metadata={
                        "index": sub.index,
                        "start_time": str(sub.start),
                        "end_time": str(sub.end),
                        "source": self.file_path,
                    },
                )
        except Exception as e:
            raise ValueError(f"Failed to load SRT file: {self.file_path}") from e


class EnhancedPDFLoader(PyPDFLoader):
    def lazy_load(self) -> Generator[Document, None, None]:
        for document in super().lazy_load():
            metadata = self.extract_metadata(document.metadata.get("source"))
            document.metadata.update(metadata)
            yield document

    @staticmethod
    def extract_metadata(file_path: str) -> dict:
        """Extract metadata from a PDF file."""
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                info = reader.metadata
                return {
                    "author": info.get("/Author", None),
                    "publish_date": info.get("/CreationDate", None),
                    "source": file_path,
                }
        except Exception as e:
            print(f"Failed to extract metadata from {file_path}: {e}")
            return {"author": None, "publish_date": None, "source": None}
