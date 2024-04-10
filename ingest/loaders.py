import importlib.util
from pathlib import Path
from typing import Any  # type: ignore[reportAny]

import fitz  # type: ignore[reportMissingTypeStubs]
from langchain.docstore.document import Document
from langchain.document_loaders import (
    Blob,
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.parsers.pdf import PyMuPDFParser
from langchain.document_loaders.pdf import BasePDFLoader
from typing_extensions import override


class MyEmailLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    @override
    def load(self) -> list[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


class MyPyMuPDFLoader(BasePDFLoader):
    """Load `PDF` files using `PyMuPDF`."""

    def __init__(self, file_path: str, **kwargs: Any) -> None:  # type: ignore[reportAny]
        """Initialize with a file path."""
        if importlib.util.find_spec("fitz") is None:
            raise ImportError(
                "`PyMuPDF` package not found, please install it with "
                + "`pip install pymupdf`"
            )

        super().__init__(file_path)
        self.text_kwargs = kwargs

    @override
    def load(self) -> list[Document]:
        """Load file."""

        parser = PyMuPDFParser(text_kwargs=self.text_kwargs)
        blob = Blob.from_path(self.file_path)
        return parser.parse(blob)


def get_ingest_dir():
    """Get the path to the ingest directory (which this file is contained in)."""
    return Path(__file__).parent


# Map file extensions to document loaders and their arguments
LOADER_MAPPING: dict[str, tuple[type[BaseLoader], dict[str, Any]]] = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (MyPyMuPDFLoader, {"flags": fitz.TEXT_DEHYPHENATE}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}

# Extend with our extra filetypes (i.e. code) that will fall back to text loader
with (get_ingest_dir() / "text_extensions.txt").open("r") as f:
    for line in f.readlines():
        LOADER_MAPPING[line.strip()] = (TextLoader, {"encoding": "utf8"})

print(LOADER_MAPPING.keys())
