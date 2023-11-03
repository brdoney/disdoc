#!/usr/bin/env python3
from enum import Enum
from functools import partial
import glob
import importlib.util
from multiprocessing import Pool
import os
from typing import List, NamedTuple, Tuple
from pathlib import Path

import chromadb
import chromadb.config
from dotenv import load_dotenv
import fitz
from langchain.docstore.document import Document
from langchain.document_loaders import (
    Blob,
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders.parsers.pdf import PyMuPDFParser
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from env_var import load_env

# Will share dotenv with outer directory
if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


# Load environment variables
PERSIST_DIRECTORY = load_env("PERSIST_DIRECTORY")
SOURCE_DIRECTORY = load_env("SOURCE_DIRECTORY", "source_documents")
EMBEDDINGS_MODEL_NAME = load_env("EMBEDDINGS_MODEL_NAME")
SIMILARITY_METRIC = load_env("SIMILARITY_METRIC", choices=["cosine", "l2", "ip"])
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
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

    def __init__(self, file_path: str, **kwargs) -> None:
        """Initialize with a file path."""
        if importlib.util.find_spec("fitz") is None:
            raise ImportError(
                "`PyMuPDF` package not found, please install it with "
                "`pip install pymupdf`"
            )

        super().__init__(file_path)
        self.text_kwargs = kwargs

    def load(self) -> List[Document]:
        """Load file."""

        parser = PyMuPDFParser(text_kwargs=self.text_kwargs)
        blob = Blob.from_path(self.file_path)
        return parser.parse(blob)


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (MyPyMuPDFLoader, {"flags": fitz.TEXT_DEHYPHENATE}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

# Extend with our extra filetypes (i.e. code) that will fall back to text loader
file_dir = Path(__file__).parent
with (file_dir / "text_loader.txt").open("r") as f:
    for line in f.readlines():
        LOADER_MAPPING[line.strip()] = (TextLoader, {"encoding": "utf8"})

print(LOADER_MAPPING.keys())


class DocumentType(Enum):
    Doc = 1
    Test = 2


class SegregatedDocuments(NamedTuple):
    docs: List[Document]
    tests: List[Document]


def load_single_document(
    splitter: TextSplitter, file_path: str
) -> Tuple[DocumentType, List[Document]]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        documents: List[Document] = loader.load_and_split(splitter)

        # midterm = midterm, final = final,
        # test = anything else (e.g. Test_1 back when we did multiple tests per sem)
        for key in ["midterm", "final", "test"]:
            if key in file_path.lower():
                doc_type = DocumentType.Test
                for doc in documents:
                    doc.metadata["type"] = key
                    break
        else:
            doc_type = DocumentType.Doc
            for doc in documents:
                doc.metadata["type"] = "doc"

        return doc_type, documents

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(
    source_dir: str, ignored_files: List[str] = []
) -> SegregatedDocuments:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]
    if len(filtered_files) == 0:
        print("No new documents to load")
        exit(0)

    print(f"Loading {len(filtered_files)} new documents from {SOURCE_DIRECTORY}")
    # print(filtered_files)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    load = partial(load_single_document, text_splitter)

    with Pool(processes=os.cpu_count()) as pool:
        docs = []
        tests = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for doc_type, docs in pool.imap_unordered(load, filtered_files):
                if doc_type == DocumentType.Test:
                    tests.extend(docs)
                elif doc_type == DocumentType.Doc:
                    docs.extend(docs)
                else:
                    raise ValueError(f"Unsupported document type {doc_type}")
                pbar.update()

    total_chunks = len(docs) + len(tests)
    print(
        f"Loaded and split into {total_chunks} chunks of text (max. {CHUNK_SIZE} tokens each)"
    )

    return SegregatedDocuments(docs, tests)


def process_documents(ignored_files: List[str] = []) -> SegregatedDocuments:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    return load_documents(SOURCE_DIRECTORY, ignored_files)


def does_vectorstore_exist(
    persist_directory: str, embeddings: HuggingFaceEmbeddings
) -> bool:
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()["documents"]:
        return False
    return True


def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # Chroma client
    chroma_settings = chromadb.config.Settings(
        persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
    )
    chroma_client = chromadb.PersistentClient(
        settings=chroma_settings, path=PERSIST_DIRECTORY
    )

    if does_vectorstore_exist(PERSIST_DIRECTORY, embeddings):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {PERSIST_DIRECTORY}")
        docs_db = Chroma(
            collection_name="docs",
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=chroma_settings,
            client=chroma_client,
            collection_metadata={"hnsw:space": SIMILARITY_METRIC},
        )
        tests_db = Chroma(
            collection_name="tests",
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=chroma_settings,
            client=chroma_client,
            collection_metadata={"hnsw:space": SIMILARITY_METRIC},
        )
        collection = docs_db.get()
        texts = process_documents(
            [metadata["source"] for metadata in collection["metadatas"]]
        )
        print(
            f"Creating embeddings for {len(texts.docs)} new course documents. May take some minutes..."
        )
        docs_db.add_documents(texts.docs)
        print(
            f"Creating embeddings for {len(texts.tests)} new tests. May take some minutes..."
        )
        tests_db.add_documents(texts.tests)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print("Creating embeddings. May take some minutes...")
        docs_db = Chroma.from_documents(
            texts.docs,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=chroma_settings,
            client=chroma_client,
            # Use cos sim instead of l2 (default), since we're doing doc retrieval
            collection_metadata={"hnsw:space": SIMILARITY_METRIC},
            collection_name="docs",
        )
        tests_db = Chroma.from_documents(
            texts.tests,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=chroma_settings,
            client=chroma_client,
            # Use cos sim instead of l2 (default), since we're doing doc retrieval
            collection_metadata={"hnsw:space": SIMILARITY_METRIC},
            collection_name="docs",
        )
    docs_db.persist()
    docs_db = None
    tests_db.persist()
    tests_db = None

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
