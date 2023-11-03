#!/usr/bin/env python3
import glob
from multiprocessing import Pool
import os
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
import fitz
from langchain.docstore.document import Document
from langchain.document_loaders import (
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
    Blob,
)
from langchain.document_loaders.parsers.pdf import PyMuPDFParser
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm
import importlib.util

from constants import CHROMA_SETTINGS

if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


def load_env(env_var: str, *args) -> str:
    val = os.getenv(env_var, *args)
    if val is None:
        raise ValueError(
            f"Environment variable {env_var} must be defined in .env or otherwise."
        )
    return val


# Load environment variables
persist_directory = load_env("PERSIST_DIRECTORY")
source_directory = load_env("SOURCE_DIRECTORY", "source_documents")
embeddings_model_name = load_env("EMBEDDINGS_MODEL_NAME")
chunk_size = 500
chunk_overlap = 50


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
with open("./text_loader.txt") as f:
    for line in f.readlines():
        LOADER_MAPPING[line.strip()] = (TextLoader, {"encoding": "utf8"})

print(LOADER_MAPPING.keys())


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
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
        file_path
        for file_path in all_files
        if file_path not in ignored_files
        and "final" not in file_path.lower()
        and "midterm" not in file_path.lower()
        and "test" not in file_path.lower()
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for _, docs in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


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
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Chroma client
    chroma_client = chromadb.PersistentClient(
        settings=CHROMA_SETTINGS, path=persist_directory
    )

    if does_vectorstore_exist(persist_directory, embeddings):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
            client=chroma_client,
        )
        collection = db.get()
        texts = process_documents(
            [metadata["source"] for metadata in collection["metadatas"]]
        )
        print("Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print("Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS,
            client=chroma_client,
            # Use cos sim instead of l2 (default), since we're doing doc retrieval
            collection_metadata={"hnsw:space": "cosine"},
            collection_name="docs",
        )
    db.persist()
    db = None

    print("Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
