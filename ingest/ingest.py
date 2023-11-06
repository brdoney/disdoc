#!/usr/bin/env python3
from datetime import datetime
from enum import Enum
import glob
import importlib.util
from multiprocessing import Pool
import os
from pathlib import Path
from typing import List, Literal, NamedTuple, Set, Tuple, Union

import chromadb
import chromadb.api
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from env_var import load_env

# Will share dotenv with outer directory
if not load_dotenv():
    print(
        "Could not load .env file or it is empty."
        "Please check if it exists and is readable."
    )
    exit(1)


# Load environment variables
PERSIST_DIRECTORY = load_env("PERSIST_DIRECTORY")
SOURCE_DIRECTORY = load_env("SOURCE_DIRECTORY", "source_documents")
EMBEDDINGS_MODEL_NAME = load_env("EMBEDDINGS_MODEL_NAME")
SIMILARITY_METRIC = load_env("SIMILARITY_METRIC", choices=["cosine", "l2", "ip"])
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Chroma settings, used in client
CHROMA_SETTINGS = chromadb.config.Settings(
    persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)


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
    """A course document (e.g. FAQ page, assignment spec) that generated chunks"""
    Test = 2
    """PDF of a past test"""
    Empty = 3
    """A document of any type that generated no chunks. Usually happens when a
    document consists of just images, since no OCR is performed right now."""


class SegregatedDocuments(NamedTuple):
    docs: List[Document]
    tests: List[Document]


def load_single_document(
    file_path: str,
) -> Union[
    Tuple[Literal[DocumentType.Empty], str],
    Tuple[Literal[DocumentType.Doc, DocumentType.Test], List[Document]],
]:
    """Load a single document and split it into chunks from the given file path.

    If the file splits into zero chunks (the file doesn't contain text), simply
    returns this along with the file path.
    If the file does split into chunks, this returns what type of file it is
    along with a list of the chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        documents: List[Document] = loader.load_and_split(text_splitter)

        if len(documents) == 0:
            return DocumentType.Empty, file_path

        # midterm = midterm, final = final,
        # test = anything else (e.g. Test_1 back when we did multiple tests per sem)
        for key in ["midterm", "final", "test"]:
            if ext == ".pdf" and key in file_path.lower():
                doc_type = DocumentType.Test
                for doc in documents:
                    doc.metadata["type"] = key
                break
        else:
            doc_type = DocumentType.Doc
            for doc in documents:
                doc.metadata["type"] = "doc"

        # print(doc_type, file_path)
        return doc_type, documents

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(
    source_dir: str, ignored_files: List[str] = []
) -> Tuple[SegregatedDocuments, List[str]]:
    """
    Returns a list of all documents loaded from the source documents directory
    (ignoring specified files) along with a list of documents that had no
    readable content and should be blacklisted for the future.
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
    # print(ignored_files)
    # print(filtered_files)

    print(f"Loading {len(filtered_files)} new documents from {SOURCE_DIRECTORY}")

    with Pool(processes=os.cpu_count()) as pool:
        docs: List[Document] = []
        tests: List[Document] = []
        to_blacklist: List[str] = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for doc_type, new_docs in pool.imap_unordered(
                load_single_document, filtered_files
            ):
                if doc_type == DocumentType.Test:
                    tests.extend(new_docs)  # type: ignore
                elif doc_type == DocumentType.Doc:
                    docs.extend(new_docs)  # type: ignore
                elif doc_type == DocumentType.Empty:
                    to_blacklist.append(new_docs)  # type: ignore
                else:
                    raise ValueError(f"Unsupported document type {doc_type}")
                pbar.update()

    print(
        f"Loaded and split into {len(docs)} chunks of text for docs"
        f" and {len(tests)} chunks of text for tests (max. {CHUNK_SIZE} tokens each)"
    )

    # print(docs)

    return SegregatedDocuments(docs=docs, tests=tests), to_blacklist


def does_vectorstore_exist(
    persist_directory: str, embeddings: HuggingFaceEmbeddings
) -> bool:
    """Checks if vectorstore exists"""
    db = Chroma(
        collection_name="docs",
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    if not db.get()["documents"]:
        return False
    return True


def get_stored_file_sources(collections: List[Chroma]) -> List[str]:
    """Get a list of the paths to all the files stored in the given collections."""
    sources = []
    for collection in collections:
        data = collection.get()
        sources.extend([metadata["source"] for metadata in data["metadatas"]])
    return sources


def get_file_sources(docs: List[Document]) -> Set[str]:
    """Get a set of all of the file sources for the given list."""
    return {doc.metadata["source"] + "\n" for doc in docs}


def write_sources(docs: SegregatedDocuments) -> None:
    """Write the path of each file that was added to the database in this run
    (based on the given docs).

    This will add to the docs.txt and tests.txt in the records directory and
    create timestamped <date>-docs.txt and <date>-tests.txt files in the
    records directory with just the files added in this run.
    """
    print("Adding the sources that were added to the database")

    records = Path("./records")
    records.mkdir(exist_ok=True)

    docs_sources = get_file_sources(docs.docs)
    tests_sources = get_file_sources(docs.tests)

    with (records / "docs.txt").open("a+") as f:
        f.writelines(docs_sources)
    with (records / "tests.txt").open("a+") as f:
        f.writelines(tests_sources)

    # Write the indiviual logs
    dt = datetime.strftime(datetime.now(), "%Y-%m-%d-%H.%M.%S")
    with (records / f"{dt}-docs.txt").open("w") as f:
        f.writelines(docs_sources)
    with (records / f"{dt}-tests.txt").open("w") as f:
        f.writelines(tests_sources)


def add_documents(
    collection: Chroma, collection_name: str, docs: List[Document]
) -> None:
    """Add documents to the given collection or do nothing if no documents are given."""
    if len(docs) == 0:
        print(f"No new documents for collection {collection_name}")
        return

    print(
        f"Creating embeddings for {len(docs)} new chunks in '{collection_name}'."
        " May take some minutes..."
    )
    collection.add_documents(docs)


def read_blacklist() -> List[str]:
    """Read the blacklist from the disk or returns an empty list."""
    try:
        with open("./blacklist.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        return []


def add_to_blacklist(paths_to_add: List[str]):
    """Add the specified file paths to the saved blacklist."""
    # a+ to make the file if it doesn't exist
    with open("./blacklist.txt", "a+") as f:
        for file_path in paths_to_add:
            f.write(file_path + "\n")


def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # Chroma client
    chroma_client = chromadb.PersistentClient(
        settings=CHROMA_SETTINGS, path=PERSIST_DIRECTORY
    )

    docs_db = Chroma(
        collection_name="docs",
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
        client=chroma_client,
        collection_metadata={"hnsw:space": SIMILARITY_METRIC},
    )
    tests_db = Chroma(
        collection_name="tests",
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
        client=chroma_client,
        collection_metadata={"hnsw:space": SIMILARITY_METRIC},
    )
    if does_vectorstore_exist(PERSIST_DIRECTORY, embeddings):
        print(f"Appending to existing vectorstore at {PERSIST_DIRECTORY}")

        ignored = read_blacklist()
        ignored.extend(get_stored_file_sources([docs_db, tests_db]))

        texts, to_blacklist = load_documents(SOURCE_DIRECTORY, ignored)
    else:
        print(f"Creating new vectorstore at {PERSIST_DIRECTORY}")
        texts, to_blacklist = load_documents(SOURCE_DIRECTORY)

    add_to_blacklist(to_blacklist)

    write_sources(texts)

    add_documents(docs_db, "docs", texts.docs)
    add_documents(tests_db, "tests", texts.tests)

    docs_db.persist()
    docs_db = None
    tests_db.persist()
    tests_db = None

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
