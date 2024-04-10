#!/usr/bin/env python3
from collections.abc import Iterable
import glob
import os
from datetime import datetime
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, NamedTuple

import chromadb
import chromadb.api
import chromadb.config
from dotenv import load_dotenv
from env_var import load_env
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from loaders import LOADER_MAPPING

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

# Chroma settings, used in client
CHROMA_SETTINGS = chromadb.config.Settings(
    persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

class DocumentType(Enum):
    Doc = 1
    """A course document (e.g. FAQ page, assignment spec) that generated chunks"""
    Test = 2
    """PDF of a past test"""
    Empty = 3
    """A document of any type that generated no chunks. Usually happens when a
    document consists of just images, since no OCR is performed right now."""


class SegregatedDocuments(NamedTuple):
    docs: list[Document]
    tests: list[Document]


def load_single_document(
    file_path: str,
) -> (
    tuple[Literal[DocumentType.Empty], str]
    | tuple[Literal[DocumentType.Doc, DocumentType.Test], list[Document]]
):
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
        documents: list[Document] = loader.load_and_split(text_splitter)

        if len(documents) == 0:
            return DocumentType.Empty, file_path

        # midterm = midterm, final = final,
        # test = anything else (e.g. Test_1 back when we did multiple tests per sem)
        for key in ["midterm", "final", "test"]:
            if ext == ".pdf" and key in file_path.lower():
                doc_type = DocumentType.Test
                for doc in documents:
                    doc.metadata["type"] = key  # type: ignore[reportUnknownMemberType]
                break
        else:
            doc_type = DocumentType.Doc
            for doc in documents:
                doc.metadata["type"] = "doc"  # type: ignore[reportUnknownMemberType]

        # print(doc_type, file_path)
        return doc_type, documents

    raise ValueError(f"Unsupported file extension '{ext}'")

def walk(path: Path) -> Iterable[Path]:
    for p in path.iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p

def load_documents(
    source_dir: str, ignored_files: list[str] | None = None
) -> tuple[SegregatedDocuments, list[str]]:
    """
    Returns a list of all documents loaded from the source documents directory
    (ignoring specified files) along with a list of documents that had no
    readable content and should be blacklisted for the future.
    """
    if ignored_files is None:
        ignored_files = []

    all_files: list[str] = []
    for p in walk(Path(source_dir)):
        ext = "".join(p.suffixes).lower()
        if ext in LOADER_MAPPING:
            all_files.append(ext)

    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]
    if len(filtered_files) == 0:
        print("No new documents to load")
        exit(0)
    # print(ignored_files)
    # print(filtered_files)

    print(f"Loading {len(filtered_files)} new documents from {source_dir}")

    with Pool(processes=os.cpu_count()) as pool:
        docs: list[Document] = []
        tests: list[Document] = []
        to_blacklist: list[str] = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for ret in pool.imap_unordered(load_single_document, filtered_files):
                # Done this way so mypy doesn't complain
                if ret[0] is DocumentType.Test:
                    docs = ret[1]
                    tests.extend(docs)
                elif ret[0] is DocumentType.Doc:
                    docs = ret[1]
                    docs.extend(docs)
                elif ret[0] is DocumentType.Empty:
                    file_path = ret[1]
                    to_blacklist.append(file_path)
                else:
                    raise ValueError(f"Unsupported document type {ret[0]}")
                _ = pbar.update()

    print(
        f"Loaded and split into {len(docs)} chunks of text for docs"
        + f" and {len(tests)} chunks of text for tests (max. {CHUNK_SIZE} tokens each)"
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


def get_stored_file_sources(collections: list[Chroma]) -> list[str]:
    """Get a list of the paths to all the files stored in the given collections."""
    sources: list[str] = []
    for collection in collections:
        data = collection.get()
        sources.extend([metadata["source"] for metadata in data["metadatas"]])  # type: ignore[reportAny]
    return sources


def get_file_sources(docs: list[Document]) -> set[str]:
    """Get a set of all of the file sources for the given list."""
    return {doc.metadata["source"] + "\n" for doc in docs}  # type: ignore[reportUnknownMemberType]


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
    collection: Chroma, collection_name: str, docs: list[Document]
) -> None:
    """Add documents to the given collection or do nothing if no documents are given."""
    if len(docs) == 0:
        print(f"No new documents for collection {collection_name}")
        return

    print(
        f"Creating embeddings for {len(docs)} new chunks in '{collection_name}'."
        + " May take some minutes..."
    )
    _ = collection.add_documents(docs)


def read_blacklist() -> list[str]:
    """Read the blacklist from the disk or returns an empty list."""
    try:
        with open("./blacklist.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        return []


def add_to_blacklist(paths_to_add: list[str]):
    """Add the specified file paths to the saved blacklist."""
    # a+ to make the file if it doesn't exist
    with open("./blacklist.txt", "a+") as f:
        for file_path in paths_to_add:
            _ = f.write(file_path + "\n")


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
