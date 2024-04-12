#!/usr/bin/env python3
import os
import pprint
from collections.abc import Iterable, Iterator
from datetime import datetime
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import chromadb
import chromadb.api
import chromadb.config
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from env_var import load_env
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from .loaders import LOADER_MAPPING, get_ingest_dir

# Will share dotenv with outer directory
if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


# Load environment variables
PERSIST_DIRECTORY = load_env("PERSIST_DIRECTORY")
SOURCE_DIRECTORY = Path(load_env("SOURCE_DIRECTORY", "source_documents"))
EMBEDDINGS_MODEL_NAME = load_env("EMBEDDINGS_MODEL_NAME")
SIMILARITY_METRIC = load_env("SIMILARITY_METRIC", choices=["cosine", "l2", "ip"])
COLLECTION_NAME = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Chroma settings, used in client
CHROMA_SETTINGS = chromadb.config.Settings(
    persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)


def get_tag(file_path: Path) -> str:
    # Will look like ex0/cs3214/..., so we can use parts[0]
    rel = file_path.relative_to(SOURCE_DIRECTORY)
    return rel.parts[0]


def load_single_document(
    file_path: Path,
) -> tuple[Literal[True], list[Document]] | tuple[Literal[False], Path]:
    """Load a single document and split it into chunks from the given file path.

    If the file splits into zero chunks (the file doesn't contain text), simply
    returns `False` along with the file path.
    If the file does split into chunks, returns `True` along with a list of the chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    doc_name = file_path.stem.lower()
    ext = file_path.suffix.lower() if file_path.suffix else doc_name

    if ext in LOADER_MAPPING or doc_name in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(str(file_path), **loader_args)
        documents: list[Document] = loader.load_and_split(text_splitter)

        if len(documents) == 0:
            print("Empty", file_path)
            return False, file_path

        tag = get_tag(file_path)
        for doc in documents:
            doc.metadata["group"] = tag  # type: ignore[reportUnknownMemberType]

        return True, documents

    raise ValueError(f"Unsupported file {file_path} from {file_path}'")


def walk(path: Path) -> Iterable[Path]:
    for p in path.iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p


def load_documents(
    source_dir: Path,
    previous_files: set[str] | None = None,
    blacklist: set[str] | None = None,
) -> tuple[list[Document], set[str]]:
    """
    Returns a list of all documents loaded from the source documents directory
    (ignoring specified files) along with a new blacklist, which includes any new files
    that had no readable content.
    """
    if previous_files is None:
        previous_files = set()

    if blacklist is None:
        blacklist = set()
    new_blacklist = blacklist.copy()

    ignored_files = blacklist.union(previous_files)

    files: list[Path] = []
    for p in walk(source_dir):
        doc_name = p.name.lower()
        ext = p.suffix.lower()
        if str(p) not in ignored_files:
            if ext in LOADER_MAPPING or doc_name in LOADER_MAPPING:
                print(p, "accepted")
                files.append(p)
            else:
                print(f"{p} invalid extension/name: '{doc_name}' '{ext}'")
                new_blacklist.add(str(p))
        else:
            print(p, "filtered out")

    if len(files) == 0:
        print("No new documents to load")
        exit(0)

    pprint.pp(files)

    print(f"Loading {len(files)} new documents from {source_dir}")

    # Leave one CPU for this process
    cpus = os.cpu_count()
    if cpus is not None:
        cpus //= 2

    with Pool(processes=cpus) as pool:
        docs: list[Document] = []
        with tqdm(total=len(files), desc="Loading new documents", ncols=80) as pbar:
            for ret in pool.imap_unordered(load_single_document, files):
                # Done this way so mypy doesn't complain
                if ret[0] is True:
                    new_docs = ret[1]
                    docs.extend(new_docs)
                else:
                    file_path = ret[1]
                    new_blacklist.add(str(file_path))
                _ = pbar.update()

    print(
        f"Loaded and split into {len(docs)} chunks of text (max. {CHUNK_SIZE} tokens each)"
    )

    return docs, new_blacklist


def get_stored_file_sources(collections: list[Chroma]) -> set[str]:
    """Get a list of the paths to all the files stored in the given collections."""
    sources: set[str] = set()
    for collection in collections:
        data = collection.get()
        sources.update([metadata["source"] for metadata in data["metadatas"]])  # type: ignore[reportAny]
    return sources


def get_file_sources(docs: list[Document]) -> set[str]:
    """Get a set of all of the file sources for the given list."""
    return {doc.metadata["source"] + "\n" for doc in docs}  # type: ignore[reportUnknownMemberType]


def write_sources(docs: list[Document]) -> None:
    """Write the path of each file that was added to the database in this run
    (based on the given docs).

    This will add to the docs.txt and tests.txt in the records directory and
    create timestamped <date>-docs.txt and <date>-tests.txt files in the
    records directory with just the files added in this run.
    """
    print("Adding the sources that were added to the database")

    records = get_ingest_dir() / "records"
    records.mkdir(exist_ok=True)

    sources = get_file_sources(docs)

    with (records / "docs.txt").open("a+") as f:
        f.writelines(sources)

    # Write the indiviual logs
    dt = datetime.strftime(datetime.now(), "%Y-%m-%d-%H.%M.%S")
    with (records / f"{dt}-docs.txt").open("w") as f:
        f.writelines(sources)


def create_batches(
    client: chromadb.api.API, docs: list[Document]
) -> Iterator[list[Document]]:
    """Yield batches of documents (for ingesting) based on the client's maximum batch size."""
    print(f"Yielding {ceil(len(docs) / client.max_batch_size)} batches")
    for i in range(0, len(docs), client.max_batch_size):
        yield docs[i : min(i + client.max_batch_size, len(docs))]


def add_documents(
    client: chromadb.api.API,
    collection: Chroma,
    collection_name: str,
    docs: list[Document],
) -> None:
    """Add documents to the given collection or do nothing if no documents are given."""
    if len(docs) == 0:
        print(f"No new documents for collection {collection_name}")
        return

    print(
        f"Creating embeddings for {len(docs)} new chunks in '{collection_name}'."
        + " May take some minutes..."
    )

    for batch in create_batches(client, docs):
        _ = collection.add_documents(batch)


def read_blacklist() -> set[str]:
    """Read and return the blacklist from the disk or return an empty list if no blacklist exists."""
    try:
        with (get_ingest_dir() / "blacklist.txt").open("r") as f:
            return {line.strip() for line in f.readlines()}
    except FileNotFoundError:
        return set()


def write_blacklist(paths_to_add: set[str]):
    """Write out the blacklist, overwriting any previous version."""
    with (get_ingest_dir() / "blacklist.txt").open("w") as f:
        for file_path in paths_to_add:
            _ = f.write(file_path + "\n")


def ask_reset(client: chromadb.api.API):
    answer = input("Would you like to reset the db? [y/N] ").lower()
    if answer == "y":
        res = client.reset()
        if res:
            print("Successfully reset db")
        else:
            print("Unable to reset db")


def get_collection(client: chromadb.api.API, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
        client=client,
        collection_metadata={"hnsw:space": SIMILARITY_METRIC},
    )


def does_vectorstore_exist(
    client: chromadb.api.API,
    embeddings: Embeddings,
) -> bool:
    """Checks if vectorstore exists"""
    db = get_collection(client, embeddings)
    if not db.get()["documents"]:
        return False
    return True


def main():
    # Create embeddings
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"show_progress_bar": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Chroma client
    chroma_client = chromadb.PersistentClient(
        settings=CHROMA_SETTINGS, path=PERSIST_DIRECTORY
    )

    db = get_collection(chroma_client, embeddings)

    if does_vectorstore_exist(chroma_client, embeddings):
        print(f"Appending to existing vectorstore at {PERSIST_DIRECTORY}")

        ask_reset(chroma_client)

        blacklist = read_blacklist()
        ignored = get_stored_file_sources([db])

        docs, blacklist = load_documents(SOURCE_DIRECTORY, ignored, blacklist)
    else:
        print(f"Creating new vectorstore at {PERSIST_DIRECTORY}")
        docs, blacklist = load_documents(SOURCE_DIRECTORY)

    write_blacklist(blacklist)

    write_sources(docs)

    add_documents(chroma_client, db, "docs", docs)

    db.persist()
    db = None

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
