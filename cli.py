from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Qdrant
import chromadb.config
from qdrant_client import QdrantClient
from dotenv import load_dotenv

import os
import asyncio
from timeit import default_timer as timer
from datetime import timedelta

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

CHROMA_SETTINGS = chromadb.config.Settings(
    persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)
chroma_client = chromadb.PersistentClient(
    settings=CHROMA_SETTINGS, path=PERSIST_DIRECTORY  # type: ignore
)

chroma_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
    client=chroma_client,
)

qdrant_dir = "/home/grads/brendandoney/Thesis/privateGPT/qdrant"
collection_name = "langchain"
qdrant_db = Qdrant._construct_instance(
    texts=["Scrap text"],
    path=qdrant_dir,
    collection_name=collection_name,
    embedding=embeddings,
)


async def main():
    while True:
        try:
            prompt = input("\nPrompt> ")
        except EOFError:
            return

        start = timer()
        docs = await chroma_db.asimilarity_search(prompt)
        end = timer()

        qstart = timer()
        qdocs = await qdrant_db.asimilarity_search(prompt)
        qend = timer()

        print(docs, f"{end - start}s\n")
        print(qdocs, f"{qend - qstart}s\n")


asyncio.run(main())
