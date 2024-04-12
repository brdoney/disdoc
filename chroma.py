import chromadb
import chromadb.api
import chromadb.config
from env_var import EMBEDDINGS_MODEL_NAME, PERSIST_DIRECTORY, SIMILARITY_METRIC
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma

# Chroma settings, used in client
CHROMA_SETTINGS = chromadb.config.Settings(
    persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)
COLLECTION_NAME = "docs"


def create_embeddings(show_progress_bar: bool = False) -> Embeddings:
    """Create an instance of the embeddings model we use."""
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"show_progress_bar": show_progress_bar}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def create_chroma_client():
    """Create an instance of the Chroma client we use."""
    return chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=PERSIST_DIRECTORY)


def create_chroma_collection(client: chromadb.api.API, embeddings: Embeddings):
    """Create an instance of the Chroma collection we use."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
        client=client,
        collection_metadata={"hnsw:space": SIMILARITY_METRIC},
    )
