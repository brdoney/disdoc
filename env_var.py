import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


def load_env(
    env_var: str, default: str | None = None, choices: list[str] | None = None
) -> str:
    if default is not None:
        res = os.getenv(env_var, default)
        return res

    val = os.getenv(env_var)

    if val is None:
        raise ValueError(
            f"Environment variable {env_var} must be defined in .env or otherwise."
        )
    if choices is not None and val not in choices:
        raise ValueError(
            f"Value {val} for {env_var} not valid. Valid choices are: {choices}"
        )
    return val


if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)

SOURCE_DIRECTORY = Path(load_env("SOURCE_DIRECTORY", "source_documents"))
PERSIST_DIRECTORY = load_env("PERSIST_DIRECTORY")
DISCORD_TOKEN = load_env("DISCORD_TOKEN")
EMBEDDINGS_MODEL_NAME = load_env("EMBEDDINGS_MODEL_NAME")
MAPPINGS_PATH = load_env("MAPPINGS_PATH")
SIMILARITY_METRIC = load_env("SIMILARITY_METRIC", choices=["cosine", "l2", "ip"])
SQLITE_DB = load_env("SQLITE_DB")
CONSENT_URL = load_env("CONSENT_URL")
LLAMA_API_URL = load_env("LLAMA_API_URL")
CLICK_URL = urlparse(load_env("CLICK_URL"))
NUM_SEARCH_CHUNKS = int(load_env("NUM_SEARCH_CHUNKS"))
