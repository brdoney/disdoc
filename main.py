import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import discord
from discord import app_commands
from dotenv import load_dotenv
from timeit import default_timer as timer

import os
import json

from pdf_images import pdf_image, save_image_cache, load_image_cache

load_dotenv()


def load(env_var: str) -> str:
    val = os.getenv(env_var)
    if val is None:
        raise ValueError(
            f"Environment variable {env_var} must be defined in .env or otherwise."
        )
    return val


DISCORD_TOKEN = load("DISCORD_TOKEN")
PERSIST_DIRECTORY = load("PERSIST_DIRECTORY")
EMBEDDINGS_MODEL_NAME = load("EMBEDDINGS_MODEL_NAME")

with open("/home/grads/brendandoney/Thesis/privateGPT/full-mappings.json") as f:
    NAME_TO_URL = json.load(f)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

chroma_client = chromadb.PersistentClient(
    settings=CHROMA_SETTINGS, path=PERSIST_DIRECTORY
)
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
    client=chroma_client,
    collection_name="docs",
    collection_metadata={"hnsw:space": "cosine"},
)

intents = discord.Intents.default()
intents.message_content = True


class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        synced = await self.tree.sync()
        print(f"Synced {len(synced)} command(s)")


client = MyClient(intents=intents)


@client.event
async def on_ready():
    print("Running Chroma Bot!")
    print(f"Logged in as {client.user} (ID: {client.user.id})")  # type: ignore
    print("------")


SOURCE_CODE_EXT = {
    ".csv": "text",
    # "html",
    ".md": "md",
    # "txt",
    # "text",
    ".java": "java",
    ".cpp": "cpp",
    ".h": "cpp",
    ".py": "python",
    ".c": "c",
    ".y": "c",
    ".cc": "cpp",
    ".l": "c",
    ".sh": "bash",
    "Makefile": "makefile",
    ".tst": "text",
    ".js": "js",
    ".S": "x86asm",
    ".s": "x86asm",
}


@client.tree.command(description="Ask for documents related to your question")
@app_commands.describe(
    question="Question or statement you want to find documents regarding"
)
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    start = timer()
    docs = await db.asimilarity_search_with_relevance_scores(question)
    # docs = await db.amax_marginal_relevance_search(question)
    end = timer()
    print(f"{end - start}s")
    embeds = []
    files = []
    for doc, score in docs:
        source = doc.metadata["source"].strip()
        name = source.removeprefix("source_documents/")

        url = NAME_TO_URL[name]
        doc_name = url.split("/")[-1]

        title = f"{score:.2%} {doc_name}"
        # title = f"{doc_name}"
        desc = doc.page_content

        # To make things consistent b/t ingest.py loaders and source_code_ext keys
        if "." in doc_name:
            ext = "." + doc_name.split(".")[-1]
        else:
            ext = doc_name

        if ext == ".pdf":
            # Page numbers start at 0 internally, but 1 in links
            page = int(doc.metadata["page"]) + 1
            url += f"#page={page}"
            title += f" - page {page}"

            image_url = pdf_image(doc_name, doc)
            if image_url is not None:
                file = discord.File(image_url)
                files.append(file)
                embed = discord.Embed(title=title, url=url)
                embed.set_image(url=f"attachment://{file.filename}")
            else:
                embed = discord.Embed(title=title, url=url, description=desc)
        else:
            if ext in SOURCE_CODE_EXT:
                desc = f"```{SOURCE_CODE_EXT[ext]}\n{desc}\n```"

            embed = discord.Embed(title=title, url=url, description=desc)

        embeds.append(embed)

    await interaction.followup.send(
        f"> {question}",
        files=files,
        embeds=embeds,
    )


# Load the docment -> image translations from memory
load_image_cache()

try:
    client.run(DISCORD_TOKEN)
finally:
    # We've loaded and maybe modified the image cache, so we need to save it
    print("Saving pdf image cache")
    save_image_cache()
