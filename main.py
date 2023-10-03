import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import discord
from discord import app_commands
from dotenv import load_dotenv

import os
import json
import pathlib

load_dotenv()


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")

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
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("------")


@client.tree.command(description="Ask for documents related to your question")
@app_commands.describe(
    question="Question or statement you want to find documents regarding"
)
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    docs = await db.asimilarity_search_with_relevance_scores(question)
    embeds = []
    for doc, score in docs:
        source = doc.metadata["source"].strip()
        name = source.removeprefix("source_documents/")

        url = NAME_TO_URL[name]
        doc_name = url.split("/")[-1]

        title = f"{score:.2%} {doc_name}"

        if doc_name.split(".")[-1] == "pdf":
            # Page numbers start at 0 internally, but 1 in links
            page = int(doc.metadata['page']) + 1
            url += f"#page={page}"
            title += f" - page {page}"

        print(doc)

        embed = discord.Embed(title=title, url=url, description=doc.page_content)
        embeds.append(embed)

    await interaction.followup.send(f"> {question}", embeds=embeds)


client.run(DISCORD_TOKEN)
