import json
from timeit import default_timer as timer
from typing import cast
from urllib.parse import ParseResult, parse_qsl, urlencode, urlparse

import chromadb
import chromadb.config
import discord
from categories import AskCategory
from discord import app_commands
from dotenv import load_dotenv
from env_var import load_env
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pdf_images import load_image_cache, pdf_image, save_image_cache
from typing_extensions import override

_ = load_dotenv()

DISCORD_TOKEN = load_env("DISCORD_TOKEN")
PERSIST_DIRECTORY = load_env("PERSIST_DIRECTORY")
EMBEDDINGS_MODEL_NAME = load_env("EMBEDDINGS_MODEL_NAME")
MAPPINGS_PATH = load_env("MAPPINGS_PATH")
SIMILARITY_METRIC = load_env("SIMILARITY_METRIC", choices=["cosine", "l2", "ip"])

with open(MAPPINGS_PATH) as f:
    NAME_TO_URL: dict[str, str] = json.load(f)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
CHROMA_SETTINGS = chromadb.config.Settings(
    persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

chroma_client = chromadb.PersistentClient(
    settings=CHROMA_SETTINGS, path=PERSIST_DIRECTORY
)
docs_db = Chroma(
    collection_name="docs",
    client=chroma_client,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
    client_settings=CHROMA_SETTINGS,
    collection_metadata={"hnsw:space": SIMILARITY_METRIC},
)
tests_db = Chroma(
    collection_name="tests",
    client=chroma_client,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
    client_settings=CHROMA_SETTINGS,
    collection_metadata={"hnsw:space": SIMILARITY_METRIC},
)

intents = discord.Intents.default()
intents.message_content = True


class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    @override
    async def setup_hook(self):
        synced = await self.tree.sync()
        print(f"Synced {len(synced)} command(s)")


client = MyClient(intents=intents)


@client.event
async def on_ready():
    print("Running Chroma Bot!")
    if client.user is None:
        print("Not logged in. An error must have occurred")
    else:
        print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("------")


SOURCE_CODE_EXT = {
    # .html not used b/c it's useless
    # .pdf not used b/c we generate images
    ".output": "text",
    ".java": "java",
    ".sh": "bash",
    ".txt": "text",
    ".c": "c",
    ".h": "cpp",
    ".cc": "cpp",
    ".supp": "text",
    ".cpp": "cpp",
    ".md": "md",
    ".py": "python",
    ".js": "js",
    ".s": "x86asm",
    ".suppression": "text",
    ".l": "c",
    ".tst": "text",
    ".y": "c",
    ".rep": "text",
    ".3": "text",
    "makefile": "makefile",
}


def add_query_param(url: ParseResult, name: str, value: str) -> ParseResult:
    """Create a new URL from `url`, with the name and value pair added to the
    query params.

    If a query param with the given name already exists, this
    function adds to it (so there will be multiple entries for it, the
    original(s) and new one) instead of overwriting its existing value.

    Args:
        url: the url to copy
        name: the name of the query parameter
        value: the value to add

    Returns:
        a copy of the given url with the new name, value pair added to the query params
    """
    query = parse_qsl(url.query)
    query.append((name, value))
    return url._replace(query=urlencode(query))


@client.tree.command(description="Ask for documents related to your question")
@app_commands.describe(
    question="Question or statement you want to find documents regarding"
)
async def ask(
    interaction: discord.Interaction,
    category: AskCategory,
    question: str,
):
    await interaction.response.defer()

    start = timer()
    if category == AskCategory.tests:
        docs = await tests_db.asimilarity_search_with_relevance_scores(question)
    elif category == AskCategory.midterm:
        docs = await tests_db.asimilarity_search_with_relevance_scores(
            question, filter={"type": "midterm"}
        )
    elif category == AskCategory.final:
        docs = await tests_db.asimilarity_search_with_relevance_scores(
            question, filter={"type": "final"}
        )
    else:
        docs = await docs_db.asimilarity_search_with_relevance_scores(question)
    # docs = await db.amax_marginal_relevance_search(question)
    end = timer()
    print(f"{end - start}s")

    embeds: list[discord.Embed] = []
    files: list[discord.File] = []
    for i, (doc, score) in enumerate(docs):
        source = cast(str, doc.metadata["source"]).strip()  # type: ignore[reportUnknownMemberType]

        url_str = NAME_TO_URL[source.removeprefix("source_documents/")]
        url = add_query_param(urlparse(url_str), "rec_id", str(i))

        doc_name = url.path.split("/")[-1]
        # Note: score is only accurate if we're using cosine as our similarity metric
        title = f"{score:.2%} {doc_name}"
        # title = f"{doc_name}"

        desc = doc.page_content

        # To make things consistent b/t ingest.py loaders and source_code_ext keys
        # Note: if a file is "a.tar.gz", ext would just be ".gz" and if there's no extension,
        # it would be the name of the doc, like "Makefile"
        ext = doc_name[doc_name.rindex(".") :] if "." in doc_name else doc_name
        ext = ext.lower()

        if ext == ".pdf":
            # Page numbers start at 0 internally, but 1 in links
            page = int(doc.metadata["page"]) + 1  # type: ignore[reportUnknownMemberType]
            url = url._replace(fragment=f"page={page}")
            title += f" - page {page}"

            image_url = pdf_image(source, doc)
            if image_url is not None:
                file = discord.File(image_url)
                files.append(file)
                embed = discord.Embed(title=title, url=url.geturl()).set_image(
                    url=f"attachment://{file.filename}"
                )
            else:
                embed = discord.Embed(title=title, url=url.geturl(), description=desc)
        else:
            if ext in SOURCE_CODE_EXT:
                desc = f"```{SOURCE_CODE_EXT[ext]}\n{desc}\n```"

            embed = discord.Embed(title=title, url=url.geturl(), description=desc)

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
