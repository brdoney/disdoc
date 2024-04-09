import enum
import json
from timeit import default_timer as timer
from typing import cast
from urllib.parse import ParseResult, parse_qsl, urlencode, urlparse

import chromadb
import chromadb.config
import discord
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

# class FileEnumMeta(type(enum.Enum)):
#     def __new__(metacls, cls: str, bases: tuple[type, ...], classdict: enum._EnumDict):
#         filename = classdict.get('_filename', None)  # type: ignore[reportAny]
#         if filename is not None and isinstance(filename, str):
#             with open(filename, 'r') as file:
#                 lines = file.readlines()
#                 for line in lines:
#                     if line in classdict:
#                         continue
#                     member_name = line.strip()
#                     enum_member = metacls(member_name, bases, classdict)
#                     classdict[member_name] = enum_member
#         return super().__new__(metacls, cls, bases, classdict)
#
# class FileEnum(enum.Enum, metaclass=FileEnumMeta):
#     _filename = None
#
#     def __init__(self, *args: Any):  # type: ignore[reportAny]
#         super().__init__()
#         if args:
#             self._value_ = args[0]
#
#     @override
#     def __str__(self):
#         return self.name
#
# class AskCategoryFile(FileEnum):  # type: ignore[generalTypeIssues]
#     _filename = "categories.txt"
#
# print(AskCategoryFile)


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


with open("./categories.txt", "r") as f:
    AskCategory = enum.Enum("AskCategory", [line.strip() for line in f.readlines()])


@client.tree.command(description="Ask for documents related to your question")
@app_commands.describe(
    question="Question or statement you want to find documents regarding"
)
async def ask(
    interaction: discord.Interaction,
    category: AskCategory,  # type: ignore[reportInvalidTypeForm]
    question: str,
):
    await interaction.response.defer()

    start = timer()
    if category == AskCategory.tests:  # type: ignore[reportUnknownMemberType]
        docs = await tests_db.asimilarity_search_with_relevance_scores(question)
    elif category == AskCategory.midterm:  # type: ignore[reportUnknownMemberType]
        docs = await tests_db.asimilarity_search_with_relevance_scores(
            question, filter={"type": "midterm"}
        )
    elif category == AskCategory.final:  # type: ignore[reportUnknownMemberType]
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
        if "." in doc_name:
            ext = "." + doc_name.split(".")[-1]
        else:
            ext = doc_name

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
