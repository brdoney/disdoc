import json
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Any  # type: ignore[reportAny]
from urllib.parse import ParseResult, parse_qsl, urlencode, urlparse
import uuid

import discord
from dataclasses_json import DataClassJsonMixin
from discord import app_commands
from typing_extensions import override

from categories import DocGroup
from chroma import create_chroma_client, create_chroma_collection, create_embeddings
from env_var import DISCORD_TOKEN, MAPPINGS_PATH, SOURCE_DIRECTORY
from llm import LLMType
from pdf_images import load_image_cache, pdf_image, save_image_cache

with open(MAPPINGS_PATH) as f:
    NAME_TO_URL: dict[str, str] = json.load(f)

embeddings = create_embeddings()
chroma_client = create_chroma_client()
docs_db = create_chroma_collection(chroma_client, embeddings, True)

_intents = discord.Intents.default()
_intents.message_content = True
_activity = discord.Activity(type=discord.ActivityType.playing, name="/ask")

RECORDS_DIR = Path("./records").resolve()
"""Directory to store records of posts in"""
# Make the records directory if it doesn't exist already
RECORDS_DIR.mkdir(exist_ok=True)

edit_timer = 0.3
"""Number of tokens to buffer before editing a message. Theoretically, `1/5=0.2` is the minimum value."""

llm_type: LLMType = LLMType.MOCK
"""The LLM we're using"""

# Load the docment -> image translations from memory
load_image_cache()


class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents, activity: discord.Activity):
        super().__init__(intents=intents, activity=activity)
        self.tree = app_commands.CommandTree(self)

    @override
    async def setup_hook(self):
        synced = await self.tree.sync()
        print(f"Synced {len(synced)} command(s)")


client = MyClient(intents=_intents, activity=_activity)


@client.event
async def on_ready():
    print("Running bot!")
    if client.user is None:
        print("Not logged in. An error must have occurred")
    else:
        print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("------")


SOURCE_CODE_EXT = {
    # .html not used b/c we show page's raw text content
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


def add_query_params(url: ParseResult, pairs: dict[str, Any]) -> ParseResult:
    """Create a new URL from `url`, with the given query params pairs added.

    If a query param with the given name already exists, this
    function adds to it (so there will be multiple entries for it, the
    original(s) and new one) instead of overwriting its existing value.

    Args:
        url: the url to copy
        pairs: the query param and value pairs to add

    Returns:
        a copy of the given url with the new name, value pair added to the query params
    """
    query = parse_qsl(url.query)
    query.extend(pairs.items())
    return url._replace(query=urlencode(query))


def remove_group(path: Path) -> Path:
    """Remove everything up to and including the group part of a path.

    Example:

        >>> remove_group(Path("source_documents/ex0/cs3214/sample.txt"))
        "cs3214/sample.txt"
    """
    after_group = path.relative_to(SOURCE_DIRECTORY).parts[1:]
    return Path(*after_group)


def get_click_url(dest: ParseResult, rec_id: int) -> ParseResult:
    # Add rec_id to prevent duplicate links (separate snippets in the same file) from grouping together
    params = {"rec_id": rec_id}
    return add_query_params(dest, params)


@dataclass
class EmbedRecord(DataClassJsonMixin):
    title: str
    content: str
    dest: str
    score: float
    image: str | None = None


@dataclass
class AskRecord(DataClassJsonMixin):
    post_id: str
    embeds: list[EmbedRecord]


@client.tree.command(description="Ask for documents related to your question")
@app_commands.describe(category="The category this question is about")
@app_commands.describe(question="Question or statement you're interested in")
async def ask(
    interaction: discord.Interaction,
    category: DocGroup,
    question: str,
):
    # Log post in DB
    post_id = uuid.uuid4()

    # Defer b/c loading vector DB from scratch takes longer than discord's response timeout
    await interaction.response.defer(thinking=True)

    start = timer()
    docs = await docs_db.asimilarity_search_with_relevance_scores(
        question, filter=category.get_filter()
    )
    end = timer()
    retrieval_time = end - start
    print(f"Retrieval: {retrieval_time}s")

    embed_records: list[EmbedRecord] = []
    embeds: list[discord.Embed] = []
    files: list[discord.File] = []
    new_images = False
    for i, (doc, score) in enumerate(docs):
        source = Path(doc.metadata["source"])  # type: ignore[reportUnknownMemberType]

        dest_url_str = NAME_TO_URL[str(remove_group(source))]
        dest_url = urlparse(dest_url_str)

        doc_name = dest_url.path.split("/")[-1]
        # Note: score is only accurate if we're using cosine as our similarity metric
        title = f"{score:.2%} {doc_name}"
        # title = f"{doc_name}"

        desc = doc.page_content

        # To make things consistent b/t ingest.py loaders and source_code_ext keys
        # Note: if a file is "a.tar.gz", ext would just be ".gz" and if there's no extension,
        # it would be the name of the doc, like "Makefile"
        ext = doc_name[doc_name.rindex(".") :] if "." in doc_name else doc_name
        ext = ext.lower()

        image_path = None

        if ext == ".pdf":
            # Page numbers start at 0 internally, but 1 in links
            page = int(doc.metadata["page"]) + 1  # type: ignore[reportUnknownMemberType]
            dest_url = dest_url._replace(fragment=f"page={page}")
            title += f" - page {page}"

            url = get_click_url(dest_url, i).geturl()

            res = pdf_image(source, doc)
            if res is not None:
                image_path, in_cache = res
                new_images |= not in_cache

                file = discord.File(image_path)
                files.append(file)
                embed = discord.Embed(title=title, url=url).set_image(
                    url=f"attachment://{file.filename}"
                )
            else:
                embed = discord.Embed(title=title, url=url, description=desc)
        else:
            if ext in SOURCE_CODE_EXT:
                desc = f"```{SOURCE_CODE_EXT[ext]}\n{desc}\n```"

            url = get_click_url(dest_url, i).geturl()
            embed = discord.Embed(title=title, url=url, description=desc)

        image_path_str = str(image_path) if image_path is not None else None
        embed_records.append(
            EmbedRecord(title, desc, dest_url.geturl(), score, image_path_str)
        )

        embeds.append(embed)

    if new_images:
        save_image_cache()

    # We use followup since we deferred earlier
    await interaction.followup.send(
        f"> {question}",
        files=files,
        embeds=embeds,
    )

    record = AskRecord(str(post_id), embed_records)
    with (RECORDS_DIR / f"post_{post_id}.json").open("w") as f:
        s = record.to_json(indent=2)  # type: ignore[reportUnknownMemberType]
        _ = f.write(s)


client.run(DISCORD_TOKEN)
