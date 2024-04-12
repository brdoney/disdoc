import json
import sqlite3
from pathlib import Path
from timeit import default_timer as timer
from typing import Any  # type: ignore[reportAny]
from urllib.parse import ParseResult, parse_qsl, urlencode, urlparse, urlunparse

import discord
from categories import DocGroup
from chroma import create_chroma_client, create_chroma_collection, create_embeddings
from consent import check_consent
from discord import app_commands
from env_var import (
    CONSENT_URL,
    DISCORD_TOKEN,
    MAPPINGS_PATH,
    SOURCE_DIRECTORY,
    SQLITE_DB,
)
from pdf_images import load_image_cache, pdf_image, save_image_cache
from typing_extensions import override

with open(MAPPINGS_PATH) as f:
    NAME_TO_URL: dict[str, str] = json.load(f)

embeddings = create_embeddings()
chroma_client = create_chroma_client()
docs_db = create_chroma_collection(chroma_client, embeddings)

sqlite_cursor = sqlite3.connect(SQLITE_DB).cursor()

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


@client.tree.command(description="Ask for documents related to your question")
@app_commands.describe(
    question="Question or statement you want to find documents regarding"
)
async def ask(
    interaction: discord.Interaction,
    category: DocGroup,
    question: str,
):
    if check_consent(sqlite_cursor, interaction.user.id) is None:
        await interaction.response.send_message(
            "You have not indicated your consent status. "
            + "Please do so with the `/consent` command before using any other commands."
        )
        return

    # Defer b/c loading vector DB from scratch takes longer than discord's response timeout
    await interaction.response.defer()

    start = timer()
    # docs = await docs_db.asimilarity_search_with_relevance_scores(
    #     question, filter=category.get_filter()
    # )
    docs = await docs_db.asimilarity_search_with_relevance_scores(
        question, filter=category.get_filter()
    )
    # docs = await db.amax_marginal_relevance_search(question)
    end = timer()
    print(f"{end - start}s")

    embeds: list[discord.Embed] = []
    files: list[discord.File] = []
    for i, (doc, score) in enumerate(docs):
        source = Path(doc.metadata["source"])  # type: ignore[reportUnknownMemberType]

        url_str = NAME_TO_URL[str(remove_group(source))]
        # Add rec_id to prevent duplicate links (separate snippets in the same file) from grouping together
        url = add_query_params(urlparse(url_str), {"rec_id": i})

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

    # We use followup since we deferred earlier
    await interaction.followup.send(
        f"> {question}",
        files=files,
        embeds=embeds,
    )


@client.tree.command(
    description="Generates a link you can use to indicate your consent status"
)
async def consent(interaction: discord.Interaction) -> None:
    new_url = urlparse(CONSENT_URL)

    # https://courses.cs.vt.edu/cs3214/test/consent.html?discordId=hello
    user = interaction.user

    params = {
        "discordId": user.id,
        "discordName": user.name,
        "discordNick": user.display_name,
    }
    new_url = add_query_params(new_url, params)
    url = urlunparse(new_url)

    await interaction.response.send_message(
        embed=discord.Embed(
            title="Consent form",
            description="Use this form to check and/or indicate your consent status",
            url=url,
        ),
        ephemeral=True,
    )


DocGroup.check_members()

# Load the docment -> image translations from memory
load_image_cache()

try:
    client.run(DISCORD_TOKEN)
finally:
    # We've loaded and maybe modified the image cache, so we need to save it
    print("Saving pdf image cache")
    save_image_cache()
