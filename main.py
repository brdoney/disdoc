import json
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Any  # type: ignore[reportAny]
from urllib.parse import ParseResult, parse_qsl, urlencode, urlparse

import discord
from dataclasses_json import DataClassJsonMixin
from discord import app_commands
from discord.ext import commands
from typing_extensions import override

from categories import DocGroup
from chroma import create_chroma_client, create_chroma_collection, create_embeddings
from env_var import (
    CLICK_URL,
    CONSENT_URL,
    DISCORD_TOKEN,
    MAPPINGS_PATH,
    SOURCE_DIRECTORY,
)
from llm import LLMType
from pdf_images import load_image_cache, pdf_image, save_image_cache
from reviews import ReviewButtonView, ReviewType
from sqlite_db import check_consent, get_review_count, log_post, log_post_times

with open(MAPPINGS_PATH) as f:
    NAME_TO_URL: dict[str, str] = json.load(f)

embeddings = create_embeddings()
chroma_client = create_chroma_client()
docs_db = create_chroma_collection(chroma_client, embeddings)

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


def get_click_url(dest: ParseResult, post_id: int | None, rec_id: int) -> ParseResult:
    # Add rec_id to prevent duplicate links (separate snippets in the same file) from grouping together
    params = {"to": dest.geturl(), "rec_id": rec_id}
    if post_id is not None:
        params["postid"] = post_id

    return add_query_params(CLICK_URL, params)


@dataclass
class EmbedRecord(DataClassJsonMixin):
    title: str
    content: str
    dest: str
    score: float
    image: str | None = None


@dataclass
class AskRecord(DataClassJsonMixin):
    post_id: int
    embeds: list[EmbedRecord]
    answer: str | None = None


@app_commands.guild_only()
@client.tree.command(description="Ask for documents related to your question")
@app_commands.describe(answer="Whether to generate an answer using an LLM")
@app_commands.describe(category="The category this question is about")
@app_commands.describe(question="Question or statement you're interested in")
async def ask(
    interaction: discord.Interaction,
    answer: bool,
    category: DocGroup,
    question: str,
):
    user = check_consent(interaction.user.id)
    if user is None:
        await interaction.response.send_message(
            "You have not indicated your consent status. "
            + "Please do so with the `/consent` command before using any other commands.",
            ephemeral=True,
        )
        return

    # Log post in DB
    post_id = log_post(interaction.id, user, answer, llm_type)

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

            url = get_click_url(dest_url, post_id, i).geturl()

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

            url = get_click_url(dest_url, post_id, i).geturl()
            embed = discord.Embed(title=title, url=url, description=desc)

        image_path_str = str(image_path) if image_path is not None else None
        embed_records.append(
            EmbedRecord(title, desc, dest_url.geturl(), score, image_path_str)
        )

        embeds.append(embed)

    if new_images:
        save_image_cache()

    retrieval_review = ReviewButtonView(ReviewType.RETRIEVAL, post_id)

    # We use followup since we deferred earlier
    await interaction.followup.send(
        f"> {question}",
        files=files,
        embeds=embeds,
        view=retrieval_review,
    )

    # Don't continue if we're not generating the answer with an LLM
    generation_time = None
    answer_text = None
    if answer:
        # Send a message to mark that we're generating the answer
        msg = await interaction.followup.send("Generating answer...", wait=True)

        start = timer()
        elapsed = 0
        # before = timer()
        line = None

        start = timer()
        # The Discord edit usually throttles after 3-4 llama.cpp tokens (with delays as high as 2s)
        async for line in llm_type(question, docs):
            curr_time = timer()
            elapsed = curr_time - start
            if elapsed < edit_timer:
                continue
            start = curr_time

            # after = timer()
            # print(f"llama.cpp token: {after - before}")
            # before = timer()
            msg = await msg.edit(content=line)
            # after = timer()
            # print(f"edit: {after - before}")
            # before = timer()
        end = timer()
        generation_time = end - start
        print(f"Generation: {generation_time}s")

        # We have the full answer now
        answer_text = line

        llm_review = ReviewButtonView(ReviewType.LLM, post_id)

        # Just in case last necessary edit didn't go through due to timeout
        # Also take the time to add a review button
        msg = await msg.edit(content=answer_text, view=llm_review)

    if post_id is not None:
        # Only log info if we have a post on the books (i.e. consent was given)

        log_post_times(post_id, retrieval_time, generation_time)

        record = AskRecord(post_id, embed_records, answer_text)
        with (RECORDS_DIR / f"post_{post_id}.json").open("w") as f:
            s = record.to_json(indent=2)  # type: ignore[reportUnknownMemberType]
            _ = f.write(s)


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
    url = new_url.geturl()

    await interaction.response.send_message(
        embed=discord.Embed(
            title="Consent form",
            description="Use this form to check and/or indicate your consent status",
            url=url,
        ),
        ephemeral=True,
    )


@client.tree.command(description="Tells you how many reviews you've given")
async def reviewcount(interaction: discord.Interaction) -> None:
    user = check_consent(interaction.user.id)
    if user is None:
        await interaction.response.send_message(
            "You have not indicated your consent status. "
            + "Please do so with the `/consent` command before using any other commands.",
            ephemeral=True,
        )
    else:
        num_reviews = get_review_count(user)
        await interaction.response.send_message(
            f"You have given {num_reviews} reviews",
            ephemeral=True,
        )


@client.tree.command(description="Admin only: get or set the edit buffer")
@app_commands.checks.has_permissions(administrator=True)
async def editbuffer(interaction: discord.Interaction, new_timer: float | None = None):
    global edit_timer
    if new_timer is None:
        await interaction.response.send_message(
            f"Edit buffer is currently {edit_timer} seconds", ephemeral=True
        )
    else:
        edit_timer = new_timer
        await interaction.response.send_message(
            f"Set edit buffer to {new_timer} seconds", ephemeral=True
        )


@client.tree.command(description="Admin only: get or set the LLM type")
@app_commands.checks.has_permissions(administrator=True)
async def llmtype(interaction: discord.Interaction, new_llm: LLMType | None = None):
    global llm_type
    if new_llm is None:
        await interaction.response.send_message(
            f"LLM type is currently {llm_type.name}", ephemeral=True
        )
    else:
        llm_type = new_llm
        await interaction.response.send_message(
            f"Set LLM type to {new_llm.name}", ephemeral=True
        )


@client.tree.command(description="Admin only: shutdown the bot")
@app_commands.checks.has_permissions(administrator=True)
async def shutdown(interaction: discord.Interaction):
    await interaction.response.send_message("Shutting down bot", ephemeral=True)
    await interaction.client.close()
    print("Shutdown bot")


@client.event
async def on_command_error(_: commands.Context, error: Exception):  # type: ignore[reportMissingTypeArgument]
    if isinstance(error, discord.HTTPException):
        print("Hit ratelimit:", error)
    else:
        print("Unknown error:", error)


client.run(DISCORD_TOKEN)
