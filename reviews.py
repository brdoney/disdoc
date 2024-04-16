import sqlite3
from abc import ABC, abstractmethod
from enum import Enum, auto

import discord
from discord import SelectOption
from typing_extensions import Self, override

from consent import check_consent


class PostType(Enum):
    RETRIEVAL = auto()
    LLM = auto()


_HELPFULNESS_OPTIONS = [
    SelectOption(label="Very unhelpful", value="-2"),
    SelectOption(label="Slightly unhelpful", value="-1"),
    SelectOption(label="Neutral", value="0"),
    SelectOption(label="Slightly helpful", value="1"),
    SelectOption(label="Very helpful", value="2"),
]
_CORRECTNESS_OPTIONS = [
    SelectOption(label="Completely incorrect", value="-3"),
    SelectOption(label="Mostly incorrect", value="-2"),
    SelectOption(label="Slightly incorrect", value="-1"),
    SelectOption(label="Neutral", value="0"),
    SelectOption(label="Slightly correct", value="1"),
    SelectOption(label="Mostly correct", value="2"),
    SelectOption(label="Completely correc", value="3"),
]
_RELEVANCE_OPTIONS = [
    SelectOption(label="Highly irrelevant", value="-2"),
    SelectOption(label="Slightly irrelevant", value="-1"),
    SelectOption(label="Neutral", value="0"),
    SelectOption(label="Slightly relevant", value="1"),
    SelectOption(label="Highly relevant", value="2"),
]


class ReviewView(discord.ui.View, ABC):
    @abstractmethod
    def is_review_complete(self) -> bool:
        """Whether the current review is complete. Called to perform form validation."""
        raise NotImplementedError()

    @abstractmethod
    def log_review(self) -> None:
        """Log the results of the review. Called when view is complete."""
        raise NotImplementedError()

    async def submit_if_complete(self, interaction: discord.Interaction) -> None:
        """Submits and disable the review if the form is detected as complete."""
        if self.is_review_complete():
            # Disable the form
            for child in self.children:
                if isinstance(child, (discord.ui.Button, discord.ui.Select)):
                    child.disabled = True
            await interaction.response.edit_message(
                content="Review Complete!", view=self
            )
            self.log_review()
        else:
            # Mark the form as incomplete
            await interaction.response.edit_message(
                content="Review is incomplete! Did you fill in all fields before submitting?",
                view=self,
            )


class RetrievalReviewView(ReviewView):
    def __init__(self, post_id: str):
        super().__init__()
        # The ID of the post that this is originally for
        self.post_id = post_id
        self.helpfulness: int | None = None
        self.relevance: int | None = None

    @override
    def is_review_complete(self) -> bool:
        return all(val is not None for val in (self.relevance, self.helpfulness))

    @override
    def log_review(self) -> None:
        print(
            f"Logging review for {self.post_id}: {self.helpfulness} helpfulness, {self.relevance} relevance"
        )

    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="How helpful was the response?",
        options=_HELPFULNESS_OPTIONS,
        max_values=1,
        min_values=1,
    )
    async def helpfulness_callback(
        self, interaction: discord.Interaction, select: discord.ui.Select[Self]
    ):
        self.helpfulness = int(select.values[0])
        await interaction.response.defer()

    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="How relevant was the response?",
        options=_RELEVANCE_OPTIONS,
        max_values=1,
        min_values=1,
    )
    async def relevance_callback(
        self, interaction: discord.Interaction, select: discord.ui.Select[Self]
    ):
        self.relevance = int(select.values[0])
        await interaction.response.defer()

    @discord.ui.button(label="Submit", style=discord.ButtonStyle.primary)
    async def submit_callback(
        self, interaction: discord.Interaction, _: discord.ui.Button[Self]
    ):
        await self.submit_if_complete(interaction)


class LLMReviewView(ReviewView):
    def __init__(self, post_id: str):
        super().__init__()
        # The ID of the post that this is originally for
        self.post_id = post_id
        self.helpfulness: int | None = None
        self.correctness: int | None = None
        self.relevance: int | None = None

    @override
    def is_review_complete(self) -> bool:
        return all(
            val is not None
            for val in (self.relevance, self.helpfulness, self.correctness)
        )

    @override
    def log_review(self) -> None:
        print(
            f"Logging review for {self.post_id}: {self.helpfulness} helpfulness, {self.relevance} relevance, {self.correctness} correctness"
        )

    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="How helpful was the response?",
        options=_HELPFULNESS_OPTIONS,
        max_values=1,
        min_values=1,
    )
    async def helpfulness_callback(
        self, interaction: discord.Interaction, select: discord.ui.Select[Self]
    ):
        self.helpfulness = int(select.values[0])
        await interaction.response.defer()

    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="How relevant was the response?",
        options=_RELEVANCE_OPTIONS,
        max_values=1,
        min_values=1,
    )
    async def relevance_callback(
        self, interaction: discord.Interaction, select: discord.ui.Select[Self]
    ):
        self.relevance = int(select.values[0])
        await interaction.response.defer()

    @discord.ui.select(
        cls=discord.ui.Select,
        placeholder="How correct was the response?",
        options=_CORRECTNESS_OPTIONS,
        max_values=1,
        min_values=1,
    )
    async def correctness_callback(
        self, interaction: discord.Interaction, select: discord.ui.Select[Self]
    ):
        self.correctness = int(select.values[0])
        await interaction.response.defer()

    @discord.ui.button(label="Submit", style=discord.ButtonStyle.primary)
    async def submit_callback(
        self, interaction: discord.Interaction, _: discord.ui.Button[Self]
    ):
        await self.submit_if_complete(interaction)


class ReviewButtonView(discord.ui.View):
    def __init__(
        self, post_type: PostType, post_id: str, sqlite_cursor: sqlite3.Cursor
    ):
        super().__init__()
        self.post_type = post_type
        # Interaction for buttons will be different than our original thread's
        # so we need to save the original one for logging purposes
        self.post_id = post_id
        self.sqlite_cursor = sqlite_cursor

    @discord.ui.button(label="Start Review", style=discord.ButtonStyle.primary)
    async def start_review(
        self, interaction: discord.Interaction, _: discord.ui.Button[Self]
    ):
        if check_consent(self.sqlite_cursor, interaction.user.id) is None:
            await interaction.response.send_message(
                "You have not indicated your consent status. "
                + "Please do so with the `/consent` command before using any other commands.",
                ephemeral=True,
            )
            return

        if self.post_type is PostType.RETRIEVAL:
            view = RetrievalReviewView(self.post_id)
        elif self.post_type is PostType.LLM:
            view = LLMReviewView(self.post_id)
        else:
            raise ValueError(f"Unsupported PostType {self.post_type}")

        # TODO: Create review view for the clicked post and send it
        await interaction.response.send_message(
            "Review in progress...", view=view, ephemeral=True
        )
