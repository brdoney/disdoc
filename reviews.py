from abc import ABC, abstractmethod
from enum import Enum, auto

import discord
from discord import SelectOption
from typing_extensions import Self, override

from client import MyClient, disable_children
from sqlite_db import UserInfo, check_consent, log_llm_review, log_retrieval_review


class ReviewType(Enum):
    RETRIEVAL = auto()
    LLM = auto()

    @staticmethod
    def from_str(s: str) -> "ReviewType":
        return ReviewType[s]


_HELPFULNESS_OPTIONS = [
    SelectOption(label="Very helpful", value="2"),
    SelectOption(label="Slightly helpful", value="1"),
    SelectOption(label="Neutral", value="0"),
    SelectOption(label="Slightly unhelpful", value="-1"),
    SelectOption(label="Very unhelpful", value="-2"),
]
_CORRECTNESS_OPTIONS = [
    SelectOption(label="Completely correct", value="3"),
    SelectOption(label="Mostly correct", value="2"),
    SelectOption(label="Slightly correct", value="1"),
    SelectOption(label="Neutral", value="0"),
    SelectOption(label="Slightly incorrect", value="-1"),
    SelectOption(label="Mostly incorrect", value="-2"),
    SelectOption(label="Completely incorrect", value="-3"),
]
_RELEVANCE_OPTIONS = [
    SelectOption(label="Highly relevant", value="2"),
    SelectOption(label="Slightly relevant", value="1"),
    SelectOption(label="Neutral", value="0"),
    SelectOption(label="Slightly irrelevant", value="-1"),
    SelectOption(label="Highly irrelevant", value="-2"),
]


class ReviewView(discord.ui.View, ABC):
    def __init__(self, post_id: int | None, user: UserInfo, completed_users: set[int]):
        super().__init__(timeout=None)
        # The ID of the post that this is originally for
        self.post_id = post_id
        # The user who is using this UI
        self.user = user
        # The users who have completed a review of the associated post
        self.completed_users = completed_users

    @abstractmethod
    def is_review_complete(self) -> bool:
        """Whether the current review is complete. Called to perform form validation."""
        raise NotImplementedError()

    @abstractmethod
    def log_review(self, resubmitting: bool) -> None:
        """Log the results of the review. Called when view is complete."""
        raise NotImplementedError()

    async def submit_if_complete(self, interaction: discord.Interaction) -> None:
        """Submits and disable the review if the form is detected as complete."""
        if self.is_review_complete():
            # Disable the form
            disable_children(self)

            await interaction.response.edit_message(
                content="Review Complete!", view=self
            )
            # May or may not save content of review, depending on consent status
            # The SQL functions that subclasses use should take care of this check for us
            discord_id = interaction.user.id
            self.log_review(discord_id in self.completed_users)
            self.completed_users.add(discord_id)
        else:
            # Mark the form as incomplete
            await interaction.response.edit_message(
                content="Review is incomplete! Did you fill in all fields before submitting?",
                view=self,
            )


class RetrievalReviewView(ReviewView):
    def __init__(self, post_id: int | None, user: UserInfo, completed_users: set[int]):
        super().__init__(post_id, user, completed_users)
        self.helpfulness: int | None = None
        self.relevance: int | None = None

    @override
    def is_review_complete(self) -> bool:
        return all(val is not None for val in (self.relevance, self.helpfulness))

    @override
    def log_review(self, resubmitting: bool) -> None:
        assert self.is_review_complete(), "Trying to log unfinished retrieval review"
        log_retrieval_review(
            self.post_id,
            self.user,
            resubmitting,
            self.relevance,  # type: ignore[reportArgumentType]
            self.helpfulness,  # type: ignore[reportArgumentType]
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
    def __init__(self, post_id: int | None, user: UserInfo, completed_users: set[int]):
        super().__init__(post_id, user, completed_users)
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
    def log_review(self, resubmitting: bool) -> None:
        assert self.is_review_complete(), "Trying to log unfinished LLM review"
        log_llm_review(
            self.post_id,
            self.user,
            resubmitting,
            self.relevance,  # type: ignore[reportArgumentType]
            self.helpfulness,  # type: ignore[reportArgumentType]
            self.correctness,  # type: ignore[reportArgumentType]
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
    def __init__(self, post_type: ReviewType, post_id: int | None):
        super().__init__(timeout=None)
        self.post_type = post_type
        """What kind of post this button view is attached to"""
        self.post_id = post_id
        """DB ID for the post this is attached to"""
        self.completed_users: set[int] = set()
        """Set of Discord IDs of users who have finished reviews, so we don't award double credit"""

    @discord.ui.button(label="Start Review", style=discord.ButtonStyle.primary)
    async def start_review(
        self, interaction: discord.Interaction, _: discord.ui.Button[Self]
    ):
        user = check_consent(interaction.user.id)
        if user is None:
            await interaction.response.send_message(
                "You have not indicated your consent status. "
                + "Please do so with the `/consent` command before using any other commands.",
                ephemeral=True,
            )
            return

        if self.post_type is ReviewType.RETRIEVAL:
            view = RetrievalReviewView(self.post_id, user, self.completed_users)
        elif self.post_type is ReviewType.LLM:
            view = LLMReviewView(self.post_id, user, self.completed_users)
        else:
            raise ValueError(f"Unsupported PostType {self.post_type}")

        if interaction.user.id not in self.completed_users:
            message = "Review in progress..."
        else:
            message = "Editing past review..."

        await interaction.response.send_message(message, view=view, ephemeral=True)

        client = interaction.client
        if isinstance(client, MyClient):
            client.track_view(view, await interaction.original_response())
