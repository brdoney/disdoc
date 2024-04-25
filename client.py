import discord
from discord import app_commands

from typing_extensions import override


def disable_children(view: discord.ui.View):
    for child in view.children:
        if isinstance(child, (discord.ui.Button, discord.ui.Select)):
            child.disabled = True


class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents, activity: discord.Activity):
        super().__init__(intents=intents, activity=activity)
        self.tree = app_commands.CommandTree(self)
        self.sent_views: list[tuple[discord.ui.View, discord.Message]] = []

    @override
    async def setup_hook(self):
        synced = await self.tree.sync()
        print(f"Synced {len(synced)} command(s)")

    async def on_ready(self):
        print("Running bot!")
        if self.user is None:
            print("Not logged in. An error must have occurred")
        else:
            print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")

    def track_view(self, view: discord.ui.View, message: discord.Message) -> None:
        self.sent_views.append((view, message))

    async def disable_tracked_views(self):
        for view, message in self.sent_views:
            disable_children(view)
            _ = await message.edit(view=view)
        print("Disabled all")
