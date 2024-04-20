import asyncio
from collections.abc import AsyncIterable
from enum import Enum, auto

from langchain import PromptTemplate
from langchain.docstore.document import Document

prompt = PromptTemplate.from_file("./prompt_template.txt", ["question", "context"])


async def mock(*_) -> AsyncIterable[str]:
    """Mock LLM that yields words from a predefined message. Accepts arguments to be
    interchangeable with other LLMs, but doesn't use them."""

    mock_message = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc eu tellus lorem. Nulla ornare felis eleifend consequat tempus. Donec quam libero, dapibus ac dolor ac, bibendum aliquam neque. Aliquam erat volutpat. Quisque erat leo, dictum imperdiet neque non, vulputate lacinia diam. Fusce tincidunt urna et nibh placerat varius. Donec semper arcu et finibus ornare."""
    content = ""
    for word in mock_message.split():
        await asyncio.sleep(0.02)
        content += f"{word} "
        yield content


class LLMType(Enum):
    MOCK = auto()

    def __call__(self, question: str, context_docs: list[tuple[Document, float]]):
        if self is LLMType.MOCK:
            return mock(question, context_docs)
        else:
            raise ValueError(f"Unsupported LLMType: {self.name}")
