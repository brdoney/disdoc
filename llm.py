import asyncio
import json
import re
from collections.abc import AsyncIterable
from enum import Enum, auto
from timeit import default_timer as timer
from typing import Any  # type: ignore[reportAny]

import aiohttp
from langchain import PromptTemplate
from langchain.docstore.document import Document
from openai import AsyncOpenAI

from env_var import LLAMA_API_URL

prompt = PromptTemplate.from_file("./prompt_template.txt", ["question", "context"])
llama_defaults = {
    "stream": True,
    "n_predict": 500,
    "temperature": 0,
    "stop": ["</s>"],
}
openai_defaults: dict[str, Any] = dict(
    model="gpt-3.5-turbo",
    max_tokens=500,
)


def _get_prompt(question: str, context_docs: list[tuple[Document, float]]) -> str:
    context = "\n".join(doc.page_content for (doc, _) in context_docs)
    return prompt.format(question=question, context=context)


async def llama(
    question: str, context_docs: list[tuple[Document, float]]
) -> AsyncIterable[str]:
    params = llama_defaults | {"prompt": _get_prompt(question, context_docs)}

    headers = {
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    llamacpp_regex = re.compile(r"(\S+):\s(.*)$")
    leftover: str = ""  # Buffer for partially read lines
    content: str = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{LLAMA_API_URL}/completion", json=params, headers=headers
        ) as resp:
            async for result in resp.content:
                # Add any leftovers to new chunk
                text = leftover + result.decode()

                # Check if last character is a line break
                ends_with_newline = text[-1] == "\n"

                # Split into lines
                lines = text.split("\n")

                # If text doesn't end in \n, it's incomplete (so store it as leftover)
                if not ends_with_newline:
                    leftover = lines.pop()
                else:
                    # We don't have any leftover data, so reset
                    leftover = ""

                cont = True
                for line in lines:
                    # Parse events (e.g. "data" and "error") and their content
                    m = llamacpp_regex.match(line)
                    if m is not None:
                        event = m.group(1)
                        data = m.group(2)

                        if event == "data":
                            res_data: dict[str, Any] = json.loads(data)
                            new_content: str = res_data["content"]
                            if not content:
                                # Don't add an empty token as the first token
                                if nc := new_content.lstrip():
                                    content += nc
                                    yield nc
                            else:
                                # We added a token, so yield
                                content += new_content
                                yield content

                            # If server sent a stop token, it'll mark "stop" as True and we should break
                            if res_data.get("stop", False):
                                # Could potentially save generation settings here if we wanted
                                cont = False
                                break
                        elif event == "error":
                            try:
                                res_error: dict[str, Any] = json.loads(data)
                                if "slot unavailable" in res_error["message"]:
                                    raise RuntimeError("slot unavailable")
                                else:
                                    err_code: int = res_error["error"]["code"]
                                    err_type: str = res_error["error"]["type"]
                                    err_message: str = res_error["error"]["message"]
                                    print(
                                        f"llama.cpp error [{err_code} - {err_type}]: {err_message}"
                                    )
                            except json.JSONDecodeError:
                                print(f"llama.cpp error {data}")
                if not cont:
                    break


client = AsyncOpenAI()


async def openai(
    question: str, context_docs: list[tuple[Document, float]]
) -> AsyncIterable[str]:
    stream = await client.chat.completions.create(
        stream=True,
        messages=[{"role": "user", "content": _get_prompt(question, context_docs)}],
        **openai_defaults,  # type: ignore[reportAny]
    )
    content = ""
    async for chunk in stream:
        new_content = chunk.choices[0].delta.content or ""
        if not content:
            if nc := new_content.rstrip():
                content += nc
                yield content
        else:
            content += new_content
            yield content


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
    LLAMA = auto()
    OPENAI = auto()
    MOCK = auto()

    def __call__(self, question: str, context_docs: list[tuple[Document, float]]):
        if self is LLMType.LLAMA:
            return llama(question, context_docs)
        elif self is LLMType.OPENAI:
            return openai(question, context_docs)
        elif self is LLMType.MOCK:
            return mock(question, context_docs)
        else:
            raise ValueError(f"Unsupported LLMType: {self.name}")
