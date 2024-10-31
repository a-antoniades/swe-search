import enum
import logging
import random
import string

import instructor

from moatless.schema import Completion
from moatless.utils.tokenizer import count_tokens


logger = logging.getLogger(__name__)

class LLMResponseFormat(enum.Enum):
    TOOLS = "tool_call"
    ANTHROPIC_TOOLS = "anthropic_tools"
    JSON = "json_mode"
    STRUCTURED_OUTPUT = "structured_output"
    TEXT = "text"


def input_messages(
        content: str,
        completion: Completion | None,
        feedback: str | None = None):
    messages = []
    tool_call_id = None

    if completion:
        messages = completion.input

        response_message = completion.response["choices"][0]["message"]
        if response_message.get("tool_calls"):
            tool_call_id = response_message.get("tool_calls")[0]["id"]
            last_response = {
                "role": response_message["role"],
                "tool_calls": response_message["tool_calls"]
            }
        else:
            last_response = {
                "role": response_message["role"],
                "content": response_message["content"]
            }
        messages.append(last_response)

        if response_message.get("tool_calls"):
            tool_call_id = response_message.get("tool_calls")[0]["id"]

    if tool_call_id:
        new_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
    else:
        new_message = {
            "role": "user",
            "content": content,
        }

    if feedback:
        new_message["content"] += "\n\n" + feedback

    messages.append(new_message)
    return messages


def input_messages_anthropic(
        content: str,
        completion: Completion | None,
        feedback: str | None = None):
    messages = []
    tool_call_id = None

    if completion:
        messages = completion.input
        for message in messages:
            remove_cache_control(message)
            if message["role"] == "user" and len(message["content"]) == 2 and "<feedback>" in message["content"][1].get("text"):
                message["content"] = message["content"][:1]

        apply_cache_control(messages[-1])

        last_response = {
            "role": completion.response["role"],
            "content": completion.response["content"]
        }
        messages.append(last_response)

        for block in completion.response["content"]:
            if block["type"] == "tool_use":
                tool_call_id = block["id"]
                break

    new_message = {
        "role": "user",
        "content": [
        ],
    }

    if tool_call_id:
        new_message["content"].append(
            {
                "tool_use_id": tool_call_id,
                "content": content,
                "type": "tool_result",
                "cache_control": {"type": "ephemeral"}
            }
        )
    else:
        new_message["content"].append(
            {
                "text": content,
                "type": "text",
                "cache_control": {"type": "ephemeral"}
            }
        )

    if feedback:
        new_message["content"].append(
            {
                "text": f"<feedback>\n{feedback}\n</feedback>",
                "type": "text"
            }
        )

    messages.append(new_message)
    return messages


def apply_cache_control(message: dict):
    content = message["content"]
    if type(content) is str:
        content = dict(
            type="text",
            text=content,
        )
        message["content"] = [content]
    else:
        content = message["content"][-1]

    content["cache_control"] = {"type": "ephemeral"}


def remove_cache_control(message: dict):
    content = message["content"]
    if type(content) is str:
        return message

    for content in message["content"]:
        content.pop("cache_control", None)

    return message



def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars
    
    return random_string


