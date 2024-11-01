import json
import logging
from enum import Enum
from typing import Any, Optional, Union, List, Set

import litellm
from instructor import OpenAISchema
from litellm import completion_cost, cost_per_token, NotFoundError

from pydantic import BaseModel, Field

from moatless.settings import Settings

logger = logging.getLogger(__name__)


class FileWithSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="Span IDs identiying the relevant code spans. A span id is a unique identifier for a code sippet. It can be a class name or function name. For functions in classes separete with a dot like 'class.function'.",
    )

    def add_span_id(self, span_id):
        if span_id not in self.span_ids:
            self.span_ids.append(span_id)

    def add_span_ids(self, span_ids: list[str]):
        for span_id in span_ids:
            self.add_span_id(span_id)

    def __eq__(self, other: "FileWithSpans"):
        return self.file_path == other.file_path and self.span_ids == other.span_ids

class Usage(BaseModel):
    completion_cost: float = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0

    @classmethod
    def from_completion_response(
        cls, completion_response: dict | BaseModel, model: str
    ) -> Union["Usage", None]:
        if isinstance(completion_response, BaseModel) and hasattr(
            completion_response, "usage"
        ):
            usage = completion_response.usage.model_dump()
        elif isinstance(completion_response, dict) and "usage" in completion_response:
            usage = completion_response["usage"]
        else:
            logger.warning(
                f"No usage info available in completion response: {completion_response}"
            )
            return None

        logger.info(f"Usage: {json.dumps(usage, indent=2)}")

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)

        if usage.get("cache_creation_input_tokens"):
            prompt_tokens += usage["cache_creation_input_tokens"]

        completion_tokens = usage.get("completion_tokens") or usage.get(
            "output_tokens", 0
        )

        try:
            cost = litellm.completion_cost(completion_response=completion_response, model=model)
        except Exception:
            # If cost calculation fails, fall back to calculating it manually
            try:
                prompt_cost, completion_cost = cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                cost = prompt_cost + completion_cost
            except NotFoundError as e:
                logger.debug(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                cost = 0
            except Exception as e:
                logger.error(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                cost = 0

        return cls(
            completion_cost=cost,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            completion_cost=self.completion_cost + other.completion_cost,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
        )

    def __str__(self) -> str:
        return (
            f"Usage(cost: ${self.completion_cost:.4f}, "
            f"completion tokens: {self.completion_tokens}, "
            f"prompt tokens: {self.prompt_tokens})"
        )


class Completion(BaseModel):
    model: str
    input: list[dict] | None = None
    response: dict[str, Any] | None = None
    usage: Usage | None = None

    @classmethod
    def from_llm_completion(
        cls, input_messages: list[dict], completion_response: Any, model: str
    ) -> Optional["Completion"]:
        if isinstance(completion_response, BaseModel):
            response = completion_response.model_dump()
        elif isinstance(completion_response, dict):
            response = completion_response
        else:
            logger.error(
                f"Unexpected completion response type: {type(completion_response)}"
            )
            return None

        usage = Usage.from_completion_response(completion_response, model)

        return cls(
            model=model,
            input=input_messages,
            response=response,
            usage=usage,
        )


class Response(BaseModel):
    status: str
    message: str
    output: Optional[dict[str, Any]] = None


class VerificationIssueType(Enum):
    SYNTAX_ERROR = "syntax_error"
    TEST_FAILURE = "test_failure"
    LINT = "lint"
    RUNTIME_ERROR = "runtime_error"


class ChangeType(str, Enum):
    addition = "addition"
    modification = "modification"
    deletion = "deletion"



class RankedFileSpan(BaseModel):
    file_path: str
    span_id: str
    rank: int = 0
    tokens: int = 0


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class TestResult(BaseModel):
    status: TestStatus
    message: Optional[str] = None
    file_path: Optional[str] = None
    span_id: Optional[str] = None
    line: Optional[int] = None
    relevant_files: List[RankedFileSpan] = Field(
        default_factory=list,
        description="List of spans that are relevant to the issue",
    )


class CodeChange(BaseModel):
    instructions: str = Field(..., description="Instructions to do the code change.")
    file_path: str = Field(..., description="The file path of the code to be updated.")
    span_id: str = Field(..., description="The span id of the code to be updated.")


class ValueFunctionResult(BaseModel):
    completions: List[Completion] = Field(default_factory=list, description="List of completions done in the get_reward function")
    total_reward: float = Field(0.0, description="Sum of rewards from all completions")
    n_calls: int = Field(0, description="Number of calls made to the value function")
    average_reward: float = Field(0.0, description="Average reward (total_reward / n_calls)")

    @classmethod
    def from_completions(cls, completions: List[Completion], total_reward: float, n_calls: int):
        average_reward = total_reward / n_calls if n_calls > 0 else 0.0
        return cls(
            completions=completions,
            total_reward=total_reward,
            n_calls=n_calls,
            average_reward=average_reward
        )
