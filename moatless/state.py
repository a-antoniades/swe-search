import enum
import logging
import os
import sys
import importlib
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Optional, List, Dict, Tuple, Union
from copy import deepcopy
import json

import instructor
import litellm
import openai
from anthropic import Anthropic, AnthropicBedrock, NotGiven, NOT_GIVEN
from anthropic.types import ToolUseBlock
from instructor import OpenAISchema
from instructor.exceptions import InstructorRetryException
from instructor.utils import classproperty, extract_json_from_codeblock
from litellm import token_counter
from litellm.types.utils import ModelResponse
from openai import OpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, model_validator, Extra, ValidationError

from moatless.file_context import FileContext
from moatless.repository import FileRepository
from moatless.schema import (
    Completion,
    FileWithSpans, ValueFunctionResult, Usage
)
from moatless.settings import Settings
from moatless.utils.llm_utils import LLMResponseFormat, generate_call_id, input_messages_anthropic, input_messages
from moatless.workspace import Workspace

from copy import deepcopy

logger = logging.getLogger(__name__)

class Visit(BaseModel):
    """
    Represent a visit to a state in Monte Carlo Tree Search.
    """
    source_state_id: int = Field(..., description="The future state from which the visit was initaited")
    value: float = Field(0.0, description="The reward value for the current state")
    explanation: Optional[str] = Field(default=None, description="Explanation for the reward value")


class ActionRequest(OpenAISchema):

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        return cls(**tool_args)

    @property
    def action_name(self):
        return self.__class__.__name__

    @classproperty
    def openai_tool_schema(cls):
        return {
            "type": "function",
            "function": cls.openai_schema
        }

    @property
    def log_name(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.__class__.__name__

    def to_prompt(self):
        prompt = f"Action: {self.__class__.__name__}\n"
        prompt += "\n".join([f"  {k}: {v}" for k, v in self.model_dump(
            exclude={"thoughts", "scratch_pad"}).items()])
        return prompt

    model_config = ConfigDict(
        extra='allow',
        arbitrary_types_allowed=True,
    )

    @classmethod
    def parse_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message.content or ""

        # Because Qwen-2.5-72B-Instruct keeps adding those to the responses...
        if '\x00' in message:
            logger.info(f"parse_json() Replace \x00 in: {message}")
            message = message.replace('\x00', '')
        message = extract_json_from_codeblock(message)

        return cls.model_validate_json(
            message,
            context=validation_context,
            strict=strict,
        )


class TakeAction(ActionRequest):

    action: ActionRequest = Field(
        ...,
        description="The action to take",
    )

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        if not tool_name:
            if "action" not in tool_args:
                tool_args["action"] = tool_args

            logger.warning(f"No tool name found, will try to parse with args: {tool_args}")
            return cls(**tool_args)

        action_type = cls.get_sub_type(tool_name)
        if action_type:
            action = action_type(**tool_args)
            return cls(action=action)
        else:
            logger.warning(f"Failed to parse action request from tool call. No action found with name: {tool_name}. Tool args: {tool_args}")
            raise ValueError(f"Failed to parse action request from tool call. No action found with name: {tool_name}.")

    @classmethod
    def get_sub_type(cls, name: str) -> type[ActionRequest] | None:
        for action in cls.available_actions():
            if action.__name__ == name:
                return action
        return None

    @model_validator(mode="before")
    @classmethod
    def validate_steps(cls, data: Any):
        if isinstance(data, dict) and "action_name" in data:
            for action in cls.available_actions():
                if action.__name__ == data["action_name"]:
                    data["action"] = action.model_validate(data["action"])
                    return data

        return data

    def model_dump(self, **kwargs):
        # Merge the exclude sets if provided in both places
        default_exclude = {"previous_state", "next_states", "origin_state", "clones"}
        if "exclude" in kwargs:
            if isinstance(kwargs["exclude"], set):
                kwargs["exclude"] = default_exclude.union(kwargs["exclude"])
            elif isinstance(kwargs["exclude"], dict):
                kwargs["exclude"] = {**dict.fromkeys(default_exclude), **kwargs["exclude"]}
        else:
            kwargs["exclude"] = default_exclude

        data = super().model_dump(**kwargs)
        data["action_name"] = self.action.action_name
        data["action"] = self.action.model_dump(**kwargs)
        return data

    @classmethod
    def available_actions(cls) -> List[type[ActionRequest]]:
        return []

    @property
    def log_name(self):
        return self.action.log_name

    @property
    def name(self):
        return self.action.name

    def to_prompt(self):
        return self.action.to_prompt()


class StateOutcome(BaseModel):
    trigger: Optional[str] = Field(
        default=None,
        description="Trigger to transition to the next state. If None, no transition is made.",
    )
    output: Optional[dict[str, Any]] = Field(
        default=None,
        description="Output data to be passed to the next state.",
    )
    retry_message: Optional[str] = Field(
        default=None, description="Message to use in retry."
    )

    @classmethod
    def retry(cls, retry_message: str):
        return cls(trigger="retry", retry_message=retry_message)

    @classmethod
    def finish(cls, output: dict[str, Any] | None = None):
        output = output or {}
        return cls(trigger="finish", output=output)

    @classmethod
    def reject(cls, message: str, output: dict[str, Any] | None = None):
        output = output or {}
        return cls(trigger="reject", output={"message": message, **output})

    @classmethod
    def transition(cls, trigger: str, output: dict[str, Any] | None = None):
        output = output or {}
        return cls(trigger=trigger, output=output)

    @classmethod
    def stay_in_state(cls, output: dict[str, Any]):
        return cls(output=output)

    @classmethod
    def send_message(cls, message: str, **output):
        """
        Will stay in the same state and provide a message back to the LLM.
        """
        return cls(output={"message": message, **output})


class Content(ActionRequest):
    content: str


class Message(BaseModel):
    role: str
    content: Optional[str] = None


class AssistantMessage(Message):
    role: str = "assistant"
    content: Optional[str] = None
    action: Optional[ActionRequest] = None


class UserMessage(Message):
    role: str = "user"
    content: Optional[str] = None


class State(ABC, BaseModel):
    id: int = Field(..., description="The unique identifier of the state")
    previous_state: Optional["State"] = Field(
        default=None, description="The state that led to this state"
    )
    origin_state: Optional["State"] = Field(
        default=None, description="If the state was cloned from another state"
    )
    clones: List["State"] = Field(
        default_factory=list, description="The clones of the state"
    )

    next_states: List["State"] = Field(
        default_factory=list, description="The states this state transitioned to"
    )

    max_iterations: Optional[int] = Field(
        None, description="The maximum number of transitions to this state."
    )

    max_expansions: int = Field(
        default=3, description="The maximum number of times this state can be expanded."
    )

    visits: List[Visit] = Field(
        default_factory=list, description="The visits to the state in MCTS backpropagation"
    )

    value_function_result: Optional[ValueFunctionResult] = Field(
        default=None,
        description="The result of the value function during MCTS"
    )

    provide_feedback: bool = Field(
        default=True, description="If feedback should be provided from states in alternative branches"
    )

    feedback: Optional[str] = Field(
        default=None, description="Feedback provided the prompt"
    )

    _workspace: Optional[Workspace] = PrivateAttr(None)
    _initial_message: Optional[str] = PrivateAttr(None)
    _executed: bool = PrivateAttr(False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, exclude={"previous_state", "next_states", "origin_state", "clones"}
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._workspace = data.get("_workspace")
        self._initial_message = data.get("_initial_message")

    @abstractmethod
    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def trace_name(self):
        return f"{self.__class__.__name__}{self.id}"

    @property
    def executed(self):
        return self._executed

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def file_repo(self) -> FileRepository:
        return self._workspace.file_repo

    @property
    def file_context(self) -> FileContext:
        return self._workspace.file_context

    @property
    def initial_message(self) -> str:
        return self._initial_message

    def create_file_context(
        self, files: list[FileWithSpans] = None, **kwargs
    ) -> FileContext:
        if files is None:
            files = []
        return self.workspace.create_file_context(files, **kwargs)

    @classmethod
    def required_fields(cls) -> set[str]:
        return set()

    def get_previous_state(
        self, state: "State" = None
    ) -> Optional["State"]:
        prevous_states = self.get_previous_states(state)
        return prevous_states[-1] if prevous_states else None

    def get_previous_states(
        self, state: Optional["State"] = None
    ) -> list["State"]:
        """
        Retrieves previous states of the same type as the given state.
        If no state is provided, it returns all previous states.

        Args:
            state (State | None): The state to filter by. If None, all previous states are returned.

        Returns:
            list: A list of previous states, filtered by type if a state is provided.
        """
        previous_states = []
        current_state = self

        while current_state and current_state.previous_state:
            current_state = current_state.previous_state
            if not state or isinstance(current_state, type(state)):
                previous_states.insert(0, current_state)

        logger.debug(
            f"Found {len(previous_states)} previous states of type {state.__class__.__name__ if state else 'all types'}"
        )

        return previous_states

    def __str__(self):
        return self.model_dump_json(exclude={"previous_state", "next_states", "origin_state", "clones"})

    @property
    def log_name(self):
        return self.__class__.__name__

    @classmethod
    @model_validator(mode="before")
    def validate_previous_state(cls, obj):
        if isinstance(obj, dict) and "previous_state_id" in obj:
            obj = obj.copy()
            obj["previous_state"] = None
        return super().model_validate(obj)

    def model_dump(self, **kwargs):
        exclude = {"previous_state", "next_states", "origin_state", "clones", "value_function_result"}

        state_kwargs = kwargs.copy()
        if "exclude" in kwargs:
            exclude.update(kwargs["exclude"])
            del state_kwargs["exclude"]

        data = super().model_dump(exclude=exclude, **state_kwargs)
        data["value_function_result"] = self.value_function_result.model_dump(**kwargs) if self.value_function_result else None

        return data

    def clone(self) -> "State":
        new_state = self.__class__(**self.model_dump())
        new_state.origin_state = self

        if hasattr(self, "_workspace"):
            new_state._workspace = self._workspace
        if hasattr(self, "_initial_message"):
            new_state._initial_message = self._initial_message

        return new_state

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if self.model_dump() != other.model_dump():
            return False
        return True

    def set_value_function_result(self, result: ValueFunctionResult):
        """
        Set the value function result used during MCTS.

        Args:
            result (ValueFunctionResult): The result of the value function.
        """
        self.value_function_result = result


class NoopState(State):

    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        raise RuntimeError(f"{self.trace_name} is a NoopState that should not be executed")


class Finished(NoopState):
    message: Optional[str] = None
    output: dict[str, Any] | None = None


class Rejected(NoopState):
    message: Optional[str] = None
    error: Optional[str] = None
    output: dict[str, Any] | None = None


class Pending(NoopState):
    def __init__(self, **data):
        if "id" not in data:
            data["id"] = 0
        super().__init__(**data)


class ActionTransaction(BaseModel):
    request: ActionRequest
    response: Optional[StateOutcome] = None
    completion: Optional[Completion] = None

    def model_dump(self, **kwargs):
        if "exclude" in kwargs:
            kwargs["exclude"] = kwargs["exclude"].union({"request", "response"})

        data = super().model_dump(**kwargs)
        data["request"] = self.request.model_dump(**kwargs)
        data["response"] = self.response.model_dump(**kwargs) if self.response else None

        return data



class AgenticState(State):
    model: Optional[str] = Field(
        default=None, description="The model to use for completion"
    )
    temperature: float = Field(0.0, description="The temperature to use for completion")
    model_base_url: Optional[str] = Field(
        default=None, description="The base URL for the model API"
    )
    model_api_key: Optional[str] = Field(
        default=None, description="The API key for the model"
    )
    tool_model: Optional[str] = Field(
        default=None, description="Model to use for tool calls, in combination with the current"
    )
    response_format: LLMResponseFormat = Field(
        LLMResponseFormat.TOOLS, description="The response format expected from the LLM"
    )

    max_tokens: int = Field(
        2000, description="The maximum number of tokens to generate"
    )
    max_message_tokens: Optional[int] = Field(
        None, description="The maximum number of tokens in a single message, can be used as a sanity check"
    )

    include_message_history: bool = Field(
        default=False,
        description="The message history from previous initations should be included in the completion request",
    )

    use_completion_message_history: bool = Field(
        default=False,
        description="Don't generate message history but use the previous completion message log.",
    )

    max_iterations: Optional[int] = Field(
        None, description="The maximum number of transitions to this state."
    )

    max_retries: int = Field(
        5, description="The maximum number of retries before rejecting the state."
    )

    _actions: List[ActionTransaction] = PrivateAttr(default_factory=list)
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validate_response_format(self):
        if self.tool_model:
            tool_model = self.tool_model
        else:
            tool_model = self.model

        if tool_model.startswith("claude") or tool_model.startswith("anthropic"):
            self.response_format = LLMResponseFormat.ANTHROPIC_TOOLS
        elif tool_model in ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "gpt-4o-mini"]:
            self.response_format = LLMResponseFormat.STRUCTURED_OUTPUT
        elif tool_model == "deepseek/deepseek-chat":
            self.response_format = LLMResponseFormat.JSON
        else:
            try:
                support_function_calling = litellm.supports_function_calling(model=tool_model)
            except Exception as e:
                support_function_calling = False

            if not support_function_calling:
                logger.info(f"The model {tool_model} doens't support function calling, set response format to JSON")
                self.response_format = LLMResponseFormat.JSON

        return self

    def execute(self, mocked_action_request: ActionRequest | None = None) -> StateOutcome:
        if self._executed:
            raise ValueError(f"State has already been executed")

        if mocked_action_request:
            action = mocked_action_request
            completion = None
        else:
            try:
                action, completion = self._next_action()
            except ValidationError as e:
                logger.exception(f"{self.trace_name}: Failed to parse request. {str(e)}")
                return StateOutcome.retry(f"Failed to parse request. Error: {str(e)}")
            except ValueError as e:
                logger.exception(f"{self.trace_name}: Failed to parse request. {str(e)}")
                raise e

            except InstructorRetryException as e:
                if e.last_completion:
                    logger.warning(
                        f"{self.trace_name}: Failed to get a valid complection response from the LLM for action request {self.action_type()}. Will set state as rejected. Error {e}. Last completion: {e.last_completion}."
                    )
                    # TODO: Throw an error to abort the flow?

                    error = f"Failed to get a valid response from {self.__class__.__name__}. Error: {e}. Last completion: {e.last_completion}"
                    return StateOutcome.reject(f"Failed to get a valid response from {self.__class__.__name__}.", output={"error": error})
                else:
                    logger.error(
                        f"{self.trace_name}: Failed to get a valid completion response from the LLM for action request {self.action_type()}. Error {e}."
                    )
                    raise e

        logger.debug(f"{self.trace_name}: Received new action {action.action_name}.")
        if self.action_type() and not isinstance(action, self.action_type()):
            raise RuntimeError(
                f"Invalid action type {action.__class__.__name__}, expected type {self.action_type().__name__}"
            )

        response = self._execute_action(action)
        self._actions.append(
            ActionTransaction(request=action, response=response, completion=completion)
        )

        if response.trigger and response.trigger != "retry":
            self._executed = True

        return response

    @abstractmethod
    def _execute_action(self, action: ActionRequest) -> StateOutcome:
        raise NotImplementedError

    @abstractmethod
    def action_type(self) -> type[ActionRequest] | None:
        """
        The type of the action to expect in the completion response.
        If not set a content string is expected.
        """
        raise NotImplementedError

    def init(self) -> Optional[StateOutcome]:
        """
        Initalize the state before exectuting with an action provided wby the LLM.
        Returns a StateOutcome if the state should transition immediately.
        """
        pass

    def handle_action(
        self, action: ActionRequest, completion: Completion | None
    ) -> StateOutcome:
        if self._executed:
            raise ValueError(f"State has already been executed")

        if self.action_type() and not isinstance(action, self.action_type()):
            raise ValueError(
                f"Invalid action type {action.__class__.__name__}, expected type {self.action_type().__name__}"
            )

        response = self._execute_action(action)
        self._actions.append(
            ActionTransaction(request=action, response=response, completion=completion)
        )
        logger.info(f"Added action to {self.name}: {len(self._actions)}")

        if response.trigger and response.trigger != "retry":
            self._executed = True

        return response

    @property
    def actions(self) -> List[ActionTransaction]:
        return self._actions

    @property
    def last_action(self) -> ActionTransaction | None:
        return self._actions[-1] if self._actions else None

    @property
    def action_request(self) -> ActionRequest | None:
        return self._actions[-1].request if self._actions else None

    @property
    def response(self) -> StateOutcome | None:
        return self._actions[-1].response if self._actions else None

    @property
    def outcome(self) -> dict | None:
        return (
            self._actions[-1].response.output
            if self._actions and self._actions[-1].response
            else None
        )

    @property
    def completion(self) -> Completion | None:
        return (
            self._actions[-1].completion
            if self._actions and self._actions[-1].completion
            else None
        )

    @property
    def initial_message(self) -> str:
        return self._initial_message

    @property
    def log_name(self):
        if not self._actions:
            return f"{self.__class__.__name__} (Not executed)"

        return f"{self.__class__.__name__} ({self._actions[-1].request.log_name})"

    def create_message(self) -> str:
        raise NotImplementedError

    def _next_action(
        self,
    ) -> Tuple[ActionRequest, Completion | None]:
        previous_state = self.get_previous_state(self)
        if previous_state and previous_state.completion:
            completion = deepcopy(previous_state.completion)

        if self.model.startswith("claude"):

            if previous_state and self.include_message_history:
                messages = input_messages_anthropic(self.create_message(), previous_state.completion, self.feedback)
            else:
                messages = input_messages_anthropic(self.create_message(), None, self.feedback)

            if self.retry_messages():
                messages.extend(self._map_completion_messages(self.retry_messages()))
        elif self.use_completion_message_history and self.include_message_history and previous_state:
            messages = input_messages(self.create_message(), previous_state.completion, self.feedback)
        else:
            messages = self._to_completion_messages()

        logger.info(f"{self.trace_name}: Create completion with {len(messages)} messages to {self.model} (response_format: {self.response_format}")

        try:
            if self.response_format == LLMResponseFormat.ANTHROPIC_TOOLS:
                action_request, completion_response = self._anthropic_completion(messages)
            elif not self.action_type():
                action_request, completion_response = self._litellm_text_completion(messages)
            elif self.tool_model:
                action_request, completion_response = self._tool_model_completion(messages)
            elif self.response_format == LLMResponseFormat.STRUCTURED_OUTPUT:
                action_request, completion_response = self._openai_completion(messages)
            elif self.response_format == LLMResponseFormat.TOOLS:
                action_request, completion_response = self._litellm_tool_completion(messages)
            else:
                action_request, completion_response = self._instructor_completion(messages)
        except InstructorRetryException as e:
            logger.warning(f"Failed to get completion response from LLM. {e}\n\nCompletion: {e.last_completion}")
            raise e
        except Exception as e:
            logger.warning(f"Failed to get completion response from LLM. {e}. Input messages:\n {json.dumps(messages, indent=2)}")
            raise RuntimeError(f"Failed to get completion response from LLM. Error: {str(e)}")

        completion = Completion.from_llm_completion(
            input_messages=messages,
            completion_response=completion_response,
            model=self.model,
        )

        return action_request, completion

    def create_file_context(
        self, files: list[FileWithSpans] = None, **kwargs
    ) -> FileContext:
        if files is None:
            files = []
        return self.workspace.create_file_context(files, **kwargs)

    def init(self):
        """Initialization logic for the state."""
        pass

    def messages(self) -> list[Message]:
        return []

    def retries(self) -> int:
        retries = 0
        for action in reversed(self._actions):
            if action.response.trigger == "retry":
                retries += 1
            else:
                return retries

        return retries

    def retry_messages(self) -> list[Message]:
        messages: list[Message] = []

        for action in self._actions:
            if isinstance(action.request, Content):
                messages.append(
                    AssistantMessage(
                        content=action.request.content,
                    )
                )
            else:
                if hasattr(action.request, "action"):
                    action_request = action.request.action
                else:
                    action_request = action.request
                messages.append(AssistantMessage(action=action_request))

            if action.response.retry_message:
                messages.append(
                    UserMessage(
                        content=action.response.retry_message,
                    )
                )

        return messages

    def _litellm_text_completion(self, messages: list[dict]) -> Tuple[Content, ModelResponse]:
        litellm.drop_params = True

        completion_response = litellm.completion(
            model=self.model,
            base_url=self.model_base_url,
            api_key=self.model_api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_words(),
            metadata=self._metadata,
            messages=messages,
        )
        return Content(
            content=completion_response.choices[0].message.content
        ), completion_response

    def _litellm_tool_completion(self, messages: list[dict], is_retry: bool = False) -> Tuple[ActionRequest, ModelResponse]:
        litellm.drop_params = True

        tools = []
        if hasattr(self.action_type(), "available_actions"):
            for action in self.action_type().available_actions():
                tools.append(openai.pydantic_function_tool(action))
        else:
            tools.append(openai.pydantic_function_tool(self.action_type()))

        completion_response = litellm.completion(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_words(),
            metadata=self._metadata,
            tools=tools,
            tool_choice="auto",
            messages=messages,
        )

        tool_args, tool_name, retry_message = None, None, None
        if not completion_response.choices[0].message.tool_calls and completion_response.choices[0].message.content:
            if "```json" in completion_response.choices[0].message.content:
                logger.info(f"Found no tool call but JSON in completion response, will try to parse")

                try:
                    action_request = self.action_type().from_response(
                        completion_response, mode=instructor.Mode.TOOLS
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse JSON as tool call, will try to parse as JSON ")

                    try:
                        action_request = self.action_type().from_response(
                            completion_response, mode=instructor.Mode.JSON
                        )
                    except Exception as e:
                        logger.exception(f"Failed to parse JSON as tool call from completion response: {completion_response.choices[0].message.content}")
                        raise e

                return action_request, completion_response
            elif completion_response.choices[0].message.content.startswith("{"):
                tool_args = json.loads(completion_response.choices[0].message.content)

            if tool_args:
                if "name" in tool_args:
                    tool_name = tool_args.get("name")

                if "parameters" in tool_args:
                    tool_args = tool_args["parameters"]

        elif completion_response.choices[0].message.tool_calls[0]:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            tool_args = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name

        if not tool_args:
            if is_retry:
                logger.error(
                    f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}")
                raise ValueError(f"No tool call in response from LLM.")

            logger.warning(
                f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}. Will retry")
            messages.append({"role": "assistant", "content": completion_response.choices[0].message.content})
            if not retry_message:
                retry_message = "You must response with a tool call."
            messages.append({"role": "user", "content": retry_message})
            return self._litellm_tool_completion(messages, is_retry=True)

        action_request = self.action_type().from_tool_call(tool_args=tool_args, tool_name=tool_name)
        return action_request, completion_response

    def _instructor_completion(self, messages: list[dict], is_retry: bool = False) -> Tuple[ActionRequest, ModelResponse]:
        if self.response_format == LLMResponseFormat.JSON or hasattr(self.action_type(), "available_actions"):
            client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        elif self.response_format == LLMResponseFormat.JSON:
            client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        else:
            client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.TOOLS)

        try:
            action_request, completion_response = client.chat.completions.create_with_completion(
                model=self.model,
                base_url=self.model_base_url,
                api_key=self.model_api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature if not self.tool_model else 0.0,
                stop=self.stop_words(),
                messages=messages,
                response_model=self.action_type()
            )

            return action_request, completion_response

        except InstructorRetryException as e:
            logger.error(json.dumps(e.messages, indent=2))
            raise e
        except Exception as e:
            logger.exception(f"Failed to get completion response from litellm: {e}")
            raise e

    def function_call_system_prompt(self):
        return """You are an AI language model tasked with transforming unstructured messages wrapped in the XML tag <message> into structured tool calls. Your guidelines are:

         * Do not change, omit, or add any information from the original message.
         * Focus solely on converting the existing information into the correct tool call format.
         * Extract all relevant details necessary for the tool call without altering their meaning.
         * Ignore planned steps in the tool call
         * Provide the reasoning in the scratch_pad field

        Your response should be the tool call generated from the provided unstructured message, adhering strictly to these instructions."""

    def function_call_prompt(self, llm_response: str):
        content = "<message>\n"
        content += llm_response
        content += "\n</message>"
        return content

    def _tool_model_completion(self, messages: list[dict]) -> Tuple[ActionRequest, ModelResponse]:
        litellm.drop_params = True

        if self.model.startswith("claude"):
            anthropic_client = Anthropic()
            completion_response = (
                anthropic_client.beta.prompt_caching.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=[{
                        "text": self.system_prompt(),
                        "type": "text",
                        "cache_control": {"type": "ephemeral"}
                    }],
                    messages=messages[1:],
                )
            )
            content = completion_response.content[0].text
        else:
            completion_response = litellm.completion(
                model=self.model,
                # max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.stop_words(),
                metadata=self._metadata,
                messages=messages,
            )
            content = completion_response.choices[0].message.content

        messages = [
            {"role": "system", "content": self.function_call_system_prompt()},
            {"role": "user", "content": self.function_call_prompt(content)}]

        action_request, tool_completion_response = self._openai_completion(messages)
        action_request = self._create_request_from_tool_call(action_request, content)

        logger.info(f"_tool_model_completion() Input message:\n{content}\n\nAction request:\n{action_request.model_dump_json(indent=2)}")

        return action_request, completion_response

    def _create_request_from_tool_call(self, action_request: ActionRequest, response: str):
        return action_request

    def _openai_completion(self, messages: list[dict], is_retry: bool = False):
        if os.getenv("AZURE_API_KEY"):
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_API_KEY"),
                api_version="2024-07-01-preview",
                azure_endpoint=os.getenv("AZURE_API_BASE")
            )
        else:
            client = OpenAI()

        tools = []
        if hasattr(self.action_type(), "available_actions"):
            for action in self.action_type().available_actions():
                tools.append(openai.pydantic_function_tool(action))
        else:
            tools.append(openai.pydantic_function_tool(self.action_type()))

        completion_response = client.beta.chat.completions.parse(
            model=self.tool_model or self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature if not self.tool_model else 0.0,
            stop=self.stop_words(),
            messages=messages,
            tool_choice="required",
            tools=tools,
            parallel_tool_calls=False,
            response_format=self.action_type()
        )

        if not completion_response.choices[0].message.tool_calls:
            if is_retry:
                logger.error(
                    f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}")
                raise RuntimeError(f"No tool call in response from LLM.")

            logger.warning(
                f"No tool call return on request with tools: {tools}.\n\nCompletion response: {completion_response}. Will retry")
            messages.append({"role": "assistant", "content": completion_response.choices[0].message.content})
            messages.append({"role": "user", "content": "You must response with a tool call."})
            return self._openai_completion(messages, is_retry=True)

        tool_call = completion_response.choices[0].message.tool_calls[0]
        if hasattr(self.action_type(), "available_actions"):
            tool_action_request = tool_call.function.parsed_arguments
            action_request = self.action_type()(action=tool_action_request)
        else:
            action_request = tool_call.function.parsed_arguments

        return action_request, completion_response

    def _anthropic_completion(self, messages: list[dict]) -> Tuple[ActionRequest, Message]:
        if self.model.startswith("anthropic"):
            anthropic_client = AnthropicBedrock()
        else:
            anthropic_client = Anthropic()

        if self.action_type():
            tools = []
            tool_choice = {"type": "any"}
            if hasattr(self.action_type(), "available_actions"):
                for action in self.action_type().available_actions():
                    tools.append(action.anthropic_schema)
            else:
                tools.append(self.action_type().anthropic_schema)
        else:
            tools = NOT_GIVEN
            tool_choice = NOT_GIVEN

        completion_response = (
            anthropic_client.beta.prompt_caching.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=[{
                        "text": self.system_prompt(),
                        "type": "text",
                        "cache_control": {"type": "ephemeral"}
                    }],
                tool_choice=tool_choice,
                tools=tools,
                messages=messages,
            )
        )

        try:
            if not self.action_type():
                action_request = Content(
                    content=completion_response.content[0].text
                )
            elif hasattr(self.action_type(), "available_actions"):
                action_request = None
                if hasattr(self.action_type(), "available_actions"):
                    for block in completion_response.content:
                        if isinstance(block, ToolUseBlock):
                            action = None
                            for available_action in self.action_type().available_actions():
                                if available_action.__name__ == block.name:
                                    action = available_action
                                    break

                            if not action:
                                raise ValueError(f"Unknown action {block.name}")

                            tool_action_request = action.model_validate(block.input)

                            action_request = self.action_type()(action=tool_action_request)

                            # TODO: We only support one action at the moment
                            break
                        else:
                            logger.warning(f"Unexpected block {block}]")
            else:
                action_request = self.action_type().from_response(
                    completion_response, mode=instructor.Mode.ANTHROPIC_TOOLS
                )

        except Exception as e:
            logger.exception(
                f"Failed to parse action request from completion response. Completion: {completion_response}")
            raise e

        return action_request, completion_response

    def _get_tool_call(self, completion_response) -> Tuple[str, dict]:
        if not completion_response.choices[0].message.tool_calls and completion_response.choices[0].message.content:
            if "```json" in completion_response.choices[0].message.content:
                content = completion_response.choices[0].message.content
                json_start = content.index("```json") + 7
                json_end = content.rindex("```")
                json_content = content[json_start:json_end].strip()
            elif completion_response.choices[0].message.content.startswith("{"):
                json_content = completion_response.choices[0].message.content
            else:
                return None, None

            tool_call = json.loads(json_content)
            return tool_call.get("name"), tool_call

        elif completion_response.choices[0].message.tool_calls:
            tool_call = completion_response.choices[0].message.tool_calls[0]
            tool_dict = json.loads(tool_call.function.arguments)
            return tool_call.function.name, tool_dict

        return None

    def _to_completion_messages(self) -> list[dict]:
        messages = []

        if self.response_format != LLMResponseFormat.ANTHROPIC_TOOLS and not self.model.startswith("o1"):
            messages.append({"role": "system", "content": self.system_prompt()})

        state_messages = self.messages()

        # TODO: Fix htis with a more generic solution to generate the new input message
        if self.feedback:
            state_messages[-1].content += f"\n\n{self.feedback}"

        state_messages.extend(self.retry_messages())

        messages.extend(self._map_completion_messages(state_messages))
        return messages

    def _map_completion_messages(self, state_messages: list[Message]) -> list[dict]:
        tool_call_id = None
        messages = []
        for message in state_messages:
            if message.role == "user":
                if not self.tool_model and tool_call_id and self.response_format in [
                    LLMResponseFormat.TOOLS,
                    LLMResponseFormat.STRUCTURED_OUTPUT,
                ]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message.content,
                        }
                    )
                elif (
                    not self.tool_model
                    and tool_call_id
                    and self.response_format in [
                        LLMResponseFormat.TOOLS,
                        LLMResponseFormat.STRUCTURED_OUTPUT,
                    ]
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "tool_use_id": tool_call_id,
                                    "content": message.content,
                                    "type": "tool_result",
                                }
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                if message.action:
                    tool_call_id = generate_call_id()
                    if not self.tool_model and self.response_format == LLMResponseFormat.ANTHROPIC_TOOLS:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "id": tool_call_id,
                                        "input": message.action.model_dump(),
                                        "type": "tool_use",
                                        "name": message.action.action_name,
                                    }
                                ],
                            }
                        )
                    elif not self.tool_model and self.response_format in [
                        LLMResponseFormat.TOOLS,
                        LLMResponseFormat.STRUCTURED_OUTPUT,
                    ]:
                        messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": message.action.action_name,
                                            "arguments": message.action.model_dump_json(
                                                exclude_none=True
                                            ),
                                        },
                                    }
                                ],
                            }
                        )
                    else:
                        json_content = message.action.model_dump_json(indent=2)

                        # TODO Only if self.model.startswith("deepseek"): ?
                        json_content = f"```json\n{json_content}\n```"

                        messages.append(
                            {
                                "role": "assistant",
                                "content": json_content,
                            }
                        )

                else:
                    tool_call_id = None
                    messages.append({"role": "assistant", "content": message.content})

        return messages

    def system_prompt(self) -> str:
        logger.warning(f"{self.trace_name} System prompt not implemented")
        return ""

    def stop_words(self) -> list[str] | None:
        return None

    def total_usage(self) -> Usage:
        total_usage = Usage()
        for action in self._actions:
            if action.completion and action.completion.usage:
                total_usage += action.completion.usage

        if self.value_function_result:
            for completion in self.value_function_result.completions:
                if completion.usage:
                    total_usage += completion.usage

        return total_usage

    def total_cost(self):
        total_usage = self.total_usage()
        if total_usage:
            return total_usage.completion_cost
        else:
            return 0

    def model_dump(self, **kwargs):
        if "exclude" not in kwargs:
            kwargs["exclude"] = {"previous_state", "next_states", "origin_state", "clones"}

        data = super().model_dump(**kwargs)
        return data

    def __str__(self):
        return self.model_dump_json(exclude={"previous_state", "next_states", "origin_state", "clones"})

    def __eq__(self, other):
        if not isinstance(other, AgenticState):
            return NotImplemented
        if self.model_dump() != other.model_dump():
            return False
        return True


def get_state_class(name: str) -> type[AgenticState]:
    builtin_states = {
        "NoopState": NoopState,
        "Finished": Finished,
        "Rejected": Rejected,
        "Pending": Pending,
    }
    if name in builtin_states:
        return builtin_states[name]

    # If not a built-in state, try to import dynamically
    possible_modules = [
        "moatless.edit",
        "moatless.find",
    ]

    for module_name in possible_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                cls = getattr(module, name)
                if isinstance(cls, type) and issubclass(cls, State):
                    return cls
        except ImportError:
            logger.debug(f"Could not import module {module_name}")

    # If still not found, try sys.modules as a fallback
    for module in sys.modules.values():
        if hasattr(module, name):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, State):
                return cls

    raise ValueError(f"State {name} not found")
