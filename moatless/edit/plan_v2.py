from enum import Enum
import json
import logging
from typing import Optional, List, Any, Union, Tuple

from pydantic import ConfigDict, Field, PrivateAttr, BaseModel, model_validator

from moatless.codeblocks import CodeBlockType, PythonParser
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, CodeBlock, SpanType
from moatless.edit.plan import _get_post_end_line_index, _get_pre_start_line
from moatless.edit.prompt_v2 import PLAN_TO_CODE_SYSTEM_PROMPT, TOOL_MODEL_PROMPT, FEW_SHOT_JSON
from moatless.file_context import ContextFile, FileContext
from moatless.index.code_index import is_test
from moatless.index.types import SearchCodeResponse
from moatless.repository import CodeFile
from moatless.schema import (
    VerificationIssueType,
    FileWithSpans,
    ChangeType,
    TestStatus,
    RankedFileSpan,
)
from moatless.state import (
    AgenticState,
    ActionRequest,
    StateOutcome,
    AssistantMessage,
    Message,
    UserMessage,
    TakeAction,
)
from moatless.utils.llm_utils import LLMResponseFormat
from moatless.utils.tokenizer import count_tokens

from moatless.verify.lint import TestResult

logger = logging.getLogger("PlanToCode")


class RequestCodeChange(ActionRequest):
    """
    Request for the next code change.
    """

    scratch_pad: str = Field(
        ...,
        description="Your step by step reasoning on how to do the code change and whats the next step is.",
    )
    change_type: ChangeType = Field(
        ...,
        description="A string that can be set to 'addition', 'modification', or 'deletion'. 'Addition' refers to adding a new function or class, 'modification' refers to changing existing code, and 'deletion' refers to removing a function or class.",
    )
    instructions: str = Field(
        ..., description="Instructions about the next step to do the code change."
    )
    file_path: str = Field(..., description="The file path of the code to be updated.")
    start_line: int = Field(
        ..., description="The start line of the existing code to be updated."
    )
    end_line: int = Field(
        ...,
        description="The end line of the code to be updated when modifying existing code.",
    )
    pseudo_code: str = Field(..., description="Pseudo code for the code change.")

    planned_steps: List[str] = Field(
        default_factory=list,
        description="Planned steps that should be executed after the current step.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_steps(cls, data: Any):
        # Claude sometimes returns steps as a string instead of a list
        if isinstance(data, dict):

            if "planned_steps" in data and isinstance(data["planned_steps"], str):
                logger.info(
                    f"validate_steps: Converting planned_steps to list: {data['planned_steps']}"
                )
                data["planned_steps"] = data["planned_steps"].split("\n")

            if "pseudo_code" not in data:
                data["pseudo_code"] = ""

            if "instructions" not in data:
                data["instructions"] = ""

        return data

    @model_validator(mode="before")
    def convert_status_to_enum(cls, values):
        if isinstance(values, dict) and isinstance(values.get("change_type"), str):
            values["change_type"] = ChangeType(values["change_type"])
        return values

    @property
    def log_name(self):
        return f"RequestCodeChange({self.file_path} {self.start_line}-{self.end_line})"

    def to_prompt(self):
        return f"""Request code change in file {self.file_path}.

Instructions: {self.instructions}
Change type: {self.change_type.value}
Line numbers: {self.start_line}-{self.end_line}

Pseudo code: 
```
{self.pseudo_code}
```
"""


class CodeSpan(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    start_line: Optional[int] = Field(
        None, description="The start line of the code to add to context."
    )
    end_line: Optional[int] = Field(
        None, description="The end line of the code to add to context."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="Span IDs identiying the relevant code spans. A span id is a unique identifier for a code sippet. It can be a class name or function name. For functions in classes separete with a dot like 'class.function'.",
    )

    @property
    def log_name(self):
        log = self.file_path

        if self.start_line and self.end_line:
            log += f" {self.start_line}-{self.end_line}"

        if self.span_ids:
            log += f" {', '.join(self.span_ids)}"

        return log


class RequestMoreContext(ActionRequest):
    """
    Request to see code that is not in the current context.
    """

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    files: List[CodeSpan] = Field(
        ..., description="The code that should be provided in the file context."
    )

    @property
    def log_name(self):
        if len(self.files) == 1:
            return f"RequestMoreContext({self.files[0].log_name})"
        else:
            logs = []
            for i, file in enumerate(self.files):
                logs.append(f"{i}=[{file.log_name}]")
            return f"RequestMoreContext(" + ", ".join(logs) + ")"

    def to_prompt(self):
        prompt = "Requesting more context for the following files:\n"
        for file in self.files:
            prompt += f"* {file.file_path}\n"
            if file.start_line and file.end_line:
                prompt += f"  Lines: {file.start_line}-{file.end_line}\n"
            if file.span_ids:
                prompt += f"  Spans: {', '.join(file.span_ids)}\n"
        return prompt


class SemanticSearch(ActionRequest):
    """
    Request to search for code using semantic search.
    """

    scratch_pad: str = Field(
        ..., description="Your thoughts on how to define the search."
    )

    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    query: Optional[str] = Field(
        default=None,
        description="A semantic similarity search query. Use natural language to describe what you are looking for.",
    )

    @property
    def log_name(self):
        return f"SemanticSearch({self.query[:20]}...)"

    def to_prompt(self):
        prompt = f"Searching for code using the query: {self.query}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt


class FindClass(ActionRequest):
    """
    Find a specific class in the code base.
    """

    scratch_pad: str = Field(
        ..., description="Your thoughts on how to define the search."
    )

    class_name: str = Field(
        ..., description="Specific class name to include in the search."
    )

    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    @property
    def log_name(self):
        return f"FindClass({self.class_name})"

    def to_prompt(self):
        prompt = f"Searching for class: {self.class_name}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

class FindFunction(ActionRequest):
    """
    Find a specific function in the code base.
    """

    scratch_pad: str = Field(
        ..., description="Your thoughts on how to define the search."
    )

    function_name: str = Field(
        ..., description="Specific function names to include in the search."
    )

    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    class_name: Optional[str] = Field(
        default=None, description="Specific class name to include in the search."
    )

    @property
    def log_name(self):
        if self.class_name:
            return f"FindFunction({self.class_name}.{self.function_name})"

        return f"FindFunction({self.function_name})"

    def to_prompt(self):
        prompt = f"Searching for function: {self.function_name}"
        if self.class_name:
            prompt += f" in class: {self.class_name}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

class Review(ActionRequest):
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    review_instructions: str = Field(..., description="Review instructions.")

    @model_validator(mode="before")
    def convert_instructions(cls, values):
        if isinstance(values, dict) and values.get("instructions"):
            values["review_instructions"] = values["instructions"]
        return values

    def to_prompt(self):
        return f"Review with instructions: {self.review_instructions}"

class Finish(ActionRequest):
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    finish_reason: str = Field(..., description="Finish the request and explain why")

    @model_validator(mode="before")
    def convert_reason(cls, values):
        if isinstance(values, dict):
            if values.get("reason"):
                values["finish_reason"] = values["reason"]

            if "finish_reason" not in values:
                values["finish_reason"] = "No reason given."

            if "scratch_pad" not in values:
                values["scratch_pad"] = ""

        return values

    def to_prompt(self):
        return f"Finish with reason: {self.finish_reason}"


class Reject(ActionRequest):
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    rejection_reason: str = Field(
        ..., description="Reject the request and explain why."
    )

    @model_validator(mode="before")
    def convert_reason(cls, values):
        if values and isinstance(values, dict):
            if values.get("reason"):
                values["rejection_reason"] = values["reason"]

            if "rejection_reason" not in values:
                values["rejection_reason"] = "No reason given."

            if "scratch_pad" not in values:
                values["scratch_pad"] = ""

            return values

    def to_prompt(self):
        return f"Reject with reason: {self.rejection_reason}"


class PlanRequest(TakeAction):
    """
    Request to apply a change to the code.
    """

    action: Union[
        RequestCodeChange,
        RequestMoreContext,
        SemanticSearch,
        FindClass,
        FindFunction,
        Review,
        Finish,
        Reject,
    ]

    @classmethod
    def available_actions(cls) -> List[ActionRequest]:
        return [
            RequestCodeChange,
            RequestMoreContext,
            SemanticSearch,
            FindClass,
            FindFunction,
            Review,
            Finish,
            Reject,
        ]


class PlanToCode(AgenticState):
    message: Optional[str] = Field(
        None,
        description="Message from last transitioned state",
    )

    diff: Optional[str] = Field(
        None,
        description="The diff of a previous code change.",
    )

    updated_file_context: Optional[List[FileWithSpans]] = Field(
        None,
        description="The last updated file context.",
    )

    max_prompt_file_tokens: int = Field(
        4000,
        description="The maximum number of tokens in the file context to show in the prompt.",
    )

    max_tokens_in_edit_prompt: int = Field(
        500,
        description="The maximum number of tokens in a span to show in the edit prompt.",
    )

    min_tokens_in_edit_prompt: int = Field(
        50,
        description="The minimum number of tokens in a span to show in the edit prompt.",
    )

    max_tokens_in_search_response: int = Field(
        1000,
        description="The maximum number of tokens in the search response to show. If the response is larger, the user will be asked to narrow down the search.",
    )

    finish_on_review: bool = Field(
        False, description="Whether to finish the task if a review is requested."
    )

    include_message_history: bool = Field(
        True,
        description="Whether to include the message history in the prompt.",
    )

    verify: bool = Field(
        True,
        description="Whether to run verification job before executing the next action.",
    )

    test_results: list[TestResult] | None = Field(
        None,
        description="Results from test run.",
    )

    provide_feedback: bool = Field(
        default=True, description="If feedback should be provided from states in alternative branches"
    )

    def _execute_action(self, action: TakeAction) -> StateOutcome:
        if isinstance(action.action, Review):
            if self.diff and self.finish_on_review:
                logger.info("Review suggested after diff, will finish")
                return StateOutcome.transition(
                    trigger="finish", output={"message": "Finish on suggested review."}
                )
            else:
                return StateOutcome.send_message(
                    "Review isn't possible. If the change is done you can finish or reject the task."
                )

        if isinstance(action.action, Finish):
            return StateOutcome.transition(
                trigger="finish", output={"message": action.action.finish_reason}
            )
        elif isinstance(action.action, Reject):
            return StateOutcome.transition(
                trigger="reject", output={"message": action.action.rejection_reason}
            )
        elif isinstance(action.action, SemanticSearch):
            return self._semantic_search(action.action)
        elif isinstance(action.action, FindClass):
            return self._find_class(action.action)
        elif isinstance(action.action, FindFunction):
            return self._find_function(action.action)
        elif isinstance(action.action, RequestCodeChange):
            return self._request_for_change(action.action)
        elif isinstance(action.action, RequestMoreContext):
            return self._request_more_context(action.action)
        return StateOutcome.retry(
            "You must either provide an apply_change action or finish."
        )

    def init(self) -> Optional[StateOutcome]:
        if self.test_results is not None:
            logger.info(
                f"{self.trace_name} Already ran {len(self.test_results)} tests"
            )
            return None

        if self.verify:
            # Run all test files that are in context
            test_files = [
                file for file in self.file_context.files if is_test(file.file_path)
            ]
            if not test_files:
                logger.info(
                    f"{self.trace_name} No test files in the file context, will not run tests."
                )
            else:
                test_file_paths = [file.file_path for file in test_files]
                self.test_results = self.workspace.run_tests(test_file_paths)
                if self.test_results:
                    failing_tests = [
                        issue
                        for issue in self.test_results
                        if issue.status in [TestStatus.FAILED, TestStatus.ERROR]
                    ]
                    tests_with_output = [
                        issue
                        for issue in failing_tests
                        if issue.message and issue.file_path
                    ]
                    if failing_tests:
                        logger.info(
                            f"{self.trace_name} {len(failing_tests)} out of {len(self.test_results)} tests failed. "
                            f"Include spans for {len(tests_with_output)} tests with output."
                        )

                    # Keep file context size down by replacing spans with failing spans
                    failed_test_spans_by_file_path: dict = {}
                    for issue in tests_with_output:
                        if issue.file_path:
                            failed_test_spans_by_file_path.setdefault(
                                issue.file_path, []
                            ).append(issue.span_id)

                    for test_file in test_files:
                        failed_span_ids = failed_test_spans_by_file_path.get(
                            test_file.file_path
                        )
                        if failed_span_ids:
                            # TODO: Find a way to rank spans to keep the most relevant ones instead of replacing...
                            test_file.remove_all_spans()
                            test_file.add_spans(failed_span_ids)

    def action_type(self) -> type[PlanRequest]:
        return PlanRequest

    def _request_more_context(self, action: RequestMoreContext) -> StateOutcome:
        logger.info(f"{self.trace_name}:RequestMoreContext: {action.files}")

        updated_file_context = self.create_file_context()

        retry_message = ""
        for file_with_spans in action.files:
            file = self.file_repo.get_file(file_with_spans.file_path)
            if not file:
                logger.info(
                    f"{self.trace_name}:RequestMoreContext: {file_with_spans.file_path} is not found in the file repository."
                )
                return StateOutcome.retry(f"The requested file {file_with_spans.file_path} is not found in the file repository. Use the search functions to search for the code if you are unsure of the file path.")

            if file_with_spans.start_line and file_with_spans.end_line:
                self.file_context.add_line_span_to_context(
                    file.file_path, file_with_spans.start_line, file_with_spans.end_line
                )
            elif not file_with_spans.span_ids and not file.module:
                return StateOutcome.retry(
                    f"Please provide the line numbers to add to context."
                )

            elif not file_with_spans.span_ids:
                return StateOutcome.retry(
                    f"Please provide the line numbers or span ids for the code to add to context.\n\nAvailable span ids:\n{self.span_id_list(file.module.span_ids)})"
                )

            missing_span_ids = set()
            suggested_span_ids = set()
            found_span_ids = set()
            if file_with_spans.span_ids and not file.module:
                logger.warning(
                    f"{self.trace_name}:RequestMoreContext: Tried to add span ids {file_with_spans.span_ids} to not parsed file {file.file_path}."
                )
                return StateOutcome.retry(
                    f"No span ids found in file {file.file_path}. Is it empty?"
                )

            for span_id in file_with_spans.span_ids:
                block_span = file.module.find_span_by_id(span_id)
                if not block_span:
                    # Try to find the relevant code block by code block identifier
                    block_identifier = span_id.split(".")[-1]
                    blocks = file.module.find_blocks_with_identifier(block_identifier)

                    if not blocks:
                        missing_span_ids.add(span_id)
                    elif len(blocks) > 1:
                        for block in blocks:
                            if block.belongs_to_span.span_id not in suggested_span_ids:
                                suggested_span_ids.add(block.belongs_to_span.span_id)
                    else:
                        block_span = blocks[0].belongs_to_span

                if block_span:

                    if block_span.initiating_block.type == CodeBlockType.CLASS:
                        class_block = block_span.initiating_block
                        found_span_ids.add(block_span.span_id)
                        if (
                                class_block.sum_tokens()
                                < self.max_tokens_in_edit_prompt
                        ):
                            for child_span_id in class_block.span_ids:
                                found_span_ids.add(child_span_id)
                    else:
                        found_span_ids.add(block_span.span_id)

            if missing_span_ids:
                logger.info(
                    f"{self.trace_name}:RequestMoreContext: Spans not found in {file_with_spans.file_path}: {', '.join(missing_span_ids)}"
                )
                retry_message += f"Spans not found in {file_with_spans.file_path}: {', '.join(missing_span_ids)}\n"
            else:
                updated_file_context.add_spans_to_context(file.file_path, found_span_ids)

            if retry_message and suggested_span_ids:
                logger.info(
                    f"{self.trace_name}:RequestMoreContext: Suggested spans: {', '.join(suggested_span_ids)}"
                )
                retry_message += f"Did you mean one of these spans: {', '.join(suggested_span_ids)}\n"

        if retry_message:
            return StateOutcome.retry(retry_message)

        for code_span in action.files:
            if code_span.start_line and code_span.end_line:
                updated_file_context.add_line_span_to_context(
                    code_span.file_path, code_span.start_line, code_span.end_line
                )

            for span_id in code_span.span_ids:
                if not self.file_context.has_span(code_span.file_path, span_id):
                    updated_file_context.add_span_to_context(code_span.file_path, span_id)

        self.file_context.add_files_with_spans(updated_file_context.to_files_with_spans())

        message = "Added new spans:\n"
        for file in action.files:
            message += f" * {file.file_path} ({', '.join(file.span_ids)})\n"

        return StateOutcome.stay_in_state(output={"message": message, "updated_file_context": updated_file_context.to_files_with_spans()})

    def _semantic_search(self, ss: SemanticSearch) -> StateOutcome:
        logger.info(
            f"{self.trace_name}:SemanticSearch: {ss.query} (file_pattern: {ss.file_pattern})"
        )

        search_result = self.workspace.code_index.semantic_search(
            ss.query, file_pattern=ss.file_pattern
        )
        return self._handle_search_results(search_result)

    def _find_class(self, fc: FindClass) -> StateOutcome:
        logger.info(
            f"{self.trace_name}:FindClass: {fc.class_name} (file_pattern: {fc.file_pattern})"
        )
        search_result = self.workspace.code_index.find_class(
            fc.class_name, file_pattern=fc.file_pattern
        )
        return self._handle_search_results(search_result)

    def _find_function(self, ff: FindFunction) -> StateOutcome:
        logger.info(
            f"{self.trace_name}:FindFunction: {ff.function_name} (class_name: {ff.class_name}, file_pattern: {ff.file_pattern})"
        )
        search_result = self.workspace.code_index.find_function(
            ff.function_name, class_name=ff.class_name, file_pattern=ff.file_pattern
        )
        return self._handle_search_results(search_result)

    def _handle_search_results(self, search_result: SearchCodeResponse):
        if not search_result.hits:
            logger.info("Unfortunately, I didn't find any relevant results.")
            return StateOutcome.retry(search_result.message)

        if (
                search_result.sum_tokens() <= self.max_tokens_in_search_response
                and search_result.sum_tokens() + self.file_context.context_size()
                <= self.max_prompt_file_tokens
        ):
            updated_context = self.create_file_context()
            for hit in search_result.hits:
                for span in hit.spans:
                    self.file_context.add_span_to_context(hit.file_path, span.span_id)
                    updated_context.add_span_to_context(hit.file_path, span.span_id)

            logger.info(
                f"Found code in {len(search_result.hits)} files with {search_result.sum_tokens()}. "
                f"File context size is now {self.file_context.context_size()}."
            )

            search_result_str = "I found code in the following files and added it to the file context:\n" + "\n".join(
                [f"* {hit}" for hit in search_result.hits]
            )
            return StateOutcome.send_message(search_result_str, updated_file_context=updated_context.to_files_with_spans())

        ranked_spans = []
        for hit in search_result.hits:
            for span in hit.spans:
                ranked_spans.append(
                    RankedFileSpan(
                        file_path=hit.file_path,
                        span_id=span.span_id,
                        rank=span.rank,
                        tokens=span.tokens,
                    )
                )

        return StateOutcome.transition(
            trigger="identify_code",
            output={"ranked_spans": ranked_spans},
        )

    def _request_for_change(self, rfc: RequestCodeChange) -> StateOutcome:
        logger.info(
            f"{self.trace_name}:RequestCodeChange: file_path={rfc.file_path}, start_line={rfc.start_line}, end_line={rfc.end_line}, change_type={rfc.change_type}"
        )

        if not rfc.instructions:
            return StateOutcome.retry(
                f"Please provide instructions for the code change."
            )

        if rfc.pseudo_code is None:
            return StateOutcome.retry(
                f"Please provide pseudo code for the code change."
            )

        context_file = self.file_context.get_file(rfc.file_path)
        if not context_file:
            if self.file_repo.is_directory(rfc.file_path):
                return StateOutcome.retry(
                    f"{rfc.file_path} is a directory. Please provide a file path."
                )

            file = self.file_repo.get_file(rfc.file_path)
            if not file:
                logger.info(f"{self.trace_name}:RequestCodeChange: File {rfc.file_path} is not found in the file repository. Will create it and add to context.")
                self.file_repo.create_empty_file(rfc.file_path)
                return StateOutcome.transition(
                    trigger="edit_code",
                    output={
                        "instructions": rfc.instructions,
                        "pseudo_code": rfc.pseudo_code,
                        "file_path": rfc.file_path,
                        "change_type": ChangeType.addition,
                        "start_line": 0,
                        "end_line": 0,
                    },
                )
            else:
                logger.info(f"{self.trace_name}:RequestCodeChange: File {rfc.file_path} is not found in the file context, ask for clarification.")
                if file.module:
                    span_ids = file.module.span_ids
                else:
                    logger.info(f"{self.trace_name}:RequestCodeChange: No code blocks found in module for {rfc.file_path}")
                    span_ids = []

                return StateOutcome.retry(f"File {rfc.file_path} is not found in the file context. Use the RequestMoreContext function to add it and the relevant span ids. \n\nAvailable span ids:\n{self.span_id_list(span_ids)})")

        retry_message = self.verify_request(rfc, context_file)
        if retry_message:
            return retry_message

        if context_file.module:
            start_line, end_line, change_type = self.get_line_span(
                rfc.change_type,
            context_file.file,
                rfc.start_line,
                rfc.end_line,
                self.max_tokens_in_edit_prompt,
            )
        else:
            start_line, end_line, change_type = rfc.start_line, rfc.end_line, rfc.change_type

        logger.info(
            f"{self.trace_name} Requesting code change in {rfc.file_path} from {start_line} to {end_line}"
        )

        span_ids = []
        span_to_update = context_file.file.module.find_spans_by_line_numbers(
            start_line, end_line
        )
        if span_to_update:
            # Pin the spans that are planned to be updated to context
            for span in span_to_update:
                if span.span_id not in span_ids:
                    span_ids.append(span.span_id)
            self.file_context.add_spans_to_context(
                rfc.file_path, span_ids=set(span_ids), pinned=True
            )

        return StateOutcome.transition(
            trigger="edit_code",
            output={
                "instructions": rfc.instructions,
                "pseudo_code": rfc.pseudo_code,
                "file_path": rfc.file_path,
                "change_type": change_type.value,
                "start_line": start_line,
                "end_line": end_line,
                "span_ids": span_ids,
            },
        )

    def verify_request(
            self, rfc: RequestCodeChange, context_file: ContextFile
    ) -> Optional[StateOutcome]:

        if not self.tool_model:
            try:
                parser = PythonParser(apply_gpt_tweaks=True)
                module = parser.parse(rfc.pseudo_code, file_path=rfc.file_path)
            except Exception as e:
                return StateOutcome.retry(f"The pseude code syntax is invalid.")

            existing_hallucinated_spans = self.find_hallucinated_spans(
                rfc, module, context_file
            )
            if existing_hallucinated_spans:
                context_file.add_spans(existing_hallucinated_spans)
                return StateOutcome.retry(
                    f"There where code in the pseudo code that wasn't present in the file context. "
                    f"The following code spans where added to file context: {', '.join(existing_hallucinated_spans)}. "
                    f"Please provide instructions for the code change again."
                )

        if not rfc.start_line:
            message = """You must specify the start line and end line of the code change in the variables start_line and end_line. If you believe that the lines you want to edit isn't in the file context, you can request more context by providing the file path and the line numbers or span ids to the RequestMoreContext function.
            """
            return StateOutcome.retry(message)

        if not rfc.end_line:
            if rfc.change_type != ChangeType.addition:
                return StateOutcome.retry(
                    f"If your intention is to modify an existing code span you must provide the end line for the code change in end_line."
                )

            logger.info(
                f"{self.trace_name}:RequestCodeChange: End line not set, set to start line {rfc.start_line}"
            )
            rfc.end_line = rfc.start_line

        code_lines = context_file.file.content.split("\n")
        lines_to_edit = code_lines[rfc.start_line - 1: rfc.end_line]
        code_to_edit = "\n".join(lines_to_edit)

        tokens = count_tokens(code_to_edit)
        if tokens > self.max_tokens_in_edit_prompt:
            clarify_msg = (
                f"The code span between lines {rfc.start_line} - {rfc.end_line} has {tokens} tokens, which is higher than the "
                f"maximum allowed {self.max_tokens_in_edit_prompt} tokens. "
            )
            logger.info(f"{self.trace_name} {clarify_msg}. Ask for clarification.")
            return StateOutcome.retry(
                f"The change request was rejected! {clarify_msg}. Narrow down the instructions and specify the exact part of the code that needs to be updated to fulfill the change. "
            )

        return None

    def span_id_list(self, span_ids: list[str]) -> str:
        list_str = ""
        for span_id in span_ids:
            list_str += f" * {span_id}\n"
        return list_str

    def find_hallucinated_spans(
            self, rfc: RequestCodeChange, code_block: CodeBlock, context_file: ContextFile
    ) -> set[str]:
        existing_hallucinated_spans = set()
        for child_block in code_block.children:
            if child_block.type.group != CodeBlockTypeGroup.STRUCTURE:
                continue

            if child_block.type == CodeBlockType.CLASS:
                existing_hallucinated_spans.update(
                    self.find_hallucinated_spans(rfc, child_block, context_file)
                )

            span_id = child_block.belongs_to_span.span_id
            if context_file.has_span(span_id):
                continue

            existing_block = context_file.module.find_first_by_span_id(span_id)
            if existing_block:
                existing_hallucinated_spans.add(span_id)
            else:
                if "." not in span_id:
                    # Check if there is child blocks with the span_id as identifier
                    child_blocks = context_file.module.find_blocks_with_identifier(
                        span_id
                    )

                    for child_block in child_blocks:
                        if context_file.has_span(child_block.belongs_to_span.span_id):
                            continue

                        parent_block = child_block.find_type_group_in_parents(
                            CodeBlockTypeGroup.STRUCTURE
                        )
                        if (
                                parent_block
                                and parent_block.has_lines(rfc.start_line, rfc.end_line)
                        ) or child_block.is_within_lines(rfc.start_line, rfc.end_line):
                            existing_hallucinated_spans.add(
                                child_block.belongs_to_span.span_id
                            )

        return existing_hallucinated_spans

    def find_smallest_covering_block(self, code_block: CodeBlock, start_line: int, end_line: int) -> Optional[CodeBlock]:
        # If the code_block doesn't cover the lines, return None
        if code_block.start_line > start_line or code_block.end_line < end_line:
            return None

        # Check if any child block covers the lines
        for child in code_block.children:
            if child.start_line <= start_line and child.end_line >= end_line:
                # Found a smaller block that covers the lines
                smaller_block = self.find_smallest_covering_block(child, start_line, end_line)

                if child.type.group == CodeBlockTypeGroup.STRUCTURE:
                    return smaller_block or child

        # No smaller block found, return the current block
        return code_block

    def find_lines_within_blocks(self, code_block: CodeBlock, start_line: int, end_line: int) -> List[int]:
        # Collect lines from code blocks within max_tokens
        lines = []

        def traverse_blocks(block: CodeBlock):
            if block.end_line < start_line or block.start_line > end_line:
                return

            for child in block.children:
                traverse_blocks(child)

            # It's a code block within the line range
            if block.start_line >= start_line and block.end_line <= end_line:
                lines.extend(range(block.start_line, block.end_line + 1))

        traverse_blocks(code_block)
        return sorted(set(lines))

    def get_line_span(
            self,
            change_type: ChangeType,
            file: CodeFile,
            start_line: int,
            end_line: int,
            max_tokens: int,
    ) -> tuple[Optional[int], Optional[int], Optional[ChangeType]]:
        if not end_line:
            end_line = start_line

        structure_block = self.find_smallest_covering_block(file.module, start_line, end_line)
        if structure_block:
            logger.info(
                f"{self.trace_name} Found smallest covering block {structure_block.display_name} (start_line: {structure_block.start_line}, end_line: {structure_block.end_line}, tokens: {structure_block.sum_tokens()})"
            )

            if structure_block.type == CodeBlockType.CLASS:
                class_start_line, init_end_line, tokens = self.get_class_init_span(structure_block)

                if class_start_line <= start_line <= end_line <= init_end_line and tokens < max_tokens:
                    logger.info(
                        f"{self.trace_name} Return class init block {structure_block.display_name} (start_line: {class_start_line}, end_line: {init_end_line}, tokens: {tokens})"
                    )
                    return class_start_line, init_end_line, change_type

            if structure_block.sum_tokens() < max_tokens:
                logger.info(
                    f"{self.trace_name} Return block {structure_block.display_name} (start_line: {structure_block.start_line}, end_line: {structure_block.end_line}, tokens: {structure_block.sum_tokens()}"
                )

                return structure_block.start_line, structure_block.end_line, change_type

        lines = self.find_lines_within_blocks(file.module, max(0, start_line-5), min(file.module.end_line, end_line+5))
        if lines and len(lines) > 1:
            logger.info(
                f"{self.trace_name} Could not find start and end block for lines {start_line}-{end_line}. Return {lines[0]}-{lines[-1]}"
            )
            return lines[0], lines[-1], change_type
        else:
            logger.info(
                f"{self.trace_name} Could not find any lines within blocks for lines {start_line}-{end_line}. Returning original start and end lines."
            )
            return start_line, end_line, change_type

    def get_class_init_span(self, class_block: CodeBlock):
        """
        Get end line of the class initation span by including all lines until the first function or class
        """
        end_line = class_block.start_line + len(class_block.content_lines) - 1
        tokens = class_block.tokens
        for child in class_block.children:
            if (child.type.group == CodeBlockTypeGroup.STRUCTURE
                    and child.type != CodeBlockType.CONSTRUCTOR):
                break

            end_line = child.end_line
            tokens += child.tokens

        return class_block.start_line, end_line, tokens

    def _create_request_from_tool_call(self, action_request: ActionRequest, response: str):
        if isinstance(action_request.action, RequestCodeChange):
            response_split = response.split("RequestCodeChange")
            if len(response_split) > 1:
                response = response_split[1]

            action_request.action.instructions = response

        return action_request

    def function_call_system_prompt(self):
        return """You are an AI language model tasked with transforming unstructured messages wrapped in the XML tag <message> into structured tool calls. Your guidelines are:

 * Do not change, omit, or add any information from the original message.
 * Focus solely on converting the existing information into the correct tool call format.
 * Extract all relevant details necessary for the tool call without altering their meaning.
 * Just provide an empty string in the scratch_pad field 


Important rules for the RequestCodeChange function:
* You need to verify and include the correct line numbers based on the file context provided in the XML tag <file_context>.
* Just provide an empty string in instructions field.
* Just provide an empty string in pseudo code field
* Ignore planned steps in the tool call

Your response should be the tool call generated from the provided unstructured message, adhering strictly to these instructions."""

    def function_call_prompt(self, llm_response: str):
        file_context_str = self.file_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        content = ""
        if file_context_str:
            content += f"<file_context>\n{file_context_str}\n</file_context>\n\n"

        content = "<message>\n"
        content += llm_response
        content += "\n</message>"
        return content

    def system_prompt(self) -> str:
        if self.tool_model:
            return TOOL_MODEL_PROMPT

        if self.response_format == LLMResponseFormat.JSON:
            return PLAN_TO_CODE_SYSTEM_PROMPT + "\n\n" + FEW_SHOT_JSON

        return PLAN_TO_CODE_SYSTEM_PROMPT

    def verification_result(self, verbose: bool = True) -> str:
        if not self.verify:
            return ""

        if self.test_results:
            failure_count = sum(
                1 for issue in self.test_results if issue.status == TestStatus.FAILED
            )
            error_count = sum(
                1 for issue in self.test_results if issue.status == TestStatus.ERROR
            )

            passed_count = len(self.test_results) - failure_count - error_count

            response_msg = f"Ran {len(self.test_results)} tests. {passed_count} passed. {failure_count} failed. {error_count} errors."

            test_results = []
            for issue in self.test_results:
                if not issue.message or issue.status not in [
                    TestStatus.FAILED,
                    TestStatus.ERROR,
                ]:
                    continue

                attribues = ""
                if issue.file_path:
                    attribues += f"{issue.file_path}"

                    if issue.span_id:
                        attribues += f" {issue.span_id}"

                    if issue.line:
                        attribues += f", line: {issue.line}"

                if verbose:
                    test_results.append(
                        f"* {issue.status.value} {attribues}>\n```\n{issue.message}\n```\n"
                    )
                else:
                    last_line = issue.message.split("\n")[-1]
                    f"* {issue.status.value} {attribues}>\n```\n{last_line}\n```\n"

            if test_results:
                response_msg += "\n\n"
                response_msg += "\n".join(test_results)
        else:
            response_msg = "No tests were run"

        return response_msg

    def to_message(self, verbose: bool = True) -> str:
        response_msg = ""

        if self.message:
            response_msg += self.message

        if self.diff:
            response_msg += f"\n\n<diff>\n{self.diff}\n</diff>\n"

        if self.verify:
            response_msg += "\n\n<test_results>\n"
            response_msg += self.verification_result(verbose)
            response_msg += "</test_results>"

        return response_msg

    def create_message(self) -> str:
        previous_state = self.get_previous_state(self)
        content = ""
        if not previous_state:
            if self.initial_message:
                content += f"<issue>\n{self.initial_message}\n</issue>\n"

            file_context_str = self.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )

            content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"
        elif self.updated_file_context:
            logger.info(f"{self.trace_name} Updated file context: {self.updated_file_context}")
            file_context = self.workspace.create_file_context()
            file_context.add_files_with_spans(self.updated_file_context)

            for test_result in self.test_results:
                if test_result.status in [TestStatus.ERROR, TestStatus.FAILED] and test_result.file_path:
                    if test_result.span_id and not self.file_context.has_span(test_result.file_path,
                                                                              test_result.span_id):
                        file_context.add_span_to_context(test_result.file_path, test_result.span_id)
                        self.file_context.add_span_to_context(test_result.file_path, test_result.span_id)
                    elif test_result.line:
                        file_context.add_line_span_to_context(test_result.file_path, test_result.line, test_result.line)

            file_context_str = file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )

            content += f"\n\n<updated_file_context>\n{file_context_str}\n</updated_file_context>"

        content += self.to_message()

        return content

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        if self.initial_message:
            content = f"<issue>\n{self.initial_message}\n</issue>\n"
        else:
            content = ""

        previous_states = self.get_previous_states(self)

        for previous_state in previous_states:
            new_message = previous_state.to_message(verbose=False)
            if new_message and not content:
                content = new_message
            elif new_message:
                content += f"\n\n{new_message}"

            messages.append(UserMessage(content=content))

            if hasattr(previous_state.last_action.request, "action"):
                action = previous_state.last_action.request.action
            else:
                action = previous_state.last_action.request

            messages.append(AssistantMessage(action=action))
            content = ""

        content += self.to_message()

        file_context_str = self.file_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        messages.append(UserMessage(content=content))

        return messages
