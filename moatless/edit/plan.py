from enum import Enum
import json
import logging
from typing import Optional, List, Any, Union

from pydantic import ConfigDict, Field, PrivateAttr, BaseModel, model_validator

from moatless.codeblocks import CodeBlockType, PythonParser
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, CodeBlock
from moatless.edit.prompt import PLAN_TO_CODE_SYSTEM_PROMPT
from moatless.file_context import ContextFile
from moatless.index.code_index import is_test
from moatless.repository import CodeFile
from moatless.schema import VerificationIssueType, FileWithSpans, ChangeType, TestStatus
from moatless.state import (
    AgenticState,
    ActionRequest,
    StateOutcome,
    AssistantMessage,
    Message,
    UserMessage,
    TakeAction,
)
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
    start_line: int = Field(
        ..., description="The start line of the existing code to be updated."
    )
    end_line: int = Field(
        ...,
        description="The end line of the code to be updated when modifying existing code.",
    )

    pseudo_code: str = Field(..., description="Pseudo code for the code change.")
    file_path: str = Field(..., description="The file path of the code to be updated.")

    planned_steps: List[str] = Field(
        default_factory=list,
        description="Planned steps that should be executed after the current step.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_steps(cls, data: Any):
        # Claude sometimes returns steps as a string instead of a list
        if (
            isinstance(data, dict)
            and "planned_steps" in data
            and isinstance(data["planned_steps"], str)
        ):
            logger.info(
                f"validate_steps: Converting planned_steps to list: {data['planned_steps']}"
            )
            data["planned_steps"] = data["planned_steps"].split("\n")
        return data

    @model_validator(mode="before")
    def convert_status_to_enum(cls, values):
        if isinstance(values, dict) and isinstance(values.get("change_type"), str):
            values["change_type"] = ChangeType(values["change_type"])
        return values




class CodeRequest(BaseModel):
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


class RequestMoreContext(ActionRequest):
    """
    Request to see code that is not in the current context.
    """

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    files: List[CodeRequest] = Field(
        ..., description="The code that should be provided in the file context."
    )



class Review(ActionRequest):
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    review_instructions: str = Field(..., description="Review instructions.")

    @model_validator(mode="before")
    def convert_instructions(cls, values):
        if isinstance(values, dict) and values.get("instructions"):
            values["review_instructions"] = values["instructions"]
        return values


class Finish(ActionRequest):
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    finish_reason: str = Field(..., description="Finish the request and explain why")

    @model_validator(mode="before")
    def convert_reason(cls, values):
        # For backward compatibility
        if isinstance(values, dict) and values.get("reason"):
            values["finish_reason"] = values["reason"]
        return values


class Reject(ActionRequest):
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    rejection_reason: str = Field(
        ..., description="Reject the request and explain why."
    )

    @model_validator(mode="before")
    def convert_reason(cls, values):
        # For backward compatibility
        if values.get("reason"):
            values["rejection_reason"] = values["reason"]
        return values


class PlanRequest(TakeAction):
    """
    Request to apply a change to the code.
    """

    action: Union[RequestCodeChange, RequestMoreContext, Review, Finish, Reject]
    action_type: str = Field(..., description="The type of action being requested")

    @model_validator(mode="before")
    @classmethod
    def set_action_type(cls, values):
        if isinstance(values, dict) and "action" in values:
            values["action_type"] = type(values["action"]).__name__
        return values

    @classmethod
    def available_actions(cls) -> List[ActionRequest]:
        return [
            RequestCodeChange,
            RequestMoreContext,
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

    max_prompt_file_tokens: int = Field(
        4000,
        description="The maximum number of tokens in the file context to show in the prompt.",
    )

    max_tokens_in_edit_prompt: int = Field(
        1000,
        description="The maximum number of tokens in a span to show in the edit prompt.",
    )

    min_tokens_in_edit_prompt: int = Field(
        50,
        description="The minimum number of tokens in a span to show in the edit prompt.",
    )

    finish_on_review: bool = Field(
        False, description="Whether to finish the task if a review is requested."
    )

    max_repeated_test_failures: int = Field(
        5,
        description="The maximum number of repeated test failures before rejecting the task.",
    )

    max_repated_git_diffs: int = Field(
        2,
        description="The maximum number of repeated git diffs before rejecting the task.",
    )

    max_repeated_instructions: int = Field(
        2,
        description="The maximum number of repeated instructions before rejecting the task.",
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

        previous_states = self.get_previous_states(self)

        diff_counts = {}
        test_result_counts = {}
        for state in previous_states:
            if not state.diff:
                diff = "rejected"
            else:
                diff = state.diff

                if state.test_results:
                    issue_keys = []
                    for issue in state.test_results:
                        if issue.file_path:
                            issue_keys.append(
                                f"{issue.file_path}:{issue.span_id}:{issue.message}"
                            )
                        elif issue.message:
                            issue_keys.append(issue.message)

                    if issue_keys:
                        issue_key = "\n".join(issue_keys)
                        test_result_counts[issue_key] = (
                            test_result_counts.get(issue_key, 0) + 1
                        )

            diff_counts[diff] = diff_counts.get(diff, 0) + 1

        # Check if any diff exceeds the maximum allowed repetitions
        for diff, count in diff_counts.items():
            if count > self.max_repated_git_diffs:
                return StateOutcome.reject(
                    f"The following diff has been repeated {count} times, which exceeds the maximum allowed repetitions of {self.max_repated_git_diffs}:\n\n{diff}"
                )

        # Check if any verification issue exceeds the maximum allowed repetitions
        for issue_key, count in test_result_counts.items():
            if count > self.max_repeated_test_failures:
                return StateOutcome.reject(
                    f"The following verification issue has been repeated {count} times, which exceeds the maximum allowed repetitions of {self.max_repeated_test_failures}:\n\n{issue_key}"
                )

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
            elif not file_with_spans.span_ids:
                return StateOutcome.retry(
                    f"Please provide the line numbers or span ids for the code to add to context. Available span ids: {file.module.span_ids}"
                )

            missing_span_ids = set()
            suggested_span_ids = set()
            found_span_ids = set()
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
                        if (
                            block_span.initiating_block.sum_tokens()
                            < self.max_tokens_in_edit_prompt
                        ):
                            found_span_ids.add(block_span.span_id)
                            for child_span_id in block_span.initiating_block.span_ids:
                                found_span_ids.add(child_span_id)
                        else:
                            retry_message += f"Class {block_span.initiating_block.identifier} has too many tokens. Specify which functions to include..\n"
                            suggested_span_ids.update(
                                block_span.initiating_block.span_ids
                            )
                    else:
                        found_span_ids.add(block_span.span_id)

            if missing_span_ids:
                logger.info(
                    f"{self.trace_name}:RequestMoreContext: Spans not found in {file_with_spans.file_path}: {', '.join(missing_span_ids)}"
                )
                retry_message += f"Spans not found in {file_with_spans.file_path}: {', '.join(missing_span_ids)}\n"
            else:
                self.file_context.add_spans_to_context(file.file_path, found_span_ids)

            if retry_message and suggested_span_ids:
                logger.info(
                    f"{self.trace_name}:RequestMoreContext: Suggested spans: {', '.join(suggested_span_ids)}"
                )
                retry_message += f"Did you mean one of these spans: {', '.join(suggested_span_ids)}\n"

        if retry_message:
            return StateOutcome.send_message(retry_message)

        self.file_context.add_files_with_spans(action.files)

        message = "Added new spans:\n"
        for file in action.files:
            message += f" * {file.file_path} ({', '.join(file.span_ids)})\n"

        return StateOutcome.send_message(message)

    def _request_for_change(self, rfc: RequestCodeChange) -> StateOutcome:
        logger.info(
            f"{self.trace_name}:RequestCodeChange: file_path={rfc.file_path}, start_line={rfc.start_line}, end_line={rfc.end_line}, change_type={rfc.change_type}"
        )

        context_file = self.file_context.get_file(rfc.file_path)
        if not context_file:
            file = self.file_repo.get_file(rfc.file_path)
            if not file:
                logger.info(f"{self.trace_name}:RequestCodeChange: File {rfc.file_path} is not found in the file repository. Will create it and add to context.")
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

                return StateOutcome.retry(f"File {rfc.file_path} is not found in the file context. Use the RequestMoreContext function to add it and the relevant span ids. Available span ids: {span_ids}")

        retry_message = self.verify_request(rfc, context_file)
        if retry_message:
            return retry_message

        start_line, end_line, change_type = self.get_line_span(
            rfc.change_type,
            context_file.file,
            rfc.start_line,
            rfc.end_line,
            self.max_tokens_in_edit_prompt,
        )

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

        # Add the two most relevant test files to file context
        test_files_with_spans = self.workspace.code_index.find_test_files(
            rfc.file_path, query=rfc.pseudo_code, max_results=2, max_spans=1
        )
        logger.info(f"Found {len(test_files_with_spans)} test files with spans.")
        for test_file_with_spans in test_files_with_spans:
            if not self.file_context.has_file(test_file_with_spans.file_path):
                logger.info(f"Adding test file {test_file_with_spans.file_path} to context.")
                self.file_context.add_files_with_spans(test_files_with_spans)

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
        if not rfc.instructions:
            return StateOutcome.retry(
                f"Please provide instructions for the code change."
            )

        if not rfc.pseudo_code:
            return StateOutcome.retry(
                f"Please provide pseudo code for the code change."
            )

        if not context_file:
            logger.warning(
                f"{self.trace_name}:RequestCodeChange: File {rfc.file_path} is not found in the file context."
            )

            files_str = ""
            for file in self.file_context.files:
                files_str += f" * {file.file_path}\n"

            return StateOutcome.send_message(
                f"File {rfc.file_path} is not found in the file context. "
                f"You can only request changes to files that are in file context:\n{files_str}"
            )

        try:
            parser = PythonParser(apply_gpt_tweaks=True)
            module = parser.parse(rfc.pseudo_code, file_path=rfc.file_path)
        except Exception as e:
            return StateOutcome.send_message(f"The pseude code syntax is invalid.")

        existing_hallucinated_spans = self.find_hallucinated_spans(
            rfc, module, context_file
        )
        if existing_hallucinated_spans:
            context_file.add_spans(existing_hallucinated_spans)
            return StateOutcome.send_message(
                f"There where code in the pseudo code that wasn't present in the file context. "
                f"The following code spans where added to file context: {', '.join(existing_hallucinated_spans)}. "
                f"Please provide instructions for the code change again."
            )

        if not rfc.start_line:
            message = """You must specify the start line and end line of the code change in the variables start_line and end_line. If you believe that the lines you want to edit isn't in the file context, you can request more context by providing the file path and the line numbers or span ids to the RequestMoreContext function.
            """
            return StateOutcome.send_message(message)

        if not rfc.end_line:
            if rfc.change_type != ChangeType.addition:
                return StateOutcome.send_message(
                    f"If your intention is to modify an existing code span you must provide the end line for the code change in end_line."
                )

            logger.info(
                f"{self.trace_name}:RequestCodeChange: End line not set, set to start line {rfc.start_line}"
            )
            rfc.end_line = rfc.start_line

        previous_states = self.get_previous_states(self)
        instruction_counts = {}
        code_location = f"{rfc.file_path}:{rfc.start_line}-{rfc.end_line}"
        instruction_counts[code_location] = instruction_counts.get(code_location, 0) + 1

        for state in previous_states:
            if state.action_request and isinstance(
                state.action_request.action, RequestCodeChange
            ):
                code_location = f"{state.action_request.action.file_path}:{state.action_request.action.start_line}-{state.action_request.action.end_line}"
                instruction_counts[code_location] = (
                    instruction_counts.get(code_location, 0) + 1
                )

            # Check if any RFC exceeds the maximum allowed repetitions
            for code_location, count in instruction_counts.items():
                if count > self.max_repeated_instructions:
                    return StateOutcome.reject(
                        f"The same code was requested {count} times, which exceeds the maximum allowed repetitions of {self.max_repeated_test_failures}:\n\n{code_location}"
                    )

        code_lines = context_file.file.content.split("\n")
        lines_to_edit = code_lines[rfc.start_line - 1 : rfc.end_line]
        code_to_edit = "\n".join(lines_to_edit)

        tokens = count_tokens(code_to_edit)
        if tokens > self.max_tokens_in_edit_prompt:
            clarify_msg = (
                f"Lines {rfc.start_line} - {rfc.end_line} has {tokens} tokens, which is higher than the "
                f"maximum allowed {self.max_tokens_in_edit_prompt} tokens."
            )
            logger.info(f"{self.trace_name} {clarify_msg}. Ask for clarification.")
            return StateOutcome.send_message(
                f"{clarify_msg}. Narrow down the instructions and specify the exact part of the code that needs to be "
                f"updated to fulfill the change. "
            )

        return None

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

        start_block = file.module.find_first_by_start_line(start_line)

        if not start_block:
            structure_block = file.module
            logger.info(
                f"{self.trace_name} Start block not found, set module as structure block"
            )
        elif start_block.type.group == CodeBlockTypeGroup.STRUCTURE and (
            start_block.end_line >= end_line and start_block.start_line <= start_line
        ):
            structure_block = start_block
            logger.info(
                f"{self.trace_name} Start block {start_block.display_name} is a structure block"
            )
        else:
            structure_block = start_block.find_type_group_in_parents(
                CodeBlockTypeGroup.STRUCTURE
            )
            logger.info(
                f"{self.trace_name} Set parent {structure_block.display_name} as structure block"
            )

        if start_line == end_line and change_type.addition:
            start_line = _get_pre_start_line(start_line, structure_block)
            end_line = _get_post_end_line_index(end_line, structure_block)
            logger.info(
                f"{self.trace_name} Set start and end line to {start_line}, {end_line} for addition"
            )
            return start_line, end_line, change_type.addition

        structure_block_tokens = structure_block.sum_tokens()
        if (
            structure_block_tokens > self.min_tokens_in_edit_prompt
            and structure_block_tokens < max_tokens
        ):
            logger.info(
                f"{self.trace_name} Return start and endline for block {structure_block.display_name} "
                f"{structure_block.start_line} - {structure_block.end_line} ({self.min_tokens_in_edit_prompt} "
                f"(min tokens) < {structure_block_tokens} (block tokens) < {max_tokens} (max tokens))"
            )
            return structure_block.start_line, structure_block.end_line, change_type

        if structure_block_tokens < max_tokens:
            previous_block = structure_block.find_last_previous_block_with_block_group(
                CodeBlockTypeGroup.STRUCTURE
            )
            if (
                previous_block
                and structure_block_tokens + previous_block.sum_tokens() < max_tokens
            ):
                start_line = previous_block.start_line
                structure_block_tokens += previous_block.sum_tokens()
                logger.info(
                    f"{self.trace_name} Start from start line of the previous block {previous_block.display_name} that fits in the prompt"
                )
            else:
                start_line = structure_block.start_line

            next_structure_block = structure_block.find_next_block_with_block_group(
                CodeBlockTypeGroup.STRUCTURE
            )
            if (
                next_structure_block
                and structure_block_tokens + next_structure_block.sum_tokens()
                < max_tokens
            ):
                end_line = next_structure_block.end_line
                structure_block_tokens += next_structure_block.sum_tokens()
                logger.info(
                    f"{self.trace_name} End at end line of the next block {next_structure_block.display_name} that fits in the prompt, at line {end_line}"
                )
            else:
                end_line = structure_block.end_line

            logger.info(
                f"{self.trace_name} Return block [{structure_block.display_name}] ({start_line} - {end_line}) with {structure_block_tokens} tokens that covers the specified line span ({start_line} - {end_line})"
            )
            return start_line, end_line, change_type.modification

        if structure_block.end_line - end_line < 5:
            logger.info(
                f"{self.trace_name} Set structure block [{structure_block.display_name}] end line {structure_block.end_line} as it's {structure_block.end_line - end_line} lines from the end of the file"
            )
            end_line = structure_block.end_line
        else:
            end_line = _get_post_end_line_index(end_line, structure_block)
            logger.info(
                f"{self.trace_name} Set end line to {end_line}, structure block {structure_block.display_name} ends at line {structure_block.end_line}"
            )

        if start_line - structure_block.start_line < 5:
            logger.info(
                f"{self.trace_name} Set structure block [{structure_block.display_name}] start line {structure_block.start_line} as it's {start_line - structure_block.start_line} lines from the start of the file"
            )
            start_line = structure_block.start_line
        else:
            start_line = _get_pre_start_line(start_line, structure_block)

            logger.info(
                f"{self.trace_name} Set start line to {start_line}, structure block {structure_block.display_name} starts at line {structure_block.start_line}"
            )

        return start_line, end_line, change_type.modification

    def system_prompt(self) -> str:
        return PLAN_TO_CODE_SYSTEM_PROMPT

    def verification_result(self, verbose: bool = True) -> str:
        if not self.verify:
            return ""

        response_msg = "\n<test_results>\n"
        if self.test_results:
            failure_count = sum(
                1 for issue in self.test_results if issue.status == TestStatus.FAILED
            )
            if failure_count:
                response_msg += f"{failure_count} failed. "

            error_count = sum(
                1 for issue in self.test_results if issue.status == TestStatus.ERROR
            )
            if error_count:
                response_msg += f"{error_count} errors. "

            passed_count = len(self.test_results) - failure_count - error_count

            response_msg += f"Ran {len(self.test_results)} tests. {passed_count} passed. {failure_count} failed. {error_count} errors.\n"

            test_results = []
            for issue in self.test_results:
                if not issue.message or issue.status not in [
                    TestStatus.FAILED,
                    TestStatus.ERROR,
                ]:
                    continue

                attribues = ""
                if issue.file_path:
                    attribues += f"file={issue.file_path}"

                    if issue.span_id:
                        attribues += f" span_id={issue.span_id}"

                if issue.status == TestStatus.ERROR:
                    tag = "error"
                else:
                    tag = "failure"

                if verbose:
                    test_results.append(
                        f"<{tag} {attribues}>\n{issue.message}\n</{tag}>"
                    )
                else:
                    last_line = issue.message.split("\n")[-1]
                    test_results.append(f"<{tag} {attribues}>\n{last_line}\n</{tag}>")

            response_msg += "\n".join(test_results)
        else:
            response_msg += "No tests were run"

        response_msg += "\n</test_results>\n"
        return response_msg

    def to_message(self, verbose: bool = True) -> str:
        response_msg = ""

        if self.message:
            response_msg += self.message

        if self.diff:
            response_msg += f"\n\n<diff>\n{self.diff}\n</diff>\n"

        if self.verify:
            response_msg += self.verification_result(verbose)

        return response_msg

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


def _get_pre_start_line(
    start_line: int, structure_block: CodeBlock, max_lines: int = 6
) -> int:
    if start_line <= max_lines:
        return 1

    min_start_line = start_line - max_lines

    line_block = structure_block.find_last_by_end_line(start_line)
    print("hej", line_block, line_block.next, line_block.start_line, start_line)
    if line_block and line_block.next and line_block.next.start_line < start_line:
        return line_block.next.start_line
    elif line_block and line_block.start_line < start_line:
        return line_block.start_line
    else:
        return min_start_line


def _get_post_end_line_index(
    end_line: int, structure_block: CodeBlock, max_lines: int = 6
) -> int:
    max_end_line = end_line + max_lines

    line_block = structure_block.find_first_by_start_line(end_line + 1)
    if (
        line_block and line_block.end_line <= max_end_line
    ):  # Include next block after end_line
        return line_block.end_line
    elif (
        line_block and line_block.start_line + len(line_block.content_lines) > end_line
    ):
        return line_block.start_line + len(line_block.content_lines)
    else:
        return max_end_line
