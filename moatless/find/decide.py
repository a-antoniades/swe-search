import logging
from typing import Optional

from pydantic import BaseModel, Field

from moatless.find import SearchCode
from moatless.state import AgenticState, ActionRequest, StateOutcome, AssistantMessage, Message, UserMessage


logger = logging.getLogger(__name__)


MAYBE_FINISH_SYSTEM_PROMPT = """You will be provided a reported issue and the file context containing existing code from the project's git repository. 
Your task is to make a decision if the code related to a reported issue is provided in the file context. 

# Input Structure:

* <issue>: Contains the reported issue.
* <file_context>: The file context.

Instructions:

 * Analyze the Issue:
   * Review the reported issue to understand what functionality or bug fix is being requested.

 * Analyze File Context:
  * Examine the provided file context to identify if the relevant code for the reported issue is present.
  * If the issue suggests that code should be implemented and doesn't yet exist in the code, consider the task completed if relevant code is found that would be modified to implement the new functionality.
  * If relevant code in the file context points to other parts of the codebase not included, note these references.

 * Make a Decision:
  * Decide if the relevant code is found in the file context.
  * If you believe all existing relevant code is identified, mark the task as complete.
  * If the specific method or code required to fix the issue is not present, still mark the task as complete as long as the relevant class or area for modification is identified.
  * If you believe more relevant code can be identified, mark the task as not complete and provide your suggestions on how to find the relevant code.

Important:
 * You CANNOT change the codebase. DO NOT modify or suggest changes to any code.
 * Your task is ONLY to determine if the file context is complete. Do not go beyond this scope.
"""


class Decision(ActionRequest):
    """Provide your decision if all relevant file context is provided."""

    scratch_pad: str = Field(
        description="Your thoughts on if the spans where relevant or not and if you found all relevant spans and can finish.."
    )

    relevant: bool = Field(
        default=False,
        description="Set to true if the relevant code have been identified.",
    )

    complete: bool = Field(
        default=False,
        description="Set to true if all the relevant code have been identified.",
    )

    search_suggestions: Optional[str] = Field(
        None,
        description="Suggestions on how to find the relevant code not found in the file context.",
    )


class DecideRelevance(AgenticState):
    expand_context_when_complete: bool = Field(
        True,
        description="If true, the file context will be expanded with additional context after complete.",
    )
    finish_after_relevant_count: int = Field(
        2,
        description="Finish the task after this many relevant decisions have been made but not complete.",
    )

    max_prompt_file_tokens: int = Field(
        4000,
        description="The maximum number of tokens to include in the file context prompt.",
    )

    def _execute_action(self, action: Decision) -> StateOutcome:
        if action.complete and action.relevant:
            return StateOutcome.transition("finish")

        if (
            action.relevant
            and self._relevant_count() >= self.finish_after_relevant_count
        ):
            return StateOutcome.transition("finish")

        return StateOutcome.transition(
            "search",
            output={"message": action.search_suggestions},
        )

    def _finish(self):
        if self.expand_context_when_complete:
            self.file_context.expand_context_with_related_spans(
                max_tokens=self.max_prompt_file_tokens
            )
        return StateOutcome.finish()

    def _relevant_count(self) -> int:
        """
        Count the number of times a decision was made that the file context was relevant.
        """
        relevant_count = 0
        previous_states = self.get_previous_states(self)
        for previous_state in previous_states:
            if (
                previous_state.last_action
                and previous_state.last_action.request.relevant
            ):
                relevant_count += 1
        return relevant_count

    def action_type(self) -> type[BaseModel] | None:
        return Decision

    def system_prompt(self) -> str:
        return MAYBE_FINISH_SYSTEM_PROMPT

    def _last_scratch_pad(self):
        previous_states = self.get_previous_states()
        if previous_states and previous_states[-1].last_action:
            last_action = previous_states[-1].last_action
            return last_action.request.scratch_pad
        else:
            return None

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        file_context_str = self.file_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=False,
            exclude_comments=True,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        content = f"""<issue>
{self.initial_message}
</issue>
"""

        scratch_pad = self._last_scratch_pad()
        if scratch_pad:
            content += f"""<scratch_pad>
{scratch_pad}
</scratch_pad>"""

        content += f"""
<file_context>
{file_context_str}
</file_context>
"""

        messages.append(UserMessage(content=content))
        return messages