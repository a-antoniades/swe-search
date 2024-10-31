import fnmatch
import logging
from typing import Optional, Any, Set

from pydantic import BaseModel, Field

from moatless.file_context import RankedFileSpan, ContextFile
from moatless.index.code_index import is_test
from moatless.state import AgenticState,ActionRequest, StateOutcome, AssistantMessage, Message, UserMessage
from moatless.schema import (
    FileWithSpans,
)
from moatless.utils.llm_utils import LLMResponseFormat

logger = logging.getLogger(__name__)


IDENTIFY_SYSTEM_PROMPT = """You are an autonomous AI assistant tasked with finding relevant code in an existing 
codebase based on a reported issue. Your task is to identify the relevant code spans in the provided search 
results and decide whether the search task is complete.

# Input Structure:

* <issue>: Contains the reported issue.
* <file_context>: Contains the context of already identified files and code spans.
* <search_results>: Contains the new search results with code divided into "code spans".

# Your Task:

1. Analyze the Reported Issue:
   Carefully read the issue provided within the <issue> tag to understand the problem.

2. Review Current File Context:
   Examine the existing file context in the <file_context> tag to understand which files and code spans have already been identified as relevant.

 3. Process New Search Results:
    3.1 Thoroughly analyze each code span in the <search_results> tag.
    3.2 Match code spans with key elements, functions, variables, or patterns identified in the reported issue.
    3.3 Evaluate the relevance of each code span based on how well it aligns with the reported issue and the current file context.
    3.4 If the issue suggests new functions or classes, identify existing code that might be relevant for implementing the new functionality.
    3.5 Review entire sections of code—not just isolated spans—to ensure you have a complete understanding before making a decision.
    3.6 Verify if there are references to other parts of the codebase that might be relevant but are not found in the search results.
    3.7 Identify any test code—including at least one test method—that can verify the solution for the reported issue.
    3.8 Extract and compile the relevant code spans and test code based on the reported issue.

 4. Respond Using the Identify Function:
    Use the Identify function to provide your response.
    
    * In the scratch_pad field, write out your step-by-step thoughts on how you identified the relevant code and why it is pertinent to the issue.
    * In the identified_spans field, list the files and code spans from the search results that are relevant to the reported issue, including any test code and ensuring that at least one test method is included.

*Note:* Be thorough and ensure that you have considered all possible relevant code spans and test code. Your goal is to assist in resolving the reported issue effectively by identifying all pertinent code, including at least one test method to verify the solution.
"""


IDENTIFY_FEW_SHOTS_JSON = """Example 1:

User: The application crashes with a "NullPointerException" when users try to generate a report without selecting a date range. This happens frequently and affects user experience.

Assistant:
```json
{
  "scratch_pad": "The NullPointerException suggests that the application expects a date range object that is null when not provided by the user. I should look for code spans where the report generation handles date ranges, specifically places where date range inputs are used without null checks.",
  "identified_spans": [
    {
      "file_path": "src/reports/ReportGenerator.java",
      "span_ids": ["ReportGenerator.generateReport"]
    },
    {
      "file_path": "src/utils/DateRangeValidator.java",
      "span_ids": ["DateRangeValidator.validate"]
    }
  ]
}
``` 

Example 2:
User: Users are unable to reset their passwords because the password reset email fails to send, showing a "SMTPAuthenticationError". This issue is critical and needs immediate attention.

Assistant:
Assistant:

```json
{
  "scratch_pad": "The SMTPAuthenticationError indicates a failure in authenticating with the SMTP server. I should find code spans related to sending emails, particularly the function that handles SMTP authentication during password reset.",
  "identified_spans": [
    {
      "file_path": "services/email_service.py",
      "span_ids": ["EmailService.send_password_reset_email"]
    },
    {
      "file_path": "config/email_config.py",
      "span_ids": ["email_settings"]
    },
    {
      "file_path": "tests/test_email_service.py",
      "span_ids": ["TestEmailService.test_send_password_reset_email"]
    }
  ]
}
```
"""

class Identify(ActionRequest):
    """Identify if the provided search result is relevant to the reported issue."""

    scratch_pad: str = Field(
        description="Your thoughts on how to identify the relevant code and why."
    )

    identified_spans: Optional[list[FileWithSpans]] = Field(
        default=None,
        description="Files and code spans in the search results identified as relevant to the reported issue.",
    )


class IdentifyCode(AgenticState):
    message: Optional[str] = Field(
        default=None,
        description="Message to display to the LLM.",
    )

    ranked_spans: Optional[list[RankedFileSpan]] = Field(
        default=None, description="Ranked file spans from the search results."
    )

    expand_context: bool = Field(
        default=True,
        description="Whether to expand the search result with relevant code spans.",
    )

    max_prompt_file_tokens: int = Field(
        default=4000,
        description="The maximum number of tokens to include in the prompt.",
    )

    find_test_files: bool = Field(
        default=True,
        description="Whether to find test files for identified spans.",
    )

    def model_dump(self, **kwargs):
        return super().model_dump(**kwargs)

    def _execute_action(self, action: Identify) -> StateOutcome:
        if action.identified_spans:
            updated_context = self.create_file_context()

            self.file_context.add_files_with_spans(action.identified_spans)
            updated_context.add_files_with_spans(action.identified_spans)

            span_count = sum([len(file.span_ids) for file in action.identified_spans])
            file_str = [file.file_path for file in action.identified_spans]
            logger.info(
                f"Identified {span_count} spans in the files {file_str}. Current file context size is {self.file_context.context_size()} tokens."
            )

            test_files = [file for file in self.file_context.files if is_test(file.file_path)]
            if self.find_test_files and not test_files:
                logger.info("The agent didn't select any tests. Will search and select test files based on identified files .")
                for file_with_spans in action.identified_spans:
                    span_query = " ".join([span_id for span_id in file_with_spans.span_ids])
                    test_files_with_spans = self.workspace.code_index.find_test_files(
                        file_with_spans.file_path, query=span_query, max_results=2, max_spans=2
                    )
                    for test_file_with_spans in test_files_with_spans:
                        if not self.file_context.has_file(test_file_with_spans.file_path):
                            logger.info(f"Add test file {test_file_with_spans}.")
                            self.file_context.add_files_with_spans([test_file_with_spans])
                            updated_context.add_files_with_spans([test_file_with_spans])
                logger.info(
                    f"File context size is {self.file_context.context_size()} tokens after adding tests."
                )

            # Always add the most relevant spans to context files with no selected spans
            for file in self.file_context.files:
                if len(file.span_ids - {"imports"}) == 0:
                    span_id = self._find_first_span(file)
                    if span_id:
                        logger.info(f"File {file.file_path} has no selected spans, will add {span_id}.")
                        file.add_span(span_id)
                        updated_context.add_span_to_context(file.file_path, span_id)
                    else:
                        logger.warning(f"File {file.file_path} has no selected spans and no ranked spans.")

            return StateOutcome.transition("finish", output={"updated_file_context": updated_context.to_files_with_spans()})
        else:
            logger.info("No spans identified.")

        message = f"The search returned {len(self.ranked_spans)} results. But unfortunately, I didn't find any of the search results relevant to the query."

        message += "\n\n"
        message += action.scratch_pad

        return StateOutcome.transition(
            "search",
            output={"message": message},
        )

    def _find_first_span(self, file: ContextFile):
        for ranked_span in self.ranked_spans:
            if ranked_span.file_path == file.file_path:
                return ranked_span.span_id

    def action_type(self) -> type[BaseModel] | None:
        return Identify

    def system_prompt(self) -> str:
        if self.tool_model:
            return IDENTIFY_SYSTEM_PROMPT + f"\n\n”Use the following format: {self.action_type().model_json_schema()}\n\n"

        if self.response_format == LLMResponseFormat.JSON:
            return IDENTIFY_SYSTEM_PROMPT + "\n\n" + IDENTIFY_FEW_SHOTS_JSON

        return IDENTIFY_SYSTEM_PROMPT

    def function_call_prompt(self, llm_response: str):
        content = self.create_message()
        content += "\n\n<message>\n"
        content += llm_response
        content += "\n</message>"
        return content

    def create_message(self) -> str:
        file_context = self.create_file_context(max_tokens=self.max_prompt_file_tokens)
        file_context.add_ranked_spans(self.ranked_spans)

        if file_context.files:

            if self.expand_context:
                file_context.expand_context_with_related_spans(
                    max_tokens=self.max_prompt_file_tokens, set_tokens=True
                )
                file_context.expand_classes(
                    max_tokens_per_class=self.max_prompt_file_tokens
                )

                logger.info(
                    f"File context size is {file_context.context_size()} tokens after expanding."
                )

            if self.find_test_files:
                for context_file in file_context.files:
                    span_query = " ".join([span_id for span_id in context_file.span_ids])

                    max_tokens = file_context.available_context_size()
                    if max_tokens < 0:
                        break

                    test_files_with_spans = self.workspace.code_index.find_test_files(
                        context_file.file_path, query=span_query, max_results=2, max_spans=3
                    )
                    file_context.add_files_with_spans(test_files_with_spans)

                logger.info(
                    f"File context size is {file_context.context_size()} tokens after adding tests."
                )

            search_result_str = file_context.create_prompt(
                show_span_ids=True,
                show_line_numbers=False,
                exclude_comments=True,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            logger.warning("No search results found.")
            search_result_str = "No new search results found."

        content = f"""<issue>
{self.initial_message}
</issue>

<extra_instructions>
{self.message}
</extra_instructions>

<search_results>
{search_result_str}
</search_results>
        """

        return content

    def messages(self) -> list[Message]:
        return [UserMessage(content=self.create_message())]


def is_test_pattern(file_pattern: str):
    test_patterns = ["test_*.py", "/tests/"]
    for pattern in test_patterns:
        if pattern in file_pattern:
            return True

    if file_pattern.startswith("test"):
        return True

    test_patterns = ["test_*.py"]

    return any(fnmatch.filter([file_pattern], pattern) for pattern in test_patterns)

if __name__ == "__main__":
    print(Identify.model_json_schema())