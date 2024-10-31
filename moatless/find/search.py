import fnmatch
import json
import logging
from typing import Optional

import instructor
from pydantic import BaseModel, Field, model_validator, ValidationError

from moatless.file_context import RankedFileSpan
from moatless.index.types import SearchCodeHit
from moatless.state import StateOutcome, AgenticState, ActionRequest, AssistantMessage, Message, UserMessage

from moatless.utils.llm_utils import LLMResponseFormat

logger = logging.getLogger(__name__)


SEARCH_SYSTEM_PROMPT = """You are an autonomous AI assistant with a superiour understanding of programming.
Your task is to locate the code relevant to an issue and provide instructions for code identification.

# Instructions:

1. Understand The Issue:
Read the <issue> tag to understand the issue.

2. Consider Suggested Files:
Review the <suggested_files> tag for any file suggestions related to the issue. These are not confirmed locations but potential areas to investigate.

3. Determine Necessary Search Parameters:
Based on the issue and suggested files, determine if specific file types, directories, function or class names, or code patterns should be part of the search.
Always try to specify the search parameters as accurately as possible.
You can formulate multiple search requests simultaneously to cover all possible relevant code.

4. Ensure At Least One Search Parameter:
Make sure that at least one of query, code_snippet, class_name, or function_name is provided in each search request.

5. Formulate the Search Function:
Set at least one of the search parameters: `query`, `code_snippet`, `class_name`, or `function_name`.

6. Provide Code Identification Instructions:
Include clear instructions for the AI agent that will identify the code in the search results. These instructions should guide the agent on:
- What specific code elements or patterns to look for
- How to prioritize results
- Any particular aspects of the code to focus on
- How to handle potential matches in the suggested files

"""


SEARCH_FUNCTIONS_FEW_SHOT_OPENAI_FUNC = """
Examples:

User: The file uploader intermittently fails with "TypeError: cannot unpack non-iterable NoneType object". This issue appears sporadically during high load conditions.

AI Assistant: functions.Search(
    scratch_pad: "The error indicates that a variable expected to be iterable is None, which might be happening due to race conditions or missing checks under high load. Investigate the file upload logic to ensure all necessary checks are in place and improve concurrency handling.",
    search_requests=[{
        query: "File upload process to fix intermittent 'TypeError: cannot unpack non-iterable NoneType object'",
        file_pattern: "**/uploader/**/*.py"
    }],
    code_identification_instructions: "Look for functions handling file uploads. Focus on areas where iterables are unpacked, especially under concurrent operations. Identify any missing null checks or race conditions that could lead to None values. Pay special attention to error handling and state management during the upload process."
)

User: There's a bug in the PaymentProcessor class where transactions sometimes fail to log correctly, resulting in missing transaction records.

AI Assistant: functions.Search(
    scratch_pad: "Missing transaction logs can cause significant issues in tracking payments. The problem may be related to how the logging mechanism handles transaction states or errors. Investigate the PaymentProcessor class, focusing on the transaction logging part.",
    search_requests=[{
        class_names: ["PaymentProcessor"]
    }],
    code_identification_instructions: "Within the PaymentProcessor class, focus on methods related to transaction processing and logging. Look for inconsistencies in error handling, asynchronous operations, or database interactions that might lead to missed log entries. Pay attention to any conditionals that determine when a transaction should be logged."
)

User: The generate_report function sometimes produces incomplete reports under certain conditions. This function is part of the reporting module. Locate the generate_report function in the reports directory to debug and fix the issue.

AI Assistant: functions.Search(
    scratch_pad: "Incomplete reports suggest that the function might be encountering edge cases or unhandled exceptions that disrupt the report generation. Reviewing the function's logic and error handling in the reporting module is necessary.",
    search_requests=[{
        function_names: ["generate_report"],
        file_pattern: "**/reports/**/*.py"
    }],
    code_identification_instructions: "Locate the generate_report function in the reports directory. Analyze the function's flow, paying special attention to loops, conditionals, and error handling that might cause premature termination. Look for any assumptions about data structures or input that might not always hold true. Identify any external dependencies or API calls that could fail silently."
)

User: The extract_data function in HTMLParser throws an "AttributeError: 'NoneType' object has no attribute 'find'" error when parsing certain HTML pages.

AI Assistant: functions.Search(
    scratch_pad: "The error occurs when 'find' is called on a NoneType object, suggesting that the HTML structure might not match expected patterns or that error handling for missing elements is insufficient.",
    search_requests:[{
        class_names: ["HTMLParser"],
        function_names: ["extract_data"]
    }],
    code_identification_instructions: "In the HTMLParser class, focus on the extract_data function. Look for calls to the 'find' method without proper null checks. Identify assumptions about HTML structure that might not always be true. Pay attention to error handling and how the function deals with unexpected or malformed HTML input."
)

User: The database connection setup is missing SSL configuration, causing insecure connections. Here's the stack trace of the error:
File "/opt/app/db_config/database.py", line 45, in setup_connection
    engine = create_engine(DATABASE_URL)
File "/opt/app/db_config/database.py", line 50, in <module>
    connection = setup_connection()

AI Assistant: functions.Search(
    scratch_pad: "The missing SSL configuration poses a security risk by allowing unencrypted connections. The issue is in the database connection setup, specifically in the create_engine call.",
    search_requests:[{
        code_snippet: "engine = create_engine(DATABASE_URL)",
        file_pattern: "db_config/database.py"
    }],
    code_identification_instructions: "Locate the setup_connection function in db_config/database.py. Focus on the line where create_engine is called with DATABASE_URL. Check if SSL parameters are missing in the DATABASE_URL or if there are any SSL-related arguments that should be passed to create_engine. Look for any configuration settings that might be overriding or preventing SSL setup. Also, examine how DATABASE_URL is constructed to ensure it includes necessary SSL parameters."
)
"""

SEARCH_FUNCTIONS_FEW_SHOT = """6. Execute the Search function:
Use the Search function with the search parameters and your thoughts on how to approach this task.

Think step by step and write out your thoughts in the scratch_pad field.

Examples:

User:
The file uploader intermittently fails with "TypeError: cannot unpack non-iterable NoneType object". This issue appears sporadically during high load conditions..

Search parameters:
    query: "File upload process to fix intermittent 'TypeError: cannot unpack non-iterable NoneType object'",
    file_pattern: "**/uploader/**/*.py"


User:
There's a bug in the PaymentProcessor class where transactions sometimes fail to log correctly, resulting in missing transaction records.

Search parameters:
    class_names: ["PaymentProcessor"]


User:
The generate_report function sometimes produces incomplete reports under certain conditions. This function is part of the reporting module. Locate the generate_report function in the reports directory to debug and fix the issue.

Search parameters:
    function_names: ["generate_report"]
    file_pattern: "**/reports/**/*.py"


User:
The extract_data function in HTMLParser throws an "AttributeError: 'NoneType' object has no attribute 'find'" error when parsing certain HTML pages.

Search parameters:
    class_names: ["HTMLParser"]
    function_names: ["extract_data"]


User:
The database connection setup is missing SSL configuration, causing insecure connections.

Here's the stack trace of the error:

File "/opt/app/db_config/database.py", line 45, in setup_connection
    engine = create_engine(DATABASE_URL)
File "/opt/app/db_config/database.py", line 50, in <module>
    connection = setup_connection()

Search parameters:
    code_snippet: "engine = create_engine(DATABASE_URL)",
    file_pattern: "db_config/database.py"

"""

SEARCH_JSON_FEW_SHOT = """6. Execute the Search:
Execute the search by providing the search parameters and your thoughts on how to approach this task in a JSON object. 

Think step by step and write out your thoughts in the scratch_pad field.

Examples:

User:
The file uploader intermittently fails with "TypeError: cannot unpack non-iterable NoneType object". This issue appears sporadically during high load conditions.

Assistant:
{
  "scratch_pad": "The error suggests that a variable expected to be iterable is None, likely due to a race condition or missing checks in the file uploader logic under high load conditions.",
  "search_requests": [
    {
      "file_pattern": "**/uploader/**/*.py",
      "query": "TypeError: cannot unpack non-iterable NoneType object",
      "code_snippet": null,
      "class_names": null,
      "function_names": null
    }
  ],
  "code_identification_instructions": "Look for places where unpacking is performed and verify checks for iterable variables."
}

User:
There's a bug in the PaymentProcessor class where transactions sometimes fail to log correctly, resulting in missing transaction records.
 
Assistant:
{
  "scratch_pad": "This issue is likely related to how the PaymentProcessor class logs transactions. It could be due to improper error handling or state management.",
  "search_requests": [
    {
      "file_pattern": null,
      "query": "transactions fail to log correctly",
      "code_snippet": null,
      "class_names": ["PaymentProcessor"],
      "function_names": null
    }
  ],
  "code_identification_instructions": "Investigate transaction logging mechanisms and error handling within the PaymentProcessor class."
}


User:
The extract_data function in HTMLParser throws an "AttributeError: 'NoneType' object has no attribute 'find'" error when parsing certain HTML pages.

Assistant:
{
  "scratch_pad": "This error indicates that the HTML structure does not match the expected pattern, leading to a NoneType error in the find function.",
  "search_requests": [
    {
      "file_pattern": null,
      "query": "AttributeError: 'NoneType' object has no attribute 'find'",
      "code_snippet": null,
      "class_names": ["HTMLParser"],
      "function_names": ["extract_data"]
    }
  ],
  "code_identification_instructions": "Check how HTMLParser is handling HTML structure and ensure proper checks before calling 'find' on objects."
}

User:
The database connection setup is missing SSL configuration, causing insecure connections.
 
Here's the stack trace of the error:

File "/opt/app/db_config/database.py", line 45, in setup_connection
    engine = create_engine(DATABASE_URL)
File "/opt/app/db_config/database.py", line 50, in <module>
    connection = setup_connection()

Assistant:
{
  "scratch_pad": "The missing SSL configuration introduces a security vulnerability, allowing unencrypted connections. Investigate the database connection setup to ensure SSL configuration is applied.",
  "search_requests": [
    {
      "file_pattern": "**/db_config/**/*.py",
      "query": null,
      "code_snippet": "engine = create_engine(DATABASE_URL)",
      "class_names": null,
      "function_names": null
    }
  ],
  "code_identification_instructions": "Locate where the database connection is being configured and ensure SSL parameters are applied."
}
"""

IGNORE_TEST_PROMPT = (
    "Test files are not in the search scope. Ignore requests to search for tests. "
)


class SearchRequest(BaseModel):
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    query: Optional[str] = Field(
        default=None,
        description="A semantic similarity search query. Use natural language to describe what you are looking for.",
    )

    code_snippet: Optional[str] = Field(
        default=None,
        description="Specific code snippet to that should be exactly matched.",
    )

    class_names: Optional[list[str]] = Field(
        default=None,
        description="Specific class names to include in the search."
    )

    function_names: Optional[list[str]] = Field(
        default=None,
        description="Specific function names to include in the search."
    )

    def has_search_attributes(self):
        return any(
            [
                self.query,
                self.code_snippet,
                self.class_names,
                self.function_names,
            ]
        )

    @model_validator(mode="after")
    def validate_search_requests(self):
        if not self.has_search_attributes:
            raise ValueError("A search request must have at least one attribute set.")
        return self

    @property
    def log_name(self):
        search_parameters = []
        if self.query:
            search_parameters.append(f"query: {self.query[:20]}...")

        if self.code_snippet:
            search_parameters.append(f"code_snippet: {self.code_snippet[:20]}...")

        if self.class_names:
            search_parameters.append(f"class_names: {','.join(self.class_names)}")

        if self.function_names:
            search_parameters.append(f"function_names: {','.join(self.function_names)}")

        return ", ".join(search_parameters)


class Search(ActionRequest):
    """Take action to search for code, identify found and finish up."""

    scratch_pad: str = Field(
        description="Scratch pad for the search. Use this to write down your thoughts on how to approach the search."
    )

    search_requests: list[SearchRequest] = Field(
        description="List of search requests.",
    )

    code_identification_instructions: Optional[str] = Field(
        None,
        description="Instructions for the AI agent on how to identify the code in the search results.",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_direct_search_attributes(cls, values):
        search_request_fields = [
            "file_pattern",
            "query",
            "code_snippet",
            "class_names",
            "function_names",
        ]
        direct_search_attrs = {
            field: values.pop(field)
            for field in search_request_fields
            if field in values
        }

        if any(direct_search_attrs.values()):
            logger.warning(
                f"The following direct search attributes were found: {direct_search_attrs}, will set as SearchRequest"
            )

            if "search_requests" in values:
                values["search_requests"].append(direct_search_attrs)
            else:
                values["search_requests"] = [direct_search_attrs]

        return values

    @model_validator(mode="after")
    def validate_search_requests(self):
        if not self.search_requests:
            logger.warning(
                f"No search requests found in the search action request."
            )
            raise ValueError("At least one search request must exist.")
        return self

    @property
    def log_name(self):
        logs = []
        for i, search_request in enumerate(self.search_requests):
            logs.append(f"{i}=[{search_request.log_name}]")
        return f"Search(" + ", ".join(logs) + ")"

    def to_prompt(self):
        content = "Search for code with the following parameters:\n"
        for i, request in enumerate(self.search_requests):
            if len(self.search_requests) > 1:
                content += f"\nSearch Request {i + 1}:\n"
            if request.file_pattern:
                content += f"File pattern: {request.file_pattern}\n"
            if request.query:
                content += f"Query: {request.query}\n"
            if request.code_snippet:
                content += f"Code snippet: {request.code_snippet}\n"
            if request.class_names:
                content += f"Class names: {', '.join(request.class_names)}\n"
            if request.function_names:
                content += f"Function names: {', '.join(request.function_names)}\n"
        return content.strip()


class SearchCode(AgenticState):
    message: Optional[str] = Field(
        None,
        description="Message to the search",
    )

    max_search_results: int = Field(
        75,
        description="The maximum number of search results.",
    )

    max_retries_with_any_file_context: int = Field(
        3,
        description="The maximum number of retries when there are identified files in file context.",
    )

    include_message_history: bool = Field(
        True,
        description="Include message history from previous iterations",
    )

    provide_initial_context: bool = True
    initial_context_tokens: int = 6000
    initial_search_results: int = 100
    initial_context_spans_per_file: int = 5

    support_test_files: bool = True

    def _execute_action(self, action: Search) -> StateOutcome:
        if isinstance(action, Search):
            for request in action.search_requests:
                if (
                    not self.support_test_files
                    and request.file_pattern
                    and is_test_pattern(request.file_pattern)
                ):
                    return self._retry("It's not possible to search for test files.")

        message = ""
        search_result: list[SearchCodeHit] = []
        for search_request in action.search_requests:
            search_response = self.workspace.code_index.search(
                file_pattern=search_request.file_pattern,
                query=search_request.query,
                code_snippet=search_request.code_snippet,
                class_names=search_request.class_names,
                function_names=search_request.function_names,
                max_results=int(self.max_search_results / len(action.search_requests)),
            )
            search_result.extend(search_response.hits)
            message += "\n" + search_response.message

        logger.info(f"Found {len(search_result)} hits.")

        ranked_spans = []
        for hit in search_result:
            for span in hit.spans:
                ranked_spans.append(
                    RankedFileSpan(
                        file_path=hit.file_path,
                        span_id=span.span_id,
                        rank=span.rank,
                        tokens=span.tokens,
                    )
                )

        if len(ranked_spans) == 0:
            logger.info("No search results found. Will retry.")
            message = "\n\nUnfortunately, I didn't find any relevant results."
            return self._retry(message)

        return StateOutcome.transition(
            trigger="did_search",
            output={
                "ranked_spans": ranked_spans,
                "message": action.code_identification_instructions
            },
        )

    def _retry(self, message: str) -> StateOutcome:
        if (
            self.retries() > self.max_retries_with_any_file_context
            and self.file_context.files
        ):
            logger.info(
                "Exceeded max retries, will finish as there are identified files in the file context. Transitioning to finish."
            )
            return StateOutcome.transition("finish")
        else:
            return StateOutcome.retry(message)

    def action_type(self) -> type[BaseModel] | None:
        return Search

    def system_prompt(self) -> str:
        system_prompt = SEARCH_SYSTEM_PROMPT

        if self.tool_model:
            system_prompt += f"\n\nUse the following format: {self.action_type().model_json_schema()}\n\n"
            system_prompt += SEARCH_FUNCTIONS_FEW_SHOT
            return system_prompt

        instructor_mode = self.response_format
        if instructor_mode == LLMResponseFormat.JSON:
            system_prompt += SEARCH_JSON_FEW_SHOT
        elif self.model.startswith("gpt"):
            system_prompt += SEARCH_FUNCTIONS_FEW_SHOT_OPENAI_FUNC
        else:
            system_prompt += SEARCH_FUNCTIONS_FEW_SHOT

        if not self.support_test_files:
            system_prompt += IGNORE_TEST_PROMPT
        return system_prompt

    def create_message(self):
        content = f"<issue>\n{self.initial_message}\n</issue>"

        if self.provide_initial_context and self.initial_message:
            logger.info("Search for initial context to provide in the prompt")

            result = self.workspace.code_index.semantic_search(
                query=self.initial_message,
                exact_match_if_possible=False,
                max_spans_per_file=5,
                max_results=self.max_search_results,
            )

            file_context = self.create_file_context(max_tokens=4000)

            for hit in result.hits:
                for span in hit.spans:
                    file_context.add_span_to_context(
                        hit.file_path, span.span_id, tokens=1
                    )

            content += "\n\nHere's some files that might be relevant when formulating the search.\n"
            content += file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=False,
                exclude_comments=True,
                show_outcommented_code=False,
            )

        if self.message:
            content += f"\n\n{self.message}\n"

        if self.file_context.files:
            file_context_str = self.file_context.create_prompt(
                exclude_comments=True,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            file_context_str = "No files found yet."

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        return content

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        content = f"<issue>\n{self.initial_message}\n</issue>"

        if self.provide_initial_context and self.initial_message:
            logger.info("Search for initial context to provide in the prompt")

            result = self.workspace.code_index.semantic_search(
                query=self.initial_message,
                exact_match_if_possible=False,
                max_spans_per_file=5,
                max_results=self.max_search_results,
            )

            file_context = self.create_file_context(max_tokens=4000)

            for hit in result.hits:
                for span in hit.spans:
                    file_context.add_span_to_context(
                        hit.file_path, span.span_id, tokens=1
                    )

            content += "\n\nHere's some files that might be relevant when formulating the search.\n"
            content += file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=False,
                exclude_comments=True,
                show_outcommented_code=False,
            )

        previous_states = self.get_previous_states(self)
        for previous_state in previous_states:
            if previous_state.message:
                content += previous_state.message
            messages.append(UserMessage(content=content))
            messages.append(
                AssistantMessage(
                    action=previous_state.last_action.request,
                )
            )
            content = ""

        if self.message:
            content += f"\n\n{self.message}\n"

        if self.file_context.files:
            file_context_str = self.file_context.create_prompt(
                exclude_comments=True,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            file_context_str = "No files found yet."

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        messages.append(UserMessage(content=content))
        return messages


def is_test_pattern(file_pattern: str):
    test_patterns = ["test_*.py", "/tests/"]
    for pattern in test_patterns:
        if pattern in file_pattern:
            return True

    if file_pattern.startswith("test"):
        return True

    test_patterns = ["test_*.py"]

    return any(fnmatch.filter([file_pattern], pattern) for pattern in test_patterns)
