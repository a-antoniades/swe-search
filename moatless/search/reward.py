import sys

from ..trajectory import Trajectory, TrajectoryState

sys.path.append("..")
sys.path.append("...")

import collections
import json
import logging
# import json5
import os
import re
# from retry import retry
import time
from datetime import datetime
from functools import partial
from typing import List, Optional, Tuple
import socket

import litellm
from litellm import ModelResponse, token_counter
from openai import OpenAI
from tqdm import tqdm

from moatless.repository.git import GitRepository
from moatless.schema import Completion, ValueFunctionResult, TestStatus
from moatless.search.models import (
    LOCAL_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
    GROQ_MODELS,
    ANTHROPIC_MODELS,
)
from .prompt import (
    VALUE_FUNCTION_SEARCH_PROMPT,
    VALUE_FUNCTION_FINISH_PROMPT,
    VALUE_FUNCTION_REQUEST_MORE_CONTEXT_PROMPT
)
from ..edit.plan_v2 import RequestMoreContext
from ..find import SearchCode
from ..state import AgenticState, Pending, State
from ..utils_search.misc import save_to_json

# litellm.set_verbose = True

logger = logging.getLogger('mcts.reward')
# litellm.set_verbose=True



# completion_response = get_client().chat.completions.create(
#         model=self.model,
#         messages=messages,
#         temperature=self.temperature,
#         max_tokens=self.max_tokens,
#         timeout=150
#     
# Qwen/Qwen2-7B-Instruct
# MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
# meta-llama/Meta-Llama-3-8B
# "Qwen/Qwen2-7B-Instruct" "Qwen/Qwen2-72B-Instruct" "Qwen/Qwen2-57B-A14B-Instruct"
# MODEL = "deepseek-ai/deepseek-coder-33b-instruct"
MODEL = "gpt-4o-mini-2024-07-18"    # "Qwen/Qwen2-7B-Instruct"
# MODEL = "gpt-4o-2024-08-06"
# MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

CLAUDE_MODELS = []
PORT = LOCAL_MODELS.get(MODEL, None)
API_KEY = "token-abc123"
BASE_URL = f"http://localhost:{PORT}/v1"
# BASE_URL = "http://localhost:8000"

STATE_OPTIONS = None
EXPLANATION_OPTIONS = None
ERROR = [True, False]
CLASSIFICATION = ["Found but not identified",
                  "Alternative identification",
                  "Failure to identify expected changes",
                  "Incomplete planning",
                  "Unnecessary edits",
                  "Unresolved issues",
                  "False positives",]

ROLE_PROMPT_VALUE = "You are an intelligent AI assistant and judge"

MAIN_OBJECTIVE_PROMPT_VALUE = """Your main objective is to assign a value to the executed state, according to it's likelihood of being the best possible path to take to resolve the issue.

The agent is trying to solve a programming issue by traversing a search tree. The agent starts at the root node, which represents the initial issue, and searches for the file(s) that need to be modified to resolve the issue. The agent then evaluates the states of the search tree to determine the best path to take. To address the issue.

There are many different functions that an agent may take at a given step. Search, Plan, Edit etc. Your task is to understand wether the state and the particular change it causes is the correct one at that certain stage, given the initial issue and the previous steps taken by the agent, and the results of the action.

Scrutinize the generated answers critically, and try to find any mistakes or potential weaknesses, by thinking about the reasoning behind the answer, and the context in which it was taken, and potential alternatives. If a mistake was made, it is paramount to understand exactly at which step the mistake was made, and why it was made. This will help us to correct the agent's behavior and improve its performance in the future.
Here's some guidelines to help you understand how to approach the evaluation of different states:

When in a Search state, was the relevant code present within the code spans returned by the search?
When in an Identify state, was the relevant code spans (if present in the context) correctly identified?
When in a plan state, were the correct code spans identified for modification, and is the proposed modification correct?
When in an edit state, was the modification correct and does it resolve the issue?

"""


STATE_EVALUATION_GUIDELINES = """When evaluating different states in the problem-solving process, consider the following criteria and be aware of *Common failure cases*:

1. Search State
   Criterion: Was the relevant code present within the code spans returned by the search?
   *common mistakes*:
   - Not Found: The relevant code spans were not present in the search results.

2. Identify State
   Criterion: Were the relevant code spans (if present in the context) correctly identified?
   *common mistakes*:
   - Found but not identified: The relevant code may be present in search results but not recognized by the agent.

3. Plan State
   Criteria: 
   a) Were the correct code spans identified for modification?
   b) Is the proposed plan for modification correct?
   *common mistakes*:
   - Incomplete planning: The planning state may create a task to modify code, but execution fails.
   - Incorrect plan: The plan may be incorrect, or not does not result in the best diff to resolve the issue.

4. Edit State
   Criteria:
   a) Was the modification necessary and correct?
   b) Does it resolve the issue?
   c) Was the modification succesfully applied?
   *common mistakes*:
   - Unnecessary edits: The agent may modify code that doesn't need changing or make incorrect modifications.
   - Unresolved issues: A significant portion of cases may result in failure, indicating unresolved problems / edge cases.
   - False positives: Some cases may be incorrectly identified as resolved when they are not.

General Guidelines:
1. Be vigilant for these common mistakes and provide accurate evaluations to guide the agent effectively.
2. Evaluate the current step within the context of the entire problem-solving process, considering the impact of the decision on subsequent actions.
3. Provide specific feedback on what went wrong if a state evaluation is negative.
4. Suggest potential improvements or alternative approaches when applicable.
5. Tests are not in scope. Do not suggest executing, searching for or writing tests.
6. Pay attention to every detail of the proposed action and diff, as an action/diff that may on the surface seem correct, may in fact be incorrect due to a small detail.
7. The <Reward> should be proportional to your certainy that the action taken was the best one possible and will lead to a successful resolution of the issue.

Remember: Accurate state evaluation is crucial for guiding the agent towards effective problem-solving and continuous improvement."""

COMMON_MISTAKES_PROMPT = """When evaluating states, be aware of these *common mistakes* and failure cases:

 * Found but not identified: The relevant code may be present in search results but not recognized by the agent.
 * Incomplete planning: The planning state may create a task to modify code, but execution fails.
 * Incorrect edits: The agent may modify code that doesn't need changing or make incorrect modifications.
 * Unresolved issues: A significant portion of cases result in failure, indicating unresolved problems.
 * False positives: Some cases are incorrectly identified as resolved when they are not.

Be vigilant for these common mistakes and provide accurate evaluations to guide the agent effectively."""

FAILURE_CASES_PROMPT = """Below are the most **common cases** that you should be aware of when evaluating the agent's actions:

    - Not Found: the relevant code was not found in the search results (<Search> states).
    - Found but not identified: the relevant code may be present in the search results but not identified by the agent (<Identify> states).
    - Incomplete Planning: although the correct file that needs to be addressed was identified, the planning state did not create a task to address the relevant patch of code (<Plan> states).
    - Incorrect Edit: although the agent identified the correct patch of code, the changes it made to it are incorrect, which will lead to a failure to resolve the issue (<Edit> states).
"""

DEFINITION_PROMPT = """
<problem_statement>: "The initial message or problem statement.
<executed_state> The state that was executed by the agent that should be assigned a reward value.
<next_state>: The next upcoming state generated from the outcome of the executed state.
<state_history> The trajectory of state transitions prior to the executed state. Pay careful attention to these to understand wether the context within which the executed state and responses are being taken make sense.
<state_file_context> The context available to the agent in the current state. Contains information about the files and code snippets.
<diff> The git diff of the changes made until the next state.
"""


DEFINITIONS = {
    "problem_statement": "The initial message or problem statement.",
    "next_state": "The next upcoming state generated from the outcome of the executed state.",
    "executed_state": "The state that was executed by the agent that should be assigned a reward value.",
    "state_history": "The trajectory of state transitions prior to the executed state. Pay careful attention to these to understand wether the context within which the executed state and responses are being taken make sense.",
    "state_info": "Additional information about the executed state.",
    "step_count": "The current step number in the trajectory.",
    "state_message": "The action or message in the current state.",
    "state_file_context": "The context available to the agent in the current state. Contains information about the files and code snippets.",
    "diff": "The git diff of the changes made until the current state.",
    "state_response": "The action taken by the agent in response to the current state.",
    "action": "The action take by the agent that has led to the corresponding state.",
    "next_state_message": "The message or action for the next state, if available."
}


VARS_MINIMIZE = [
    "nothing_for_now_bruh"
]

VALUE_OUTPUT_FORMAT = """OUTPUT FORMAT:

<Explanation>: 2-3 sentences explaining the the reasoning in your decision, alluding to the *common mistakes* where appropriate.
<Reward>: integer reward (range: -100 to 100)."""

VALUE_ASSIGNMENT_PROMPT = f"""Please assign a value to the provided state of the search tree. Keep in mind the following definitions:

<initial_state_message>: The initial issue that we are trying to solve.
<state_info>: The information about the current state the node we are analyzing is in.
<state_history>: The trajectory of state transitions prior to the current state. Pay careful attention to these to understand wether the context within which the current state and responses are being taken make sense.
<state_file_context>: The context available to the agent in the current state. Contains information about the files and code snippets.
<diff>: The git diff of the changes made until the current state.

Your value assesment should be a single integer value to the state from -100 to 100. Always strictly adhere to this output format and don't forget to provide both <Explanation> and <Reward>: 

{VALUE_OUTPUT_FORMAT}"""

ROLE_PROMPT_TREE = "You are an expert software engineer tasked with evaluating and guiding an AI agent's actions in solving a reported software issue. "

ISSUE_NAME_TREE = "problem_statement"

CONCLUSION_PROMPT = """Based on the initial problem context and the answers from the debate of the other agents, construct an optimal answer.
Consider the different perspectives presented by the agents and the context provided in order to reach the correct conclusion.
Do not refer to the participants, but rather just report your recommendations as if they were your own.
Strictly adhere to any output format used in the Agent responses, and especially any tool/api/function calls if present, like those enclosed for example those enclosed in angle brackets i.e <tool_call> or **value**.
"""

STAGE_NAME = "<State>"

MAIN_OBJECTIVE_PROMPT_TREE = f"""You will be presented with a "Search Tree" that represents the agent's decision-making process and actions taken to address the {ISSUE_NAME_TREE}. 

The Search Tree structure:

Each node in the tree represents a state of the software project and the action taken by the agent.
The tree starts with an initial state (root node) and branches out as the agent makes decisions.


Your main objectives are to:


1. Determine if the current finished state is likely to produce the correct solution to the {ISSUE_NAME_TREE}.

2. If the finished state is unlikely to solve the issue:
   a. Identify the initial stage {STAGE_NAME} in the trajectory where you believe a mistake or suboptimal decision was made.
   b. Provide clear feedback explaining why you think this was a pivotal point of error.


{FAILURE_CASES_PROMPT}


To accomplish these objectives effectively, follow these comprehensive guidelines:

* Carefully analyze the {ISSUE_NAME_TREE} to fully understand the problem that needs to be solved.

* Evaluate the effectiveness of the agent's approach by considering:
   - If each action contributes to addressing the {ISSUE_NAME_TREE} correctly
   - The impact of the decision on subsequent actions
   - If any action contains a mistake that leads to a wrong solution

* In your feedback, address:
   - Specific actions or decisions that led to the error
   - Potential consequences of the mistake on the overall solution
   - Recommendations for correcting the approach from that point

* Provide concise, actionable insights on how the agent could improve its problem-solving approach, focusing on:
   - Key areas where the agent should focus more attention
   - Strategies for more effective decision-making in similar situations
   - Suggestions for better aligning actions with the {ISSUE_NAME_TREE}

Your evaluation and guidance will be crucial in improving the AI agent's performance in addressing software issues efficiently and effectively. Offer clear, concise, and actionable insights based on your expert analysis of the Search Tree and the {ISSUE_NAME_TREE}.


Provide your answer in the exact format below:

{STAGE_NAME}: If an error is found, the state in the search tree that the initial error was identified, else reply with "not applicable".
<Explanation>: Provide an explanation.
<Classification>: Classify the type of error according to **common cases** provided above if there is one, else reply with "Resolved".
<Error>: (bool) True/False. Wether an error state/action was identified.
<Suggestion>: If an error is found, provide a suggestion that is specific to the identified {STAGE_NAME}, to help the agent correct its mistake from that point onwards. 
"""

# <next_state_message>: The next input message that was constructed using the next state created using <state_response> instructions.


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_model_port(model_name, start_port=8000, end_port=9000):
    for port in range(start_port, end_port + 1):
        if is_port_in_use(port):
            # Try to make a request to the API endpoint
            try:
                client = OpenAI(api_key="token-abc123", base_url=f"http://localhost:{port}/v1")
                response = client.models.list()
                for model in response.data:
                    if model.id == model_name:
                        return port
            except Exception:
                # If we can't connect or the model isn't found, continue to the next port
                continue
    return None

def configure_api_base(model):
    if model in LOCAL_MODELS:
        port = find_model_port(model)
        if port:
            return f"http://localhost:{port}/v1"
        else:
            raise ValueError(f"Could not find a running instance of model {model}")
    elif model.startswith("openai"):
        return os.getenv("CUSTOM_LLM_API_BASE")
    else:
        return None

def minimize_string(data):
    if isinstance(data, dict):
        return {k: minimize_string(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [minimize_string(item) for item in data]
    elif isinstance(data, str):
        # Remove consecutive spaces (keeping single spaces)
        minimized = re.sub(r' {2,}', ' ', data)
        # Remove consecutive newlines (keeping single newlines)
        minimized = re.sub(r'\n{2,}', '\n', minimized)
        # Remove spaces at the start and end of each line
        minimized = re.sub(r'^ +| +$', '', minimized, flags=re.MULTILINE)
        # Remove spaces after colons (except in strings)
        minimized = re.sub(r'(?<!"):\s+', ':', minimized)
        # Remove spaces after commas (except in strings)
        minimized = re.sub(r'(?<!"),\s+', ',', minimized)
        return minimized.strip()
    else:
        return data

def parse_alternative_suggestion(response_content) -> str | None:
    if "Feedback_to_Alternative_Branch" not in response_content:
        return None

    alternative_pattern = r'<Feedback_to_Alternative_Branch>\s*(.*?)\s*(?:</Feedback_to_Alternative_Branch>|<\w+>|$)'
    match = re.search(alternative_pattern, response_content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        return None

def parse_explanation(response_content) -> str:
    explanation_pattern = r'<Explanation>\s*(.*?)\s*(?:</Explanation>|<Feedback_to_Alternative_Branch>|<Reward>|$)'
    match = re.search(explanation_pattern, response_content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        return response_content


def parse_value(response_content, keyword='reward', allowed_values=None):
    """
    Parse the value associated with a given keyword from the LLM response content.

    Args:
    response_content (str): The content of the LLM response.
    keyword (str): The keyword to search for (default: 'reward').
    allowed_values (list or range, optional): A list or range of allowed values.

    Returns:
    int: The parsed integer value, or None if not found, not an integer, or not in allowed_values.
    """
    value_patterns = [
        fr'<\s*{keyword}\s*>\s*:?\s*(-?\d+)',
        fr'<\s*{keyword}\s*>(-?\d+)',
        fr'{keyword}:\s*(-?\d+)',
        fr'\*\*{keyword}\*\*\s*:?\s*(-?\d+)',
        fr'\*\*{keyword.capitalize()}\*\*\s*:?\s*(-?\d+)',
        fr'{keyword.capitalize()}:\s*(-?\d+)',
        fr'<\s*{keyword.capitalize()}\s*>\s*:?\s*(-?\d+)',
        fr'\*\*<\s*{keyword.capitalize()}\s*>\*\*:\s*(-?\d+)',
        fr'\*\*{keyword.capitalize()}:\*\*\s*(-?\d+)',
        fr'<\s*{keyword}\s*>\s*(-?\d+)\s*</\s*{keyword}\s*>',
        fr'<\s*{keyword}\s*>\s*(-?\d+)'
    ]

    matched_value = None
    try:
        # Try to find value using specific patterns
        for pattern in value_patterns:
            match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
            if match:
                matched_value = match.group(1).strip()
                value = int(matched_value)
                if allowed_values is None or value in allowed_values:
                    return value

        # If no pattern matches, look for any number after the keyword
        general_pattern = fr'{keyword}\s*:?\s*(-?\d+)'
        match = re.search(general_pattern, response_content, re.IGNORECASE | re.DOTALL)
        if match:
            matched_value = match.group(1).strip()
            value = int(matched_value)
            if allowed_values is None or value in allowed_values:
                return value

        # If we reach here, either no value was found or it wasn't an integer
        logger.warning(f"No valid integer {keyword} found in the response content.")
        return None
    except ValueError:
        logger.warning(f"Found value {matched_value} at {keyword}, but it's not a valid integer.")
        return None
    except Exception as e:
        logger.error(f"Error parsing {keyword}: {e}")
        return None

class ValuePrompts:
    def build_value_assignment_prompt(**kwargs):
        INTRO = "Please assign a value to the provided state of the search tree. Keep in mind the following definitions:"
        OUTRO = """Your value assesment should be a single integer value to the state from -100 to 100. Always strictly adhere to this output format and don't forget to provide both <Explanation> and <Reward>: 

OUTPUT FORMAT:

<Explanation>: A comprehensive explanation and reasoning behind your decision, alluding to the *common mistakes* where appropriate.
<Reward>: integer reward."""

        prompt_parts = []
        for field in DEFINITIONS.keys():
            if field in kwargs and kwargs[field] is not None:
                definition = DEFINITIONS[field]
                prompt_parts.append(f"<{field}>: {definition}")

        prompt_full = "\n".join(prompt_parts)
        prompt_full = f"{INTRO}\n\n{prompt_full}\n\n{OUTRO}"
        return prompt_full


class MessageCreator:
    
    def create_message(self, state: AgenticState, **kwargs):
        logger.debug(f"Creating message for state {state.name} node_id: {kwargs.get('node_id')} with kwargs {kwargs}.")
    
        VALUE_ASSIGNMENT_PROMPT = ValuePrompts.build_value_assignment_prompt(**kwargs)
        VALUE_PROMPT = f"{ROLE_PROMPT_VALUE}\n\n{MAIN_OBJECTIVE_PROMPT_VALUE}\n\n{STATE_EVALUATION_GUIDELINES}\n\n{DEFINITION_PROMPT}\n\n{VALUE_ASSIGNMENT_PROMPT}"

        state_content_message = "<problem_statement>\n"
        state_content_message += state.initial_message

        state_content_message += self._create_next_state_content(state)

        state_history = state.get_previous_states()

        if state_history:
            formatted_history = []
            for i, previous_state in enumerate(state_history):
                if isinstance(previous_state, Pending):
                    # Ignore pending state
                    continue
                formatted_state = self._format_previous_state(i, previous_state)
                formatted_history.append(formatted_state)

            state_content_message += "\n\n<executed_state>"
            state_content_message += f"\n\n{formatted_history[-1]}"

            state_content_message += "\n\n<state_history>"
            state_content_message += "\n\n".join(formatted_history[:-1])

        state_content_message += "\n\n<state_file_context>\n"
        # Generates the file context prompt as in the regular prompts
        if state.workspace.file_context:
            state_content_message += state.workspace.file_context.create_prompt()
        else:
            state_content_message += "No file context found."

        # Returns the git diff of the changes made until the current state
        if isinstance(state.workspace.file_repo, GitRepository):
            diff = state.workspace.file_repo.diff()
            if diff:
                state_content_message += "\n\n<diff>\n"
                state_content_message += diff

        logger.debug(f"state_content_message 4: {state_content_message}")
        messages = [
            {'role': 'system', 'content': VALUE_PROMPT},
            {'role': 'user', 'content': state_content_message}
        ]
        return messages

    def generate_feedback(self, best_child) -> Optional[str]:
        """
        Generate feedback for the current state, including information about the action taken
        and the next state created from it, using existing helper methods.

        Args:
            best_child (MCTSNode): The best child node from the current state.

        Returns:
            Optional[str]: Feedback string or None if feedback cannot be generated.
        """
        if not best_child.parent or not best_child.parent.state.action_request:
            logger.info("Cannot generate feedback for the root node.")
            return None

        feedback = []
        explanations = [visit.explanation for visit in reversed(best_child.state.visits) if visit.explanation]

        if best_child.parent.state.name == "SearchCode":
            prompt = "Feedback from a parallel problem-solving branch is provided within the <feedback> tag. Carefully review this feedback and use it to adjust your search parameters, ensuring that you implement a different search strategy from previous attempts. "
            prompt += "<feedback>"
            prompt += "\n".join(explanations)
            prompt += "</feedback>"
            return prompt

        FEEDBACK_PROMPT = """The following information describes an action taken in a parallel branch of problem-solving, not in your current trajectory. This action represents an approach taken by a different agent in an entirely separate problem-solving branch. It is not part of your own history or decision-making process. This information is provided solely to inform your decision-making and inspire potential improvements to your approach.

<Alternative_Branch_Action>: An action executed in a completely separate problem-solving branch, distinct from your current path. This action shows how a different agent addressed the same problem you're working on, but in a parallel decision tree. It is not a previous action in your own sequence of decisions.

<Feedback>: The evaluation feedback provided on the Alternative Branch Action. It consists of:
1) An <Assigned_Value>: A numerical score ranging from -100 (lowest) to 100 (highest), indicating the perceived effectiveness of the action in that separate branch.
2) An <Explanation>: A detailed written evaluation of the Alternative Branch Action, analyzing its strengths, weaknesses, and overall impact on solving the problem in that particular branch. This feedback does not reflect on your own actions or decisions.
"""
        feedback.append(FEEDBACK_PROMPT)

        feedback.extend([
            "<Alternative_Branch_Action>",
            best_child.parent.state.action_request.to_prompt(),
            "</Alternative_Branch_Action>",
            "",
            f"<Assigned_Value>{best_child.raw_value:.2f}</Assigned_Value>"
            "<Explanation>"
        ])

        # Add explanations from the best child's visits
        if explanations:
            feedback.extend(explanations)
        else:
            feedback.append("No specific feedback provided.")

        feedback.extend([
            "</Explanation>",
            "",
            "Based on this alternative branch information, propose a new action for your current trajectory."
        ])

        return "\n".join(feedback)

    def _create_next_state_content(self, state: AgenticState) -> str:
        """
        Creates the next_state content for the given state.

        Args:
            state (AgenticState): The current state.

        Returns:
            str: Formatted next_state content.
        """
        next_state_content = "\n\n<next_state>\n"
        next_state_content += f"**next_state**: {state.name}\n"
        state_dict = state.model_dump(
            exclude_none=True, 
            exclude_unset=True,
            exclude={
                "previous_state", "next_states", "origin_state", "clones", "id", "model", "temperature",
                "max_tokens", "include_message_history", "max_iterations"
            }
        )
        next_state_content += "\n## *".join([f"  {k}: {v}" for k, v in state_dict.items()])
        return next_state_content

    def _format_previous_state(self, index: int, previous_state: AgenticState, show_index: bool = True) -> str:
        """
        Formats the content of a previous state.

        Args:
            index (int): The index of the state in the history.
            previous_state (AgenticState): The previous state to format.

        Returns:
            str: Formatted previous state content.
        """
        if show_index:
            formatted_state = f"\n# State {index} {previous_state.name}:\n"
        else:
            formatted_state = f"\n## State Type: {previous_state.name}:\n"

        if previous_state.action_request:
            # action_request_dump = previous_state.action_request.model_dump()
            # if "thoughts" in action_request_dump or "scratch_pad" in action_request_dump:
            #     formatted_state += f"\n## Thoughts:\n"
            #     if "thoughts" in action_request_dump:
            #         formatted_state += action_request_dump['thoughts']
            #     elif "scratch_pad" in action_request_dump:
            #         formatted_state += action_request_dump['scratch_pad']

            formatted_state += f"\n\n## Action:\n"
            formatted_state += "\n".join([f"  {k}: {v}" for k, v in previous_state.action_request.model_dump(exclude={"thoughts", "scratch_pad"}).items()])

        if previous_state.outcome:
            formatted_state += f"\n\n## Outcome:\n"

            # Workaround to show the full file context prompt as outcome for SearchCode states.
            if previous_state.name == "SearchCode" and "ranked_spans" in previous_state.outcome:
                file_context = previous_state.create_file_context()
                file_context.add_ranked_spans(previous_state.outcome["ranked_spans"])
                formatted_state += file_context.create_prompt()
            else:
                formatted_state += "\n".join([f"  {k}: {v}" for k, v in previous_state.outcome.items()])

        return formatted_state
    
    
class ModelCompletion:
    def __init__(self, model, temperature=0.4, max_tokens=1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = configure_api_base(model=model)

    def get_completion(self, messages, **kwargs):

        # Helper function to get value from kwargs or class attribute
        def get_param(param_name, default=None):
            logger.debug(f"model: {self.model}")
            return kwargs.get(param_name, getattr(self, param_name, default))

        # Common parameters
        model = get_param('model', self.model)
        common_params = {
            'model': model,
            'max_tokens': get_param('max_tokens'),
            'temperature': get_param('temperature'),
            'messages': messages
        }

        if self.api_base:
            common_params['api_base'] = self.api_base

        if os.getenv('CUSTOM_LLM_API_KEY') and self.model.startswith("openai"):
            common_params['api_key'] = os.getenv('CUSTOM_LLM_API_KEY')

        if model in LOCAL_MODELS:
            common_params['model'] = f"openai/{common_params['model']}"
            common_params['api_key'] = get_param('api_key', API_KEY)

        return litellm.completion(**common_params)


class MultiAgentDebate:
    def __init__(self, n_agents=8, n_rounds=3, **kwargs):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.include_conclusion = True
        self.debates = collections.defaultdict(list)
        self.temperature = 1
        self.model_agents_map = {
            # 0: "gpt-4o-2024-08-06",
            # 1: "claude-3-5-sonnet-20240620",
            #  2: "meta-llama/Meta-Llama-3.1-70B-Instruct"
                                }

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.completion_params = {'temperature': self.temperature}

    def set_partial_vars(self, func, **kwargs):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    def conduct_debate(self, messages: List[dict], completion_func,
                       output_format, model=None, **kwargs):
        # set completion params
        completion_func = self.set_partial_vars(completion_func, **self.completion_params)

        if not messages:
            raise ValueError("Messages list cannot be empty.")

        node_id = kwargs.get("node_id", None)
        system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        if not system_message:
            raise ValueError("No system message found in the conversation history.")

        # Initialize agent contexts with the full conversation history
        agent_contexts = [messages.copy() for _ in range(self.n_agents)]

        debate_log = {
            "messages": messages,
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "rounds": []
        }

        for round in tqdm(range(self.n_rounds), desc="Debate Rounds"):
            round_log = {"round": round, "agent_responses": []}

            for agent, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:agent] + agent_contexts[agent+1:]
                    debate_message = self.construct_debate_message(agent_contexts_other, -1)
                    agent_context.append({"role": "user", "content": debate_message['content']})

                completion_result = completion_func(messages=agent_context,
                                                    model=self.model_agents_map[agent] if model is None else model)
                action, completion_response = self.parse_completion_result(completion_result)

                assistant_message = self.format_assistant_message(action, completion_response)
                agent_context.append(assistant_message)

            # Handle tool calls if present in the original completion_response
            if completion_response.get('function_call') or (completion_response.get('tool_calls') and len(completion_response['tool_calls']) > 0):
                tool_call = completion_response.get('function_call') or completion_response['tool_calls'][0]
                tool_response = self.process_tool_call(tool_call)
                agent_context.append({"role": "tool", "content": tool_response, "tool_call_id": tool_call.get('id')})

            round_log["agent_responses"].append({
                "agent": agent,
                "message": debate_message if round != 0 else None,
                "response": assistant_message,
                "tool_response": tool_response if 'tool_response' in locals() else None
            })

            debate_log["rounds"].append(round_log)

        debate_summary = self.construct_debate_summary(agent_contexts)
        debate_log["summary"] = debate_summary

        if self.include_conclusion:
            final_action, final_completion_response, final_messages = self.generate_conclusion(
                                                                        messages[-1]['content'],
                                                                        debate_summary, messages,
                                                                        completion_func,
                                                                        output_format
                                                                        )
            # debate_log["conclusion"] = final_messages[-1]['content']
        else:
            final_action, final_completion_response, final_messages = None, None, messages

        debate_log["conclusion"] = final_completion_response.choices[0].message
        # Calculate token usage
        prompt_tokens = token_counter(text=str(debate_log["messages"]))
        completion_tokens = token_counter(text=final_messages if final_messages else " ")
        total_tokens = prompt_tokens + completion_tokens

        if not node_id:
            node_id = str(len(self.debates) + 1)
        self.debates[node_id].append(debate_log)

        if hasattr(self, "debate_log_dir"):
            print(f"debate_log_dir: {self.debate_log_dir}")
            save_to_json(self.debates, self.debate_log_dir)

        return final_action, final_completion_response, final_messages
    
    def process_tool_call(self, tool_call):
        return f"Tool call: {tool_call.function.name}, Arguments: {tool_call.function.arguments}"

    def construct_debate_message(self, agents, idx):
        prefix_string = "These are the solutions to the problem from other agents: "

        for agent_num, agent in enumerate(agents):
            if idx < len(agent):
                agent_response = agent[idx]['content']
                response = f"\n\nAgent {agent_num + 1} solution: ```{agent_response}```"
                prefix_string += response
            else:
                print(f"Warning: Agent {agent_num} does not have a response at index {idx}")

        prefix_string += """\n\nGiven the provided context and responses, provide your own response.
                                You can first reason about the solutions provided by the other agents and then provide your own solution. 
                                Strictly adhere to any output format used in the responses, and especially any tool/api/function calls if present, like those enclosed in <> or **."""
        return {"role": "user", "content": prefix_string}

    def generate_conclusion(self, initial_context, debate_summary,
                            messages, completion_func, output_format):
        conclusion_prompt = f"""
        
        **Initial problem context:**
        {initial_context}

        
        **Agent Answers:**
        {debate_summary}

        
        {output_format}
        """
        # {VALUE_OUTPUT_FORMAT}, CONCLUSION_PROMPT_ACTION

        conclusion_context = [
            {"role": "system", "content": "You are a highly capable AI assistant tasked with synthesizing information and reaching conclusions."},
            {"role": "user", "content": conclusion_prompt}
        ]

        completion_result = completion_func(
            messages=conclusion_context,
            temperature=0.2  # Lower temperature for conclusion generation
            )
        action, completion_response = self.parse_completion_result(completion_result)

        return action, completion_response, completion_response.choices[0].message.content

    def parse_completion_result(self, result):
        if isinstance(result, tuple):
            return result  # It's already (action, completion_response)
        else:
            return None, result  # It's just the completion_response

    def format_assistant_message(self, action, completion_response):
        if action is None and isinstance(completion_response, ModelResponse):
            # Extract content from ModelResponse
            if completion_response.choices and completion_response.choices[0].message:
                return {
                    "role": "assistant",
                    "content": completion_response.choices[0].message.content
                }
        elif isinstance(action, dict):
            return {"role": "assistant", "content": json.dumps(action)}
        elif hasattr(action, '__dict__'):
            try:
                return {"role": "assistant", "content": json.dumps(action.__dict__)}
            except TypeError:
                return {"role": "assistant", "content": str(action)}
        else:
            return {"role": "assistant", "content": str(action)}

        # If we couldn't extract a message, return None
        return None

    def construct_debate_summary(self, agent_contexts):
        summary = "Debate Summary:\n\n"
        for i, context in enumerate(agent_contexts):
            summary += f"Agent {i+1} final response:\n{context[-1]['content']}\n\n"
        return summary

    def model_dump(self):
        return {
            "choices": [{
                "message": {
                    "content": self.choices[0].message.content
                }
            }],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens
            }
        }

class LLM_Value_Function:

    def __init__(self, model: str | None = None, log_to_file: bool = True, states_to_explore: list[str] | None = None,
                 temperature=None, **kwargs):
        self.ALLOWED_REWARD_VALUES = range(-100, 101)

        if model:
            self.model = model
        else:
            self.model = MODEL
            logger.info(f"Using default model: {self.model}")

        self.max_tokens_value = 1000
        self.max_tokens_tree = 1000
        if temperature is None:
            if kwargs.get("debate", True) or kwargs.get("n_calls", 1) > 1:
                self.temperature = 1
            else:
                self.temperature = 0.6
        else:
            self.temperature = temperature
        self.interactions = []
        self.filename = "llm_interactions.json"
        self.filename_tree = None
        self.log_to_file = log_to_file

        self.states_to_explore = states_to_explore or ["SearchCode", "PlanToCode"]

        for k, v in kwargs.items():
            setattr(self, k, v)

        base_dir = os.path.join(os.path.dirname(self.filename), self.model)
        file_name = os.path.basename(self.filename)
        logger.info(f"Base dir: {base_dir}, file name: {file_name}")
        self.filename_value = os.path.join(base_dir, "rews", file_name)
        self.file_dir_tree = os.path.join(base_dir, "tree_eval", file_name)
        self.debate_log_dir = os.path.join(base_dir, "debate_logs", file_name)

        self.model_completion = ModelCompletion(self.model, self.temperature, self.max_tokens_value)
        self.debate = MultiAgentDebate(
            debate_log_dir=self.debate_log_dir,
            **kwargs
        )

    def parse_message(self, input_list):
        formatted_str = ""
        for index, item in enumerate(input_list):
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                role = item['role']
                content = item['content'].replace('\n', ' ').strip()
                formatted_str += f"{index}: {{\n  'role': '{role}',\n  'content': '{content}'\n}}\n\n"
        return formatted_str.rstrip()  # Remove trailing newline

    def create_message(self, node: "MCTSNode"):
        executed_state = node.state.previous_state
        while executed_state:
            if not executed_state.action_request:
                logger.warning(f"No action request found for state {executed_state.name}.")
                return None

            if executed_state.name == "SearchCode":
                return self.create_message_initial_search(executed_state)
            elif executed_state.name == "PlanToCode" and isinstance(executed_state.action_request.action, RequestMoreContext):
                return self.create_message_request_more_context(executed_state)
            elif executed_state.name == "PlanToCode":
                return self.create_message_plan_to_code(executed_state, node.state)

            executed_state = executed_state.previous_state

        logger.warning(f"No previous executed state found for Node{node.id} and state {node.state.trace_name}.")
        return None

    def create_message_default(self, state: AgenticState, **kwargs):
        logger.debug(f"Creating message for state {state.name} node_id: {kwargs.get('node_id')} with kwargs {kwargs}.")

        VALUE_ASSIGNMENT_PROMPT = ValuePrompts.build_value_assignment_prompt(**kwargs)
        VALUE_PROMPT = f"{ROLE_PROMPT_VALUE}\n\n{MAIN_OBJECTIVE_PROMPT_VALUE}\n\n{STATE_EVALUATION_GUIDELINES}\n\n{DEFINITION_PROMPT}\n\n{VALUE_ASSIGNMENT_PROMPT}"

        state_content_message = "<problem_statement>\n"
        state_content_message += f"{state.initial_message}\n"
        state_content_message += "</problem_statement>\n"

        state_content_message += "\n<next_state>\n"
        state_content_message += f"{state.name}\n"
        state_dict = state.model_dump(exclude_none=True, exclude_unset=True,
                                      exclude={"previous_state", "next_states", "id", "model", "temperature",
                                               "max_tokens", "include_message_history", "max_iterations"})
        state_content_message += "\n## *".join([f"  {k}: {v}" for k, v in state_dict.items()])
        state_content_message += "\n</next_state>\n"

        state_history = state.get_previous_states()

        if state_history:
            formatted_history = []
            for i, previous_state in enumerate(state_history):
                if isinstance(previous_state, Pending):
                    # Ignore pending state
                    continue

                formatted_state = f"\n<state_{i}>\n<name>{previous_state.name}</name>\n"

                if previous_state.action_request:
                    action_request_dump = previous_state.action_request.model_dump()
                    # if "thoughts" in action_request_dump or "scratch_pad" in action_request_dump:
                    #     formatted_state += f"<thoughts>\n"
                    #     if "thoughts" in action_request_dump:
                    #         formatted_state += action_request_dump['thoughts']
                    #     elif "scratch_pad" in action_request_dump:
                    #         formatted_state += action_request_dump['scratch_pad']
                    #     formatted_state += "\n</thoughts>\n"

                    formatted_state += f"<action>\n"
                    formatted_state += "\n".join([f"  {k}: {v}" for k, v in previous_state.action_request.model_dump(
                        exclude={"thoughts", "scratch_pad"}).items()])
                    formatted_state += "\n</action>\n"

                if previous_state.outcome:
                    formatted_state += f"<outcome>\n"

                    # Workaround to show the full file context prompt as outcome for SearchCode states.
                    if previous_state.name == "SearchCode" and "ranked_spans" in previous_state.outcome:
                        file_context = previous_state.create_file_context()
                        file_context.add_ranked_spans(previous_state.outcome["ranked_spans"])
                        formatted_state += file_context.create_prompt()
                    else:
                        formatted_state += "\n".join([f"  {k}: {v}" for k, v in previous_state.outcome.items()])
                    formatted_state += "\n</outcome>\n"

                formatted_state += f"</state_{i}>\n"
                formatted_history.append(formatted_state)

            state_content_message += "\n<executed_state>"
            state_content_message += f"\n{formatted_history[-1]}"
            state_content_message += "</executed_state>\n"

            state_content_message += "\n<state_history>\n"
            state_content_message += "\n".join(formatted_history[:-1])
            state_content_message += "\n</state_history>\n"

        state_content_message += "\n<state_file_context>\n"
        # Generates the file context prompt as in the regular prompts
        if state.workspace.file_context:
            state_content_message += state.workspace.file_context.create_prompt()
        else:
            state_content_message += "No file context found."
        state_content_message += "\n</state_file_context>\n"

        # Returns the git diff of the changes made until the current state
        if isinstance(state.workspace.file_repo, GitRepository):
            diff = state.workspace.file_repo.diff()
            if diff:
                state_content_message += "\n<diff>\n"
                state_content_message += diff
                state_content_message += "\n</diff>\n"

        logger.debug(f"state_content_message 4: {state_content_message}")
        messages = [
            {'role': 'system', 'content': VALUE_PROMPT},
            {'role': 'user', 'content': state_content_message}
        ]
        return messages

    def create_message_initial_search(self, state: SearchCode):
        logger.debug(f"Creating search code message for {state.name}:{state.id}.")

        state_content_message = f"<problem_statement>\n"
        state_content_message += state.initial_message
        state_content_message += "</problem_statement>\n"

        state_content_message += "\n<search_request>\n"
        state_content_message += state.action_request.to_prompt()
        state_content_message += "\n</search_request>\n"

        state_content_message += "\n<search_results>\n"
        if state.outcome and state.outcome.get("ranked_spans"):
            file_context = state.create_file_context()
            for ranked_span in state.outcome["ranked_spans"]:
                # Use tokens=1 to only show signatures of spans
                if isinstance(ranked_span, dict):
                    file_context.add_span_to_context(ranked_span["file_path"], ranked_span["span_id"], tokens=1)
                else:
                    file_context.add_span_to_context(ranked_span.file_path, ranked_span.span_id, tokens=1)
            state_content_message += file_context.create_prompt(
                show_outcommented_code=True, outcomment_code_comment="... outcommented code")
        else:
            state_content_message += "No search results found."
        state_content_message += "\n</search_results>\n"

        state_content_message += "\n<identified_code>\n"
        if state.workspace.file_context:
            state_content_message += state.workspace.file_context.create_prompt(
                show_line_numbers=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            logger.warning("No file context found.")
            state_content_message += "No file context found."

        state_content_message += "\n</identified_code>"

        logger.debug(f"state_content_message 4: {state_content_message}")
        messages = [
            {'role': 'system', 'content': VALUE_FUNCTION_SEARCH_PROMPT},
            {'role': 'user', 'content': state_content_message}
        ]
        return messages

    def create_message_request_more_context(self, state):
        logger.debug(f"Creating request more context message for {state.name}:{state.id}.")

        state_content_message = "<problem_statement>\n"
        state_content_message += state.initial_message
        state_content_message += "</problem_statement>\n"

        state_content_message += "\n<file_context>\n"
        if state.workspace.file_context:
            state_content_message += state.workspace.file_context.create_prompt(
                show_line_numbers=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            state_content_message += "No file context found."
        state_content_message += "\n</file_context>\n"

        state_content_message += "\n<history>\n"
        state_history = state.get_previous_states()
        if state_history:
            for i, previous_state in enumerate(state_history):
                if isinstance(previous_state, Pending):
                    continue
                state_content_message += f"<state_{i}>\n"
                state_content_message += f"<name>{previous_state.name}</name>\n"
                if previous_state.action_request:
                    state_content_message += "<action>\n"
                    state_content_message += previous_state.action_request.to_prompt()
                    state_content_message += "\n</action>\n"
                if previous_state.outcome:
                    state_content_message += "<outcome>\n"
                    state_content_message += str(previous_state.outcome)
                    state_content_message += "\n</outcome>\n"
                state_content_message += f"</state_{i}>\n"
        else:
            state_content_message += "No previous state history."
        state_content_message += "\n</history>\n"

        state_content_message += "\n<executed_action>\n"
        state_content_message += state.action_request.to_prompt()
        state_content_message += "\n</executed_action>\n"

        logger.debug(f"state_content_message: {state_content_message}")
        messages = [
            {'role': 'system', 'content': VALUE_FUNCTION_REQUEST_MORE_CONTEXT_PROMPT},
            {'role': 'user', 'content': state_content_message}
        ]
        return messages

    def generate_plan_system_prompt(self, has_diff: bool, has_test_results: bool):
        prompt = """Your role is to evaluate the executed action of AI agents traversing a search tree to solve programming issues. Assess whether proposed changes and planned actions are appropriate for addressing the problem."""

        prompt += "\n\n# Evaluation Criteria\n"
        prompt += "\n * Solution Quality: Logical changes, contextual fit, syntactic correctness, and overall improvement without introducing new issues."
        prompt += "\n * Progress Assessment: Awareness of solution history, detection of repetitive actions, and evaluation of planned next steps."

        if has_diff:
            prompt += "\n * Code Modification Accuracy: Correct identification of code spans, accuracy of changes, and absence of unintended modifications."
            prompt += "\n * Python-Specific Features Utilization: Assess whether the agent has considered and appropriately utilized standard Python decorators, methods, or language features that are specifically designed for the task at hand. Reward the use of Pythonic solutions and built-in functionality that simplifies the code or makes it more efficient. Penalize if obvious Python-specific solutions are overlooked in favor of more complex or less efficient approaches."

        if has_test_results:
            prompt += "\n * Testing: Evaluation of test results, considering if failures could have been reasonably foreseen."

        if has_diff:
            prompt += """\n\n## Git Diff Evaluation:

 * Carefully analyze the provided Git diff for accuracy and correctness.
 * Ensure the diff correctly represents the intended changes without unintended modifications.
 * Check for common Git diff issues such as:
  * Incorrect line numbers
  * Unintended additions or deletions
  * Formatting errors or inconsistencies
  * Changes to unrelated parts of the code

Verify that the diff aligns with the described code change instructions. Unintended changes should be identified and heavily penalized. 

In your explanation, include a specific assessment of the Git diff quality and accuracy. If any issues are found in the Git diff, they should be clearly stated and factored into the overall reward calculation."""

        if has_test_results:
            prompt += """\n\n##Test Result Evaluation:

Analyze test results in conjunction with the proposed code changes.
Categorize test failures as:
  * Expected: Directly related to intended changes and acknowledged by the agent.
  * Foreseeable: Should have been anticipated and addressed by the agent.
  * Unforeseeable: Reveal complex interactions or edge cases not obvious from the problem statement.

Differentiate Between Error Types:
 * Minor, Easily Fixable Errors: Such as syntax errors, missing imports, or small typos that can be quickly corrected in the next iteration.
 * Significant Issues: Errors that indicate fundamental problems in the approach or logic.

 * Assess the agent's awareness and handling of potential test issues.
 * Consider the overall impact of test failures on the solution's viability.
 * Balance the severity and fixability of failures against the progress made towards solving the main problem.
 * In your explanation, describe any test failures, their likely causes, and suggest potential next steps.
 * When calculating the reward, penalize significant and foreseeable failures more heavily than minor, fixable errors.
 * Encourage iterative improvement by considering the potential for the agent to fix minor issues in subsequent iterations."""

        prompt += """\n\n# Guidelines for Feedback:

 * Provide detailed, actionable feedback for both correct and incorrect actions.
 * Consider the full context of the problem-solving process.
 * Suggest improvements or alternative approaches when applicable.
 * Pay close attention to diffs, syntax, and minor details that could invalidate an action.

# Reward Scale and Guidelines:

The reward value must be based on how confident you are that the agent’s solution is the most optimal one possible with no unresolved issues or pending tasks. The scale ranges from -100 to 100, where:

 * 100: You are fully confident that the proposed solution is the most optimal possible, has been thoroughly tested, and requires no further changes.
 * 75-99: The approach is likely the best one possible, but there are minor issues or opportunities for optimization. 
          All major functionality is correct, but some small improvements or additional testing may be needed. 
          There might be some edge cases that are not covered.
 * 0-74: The solution has been partially implemented or is incomplete or there are likely alternative approaches that might be better, i.e., this is likely not the most optimal approach. 
         The core problem might be addressed, but there are significant issues with tests, logical flow, or side effects that need attention. 
         There are likely alternative approaches that are much better.
 * 0: The solution is not yet functional or is missing key elements. The agent's assertion that the task is finished is incorrect, and substantial work is still required to fully resolve the issue.  
 * -1 to -49: The proposed solution introduces new issues or regresses existing functionality, but some elements of the solution show potential or may be salvageable.
              Modifying the wrong code, unintentionally removing or altering existing code, introducing syntax errors, or producing incorrect diffs fall into this range.
              Repetitive actions without progress fall into this range.
 * -50 to -100: The solution is entirely incorrect, causing significant new problems, or fails to address the original issue entirely. Immediate and comprehensive changes are necessary. 
                Persistent repetitive actions without progress should be heavily penalized.
"""

        prompt += """\n# Input Data Format:

 * Problem Statement: This will be provided within the <problem_statement> XML tag and contains the initial message or problem description the coding agent is trying to solve.
 * History: The sequence of actions taken prior to the current state will be contained within the <history> XML tag. This will include information on the parts of the codebase that were changed, the resulting diff and test results.
 * Executed Action: The last executed action of the coding agent will be provided within the <executed_action> XML tag, this includes the proposed changes and the resulting diff of the change.
 * File Context: The relevant code context will be provided within the <file_context> XML tag and pertains to the state the agent is operating on.
 * Full Git Diff: The full Git diff up to the current state will be provided within the <full_git_diff> XML tag. This shows all changes made from the initial state to the current one and should be considered in your evaluation to ensure the modifications align with the overall solution.
 * Test Results: The results of any test cases run on the modified code will be provided within the <test_results> XML tag. This will include information about passed, failed, or skipped tests, which should be carefully evaluated to confirm the correctness of the changes.
"""
        prompt += """\n# Feedback Structure:

You must provide your evaluation in the following format:

 * Explanation: Offer a detailed explanation and reasoning behind your decision, referring to common mistakes where relevant. If an action was incorrect, identify exactly where and why the mistake occurred, including if the agent is stuck in repetitive actions without making progress. If the action was correct, confirm why it aligns with the problem context. Always think about whether there could be an alternative approach that is better, and if so, mention it in the explanation. If test failures occurred, consider whether they could have been foreseen and address this in your explanation. If the agent modified the wrong code or introduced syntax errors, highlight this and explain the impact.
 * Reward: Assign a single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue.

# Output Format:

Please ensure your output strictly adheres to the following structure:

<Explanation>: [Your explanation of the evaluation in max two paragraphs.]
<Reward>: [A single integer reward value between -100 and 100]

Remember to strictly follow the output format of <Explanation> and <Reward> for each evaluation."""

        return prompt

    def generate_plan_system_prompt_with_suggestion(self, has_diff: bool, has_test_results: bool):
        prompt = f"""Your role is to evaluate the executed action of the search tree that our AI agents are traversing, to help us determine the best trajectory to solve a programming issue. The agent is responsible for identifying and modifying the correct file(s) in response to the problem statement.

At this stage, the agent is still working on the solution. Your task is twofold:
1. **Evaluation**: Assess both the proposed solution (instructions and pseudo code) and the actual code changes (Git diff) to determine if the agent is on the right path to resolving the issue.
2. **Alternative Feedback**: Independently of your evaluation, provide guidance for an alternative problem-solving branch. This ensures parallel exploration of different solution paths.

You will assign a reward value and provide feedback based on the alignment of the proposed changes with the problem statement, the file's content, any test cases, and the next steps proposed by the agent.
"""

        prompt += """
# Evaluation Method:

**Think through the following steps carefully and systematically:**

1. **Understand the Problem and Proposed Solution:**
   - **Problem Statement Comprehension**: Ensure you fully understand the problem the agent is trying to solve.
   - **Proposed Solution Analysis**: Evaluate if the proposed instructions and pseudo code logically and effectively address the problem.

2. **Analyze the Code Changes (Git Diff):**
   - **Line-by-Line Review**: Perform a detailed, line-by-line analysis of the Git diff to identify any discrepancies or unintended modifications to other parts of the code.
   - **Alignment with Pseudo Code**: Check if the code changes accurately implement the pseudo code.
   - **Code Correctness**: Verify that the code changes are syntactically correct and logically sound.
   - **Structural Integrity**: Ensure that code modifications do not unintentionally affect unrelated parts of the codebase.
   - **Placement of New Code**: Confirm that new methods, functions, or tests are placed correctly and are not nested incorrectly within unrelated code.

3. **Evaluate Tests:**
   - **Test Appropriateness**: Assess whether tests have been appropriately added or modified to cover the new changes.
   - **Test Outcomes**: Review test results to confirm that all tests pass successfully.

4. **Consider the Agent's Progress:**
   - **History and Action Evaluation**: Determine if the agent is making meaningful progress and not repeating previous mistakes.

5. **Identify Any Issues:**
   - **Mistakes in Editing Code**: Note any errors such as syntax issues, unintended deletions, or incorrect modifications.
   - **Unintended Modifications**: Pay attention to any changes in the Git diff that affect unrelated code.
"""

        if has_test_results:
            prompt += """
## Test Result Evaluation:

Analyze test results in conjunction with both the proposed changes and the actual code implementation.
Categorize test failures as:
  * Expected: Directly related to intended changes and acknowledged by the agent.
  * Foreseeable: Should have been anticipated and addressed by the agent.
  * Unforeseeable: Reveal complex interactions or edge cases not obvious from the problem statement.

Differentiate Between Error Types:
 * Minor, Easily Fixable Errors: Such as syntax errors, missing imports, or small typos that can be quickly corrected in the next iteration.
 * Significant Issues: Errors that indicate fundamental problems in the approach or logic.

 * Assess the agent's awareness and handling of potential test issues in both the proposal and implementation.
 * Consider the overall impact of test failures on the solution's viability.
 * Balance the severity and fixability of failures against the progress made towards solving the main problem.
 * In your explanation, describe any test failures, their likely causes, and suggest potential next steps.
 * When calculating the reward, penalize significant and foreseeable failures more heavily than minor, fixable errors.
 * Encourage iterative improvement by considering the potential for the agent to fix minor issues in subsequent iterations.
 """

        prompt += """
# Alternative Branch Guidance
Always provide guidance for an alternative problem-solving branch, regardless of the evaluation outcome of the current action. When suggesting alternative approaches, consider potential changes not only within the existing file context but also in other parts of the codebase. Describe conceptual alternative approaches or strategies without providing actual code implementations. This helps guide the alternative branch towards potentially better problem-solving methods without deviating into unnecessary details.
"""

        prompt += """
# Reward Scale and Guidelines:

The reward value must be based on how confident you are that the agent's solution is the most optimal one possible with no unresolved issues or pending tasks. The scale ranges from -100 to 100, where:

 * 100: You are fully confident that the proposed solution is the most optimal possible, strictly aligns with the Pseudo code, has been thoroughly tested, and requires no further changes.
 * 75-99: The approach is likely the best one possible, but there are minor issues or opportunities for optimization. 
          All major functionality is correct, but some small improvements or additional testing may be needed. 
          There might be some edge cases that are not covered.
 * 0-74: The solution has been partially implemented or is incomplete or there are likely alternative approaches that might be better, i.e., this is likely not the most optimal approach. 
         The core problem might be addressed, but there are significant issues with tests, logical flow, or side effects that need attention. 
         There are likely alternative approaches that are much better.
 * 0: The solution is not yet functional or is missing key elements. The agent's assertion that the task is finished is incorrect, and substantial work is still required to fully resolve the issue.  
 * -1 to -49: The proposed solution introduces new issues or regresses existing functionality, but some elements of the solution show potential or may be salvageable.
              Significant misalignment with the Pseudo code, leading to partial or incorrect implementations.
              Modifying the wrong code, unintentionally removing or altering existing code, introducing syntax errors, or producing incorrect diffs fall into this range.
              Repetitive actions without progress fall into this range.
 * -50 to -100: The solution is entirely incorrect, causing significant new problems, or fails to address the original issue entirely. Immediate and comprehensive changes are necessary. 
                Persistent repetitive actions without progress should be heavily penalized.
"""

        prompt += """
# Input Data Format:

 * Problem Statement: This will be provided within the <problem_statement> XML tag and contains the initial message or problem description the coding agent is trying to solve.
 * History: The sequence of actions taken prior to the current state will be contained within the <history> XML tag. This will include information on the parts of the codebase that were changed, the resulting diff and test results.
 * Executed Action: The last executed action of the coding agent will be provided within the <executed_action> XML tag, this includes the proposed changes and the resulting diff of the change.
 * File Context: The relevant code context will be provided within the <file_context> XML tag and pertains to the state the agent is operating on.
 * Full Git Diff: The full Git diff up to the current state will be provided within the <full_git_diff> XML tag. This shows all changes made from the initial state to the current one and should be considered in your evaluation to ensure the modifications align with the overall solution.
 * Test Results: The results of any test cases run on the modified code will be provided within the <test_results> XML tag. This will include information about passed, failed, or skipped tests, which should be carefully evaluated to confirm the correctness of the changes.
"""
        
        prompt += """
# Feedback Structure:

You must provide your evaluation in the following format:

<Explanation>
* Proposed Solution Evaluation: [Brief assessment of the proposed instructions and pseudo code]
* Code Implementation Evaluation: [Brief assessment of the actual code changes (Git diff)]
* History and Action Evaluation: [Brief assessment of the action's contribution to solving the problem]
* Test Result Evaluation: [Brief assessment of the test results]
</Explanation>

<Feedback_to_Alternative_Branch>
[Your feedback for an alternative problem-solving approach in one paragraph]
</Feedback_to_Alternative_Branch>

<Reward>[A single integer reward value between -100 and 100]</Reward>

Remember to strictly follow this output format for each evaluation. The reward should reflect your confidence in the correctness of the action and its likelihood of resolving the issue."""

        return prompt

    def create_message_plan_to_code(self, executed_state: State, next_state: State):
        logger.info(f"create_message_plan_to_code() {executed_state.trace_name} -> {next_state.trace_name}.")

        state_content_message = f"<problem_statement>\n"
        state_content_message += executed_state.initial_message
        state_content_message += "\n</problem_statement>\n\n"

        has_diff = False
        has_test_results = False

        state_history = next_state.get_previous_states()
        if state_history:
            formatted_history = []
            counter = 0

            for previous_state in state_history:
                if previous_state.name in self.states_to_explore:
                    counter += 1
                    formatted_state = f"\n# {counter}. Action: {previous_state.action_request.name}\n\n"
                    formatted_state += previous_state.action_request.to_prompt()
                    formatted_history.append(formatted_state)

            if formatted_history:
                state_content_message += "<history>\n"
                state_content_message += "\n".join(formatted_history)
                state_content_message += "\n</history>\n\n"

        if next_state.name == "Finished":
            state_content_message += "<reasoning_for_completion>\n"

            state_content_message += "\n\n# Statement: \n"
            state_content_message += executed_state.action_request.action.finish_reason
            state_content_message += "</reasoning_for_completion>\n"
        else:
            state_content_message += "\n\n<executed_action>\n"
            state_content_message += executed_state.action_request.to_prompt()

            state_content_message += "\n\n## Outcome\n"

            if hasattr(next_state, "diff") and next_state.diff:
                state_content_message += f"\n<git_diff>\n```\n{next_state.diff}\n```\n</git_diff>\n"
                has_diff = True
            elif executed_state.action_request.action.name == "RequestCodeChange":
                state_content_message += f"\nThe requested code change was refused. \n"

                if hasattr(next_state, "message") and next_state.message:
                    state_content_message += f"Reason: {next_state.message}\n"
            elif executed_state.action_request.action.name in ["FindFunction", "FindClass", "SemanticSearch"] and hasattr(next_state, "updated_file_context") and next_state.updated_file_context:

                if hasattr(next_state, "message") and next_state.message:
                    state_content_message += next_state.message

                state_content_message += "\n\nFound the following code and added it to file context:\n"
                file_context = next_state.create_file_context()
                file_context.add_files_with_spans(next_state.updated_file_context)
                state_content_message += file_context.create_prompt(
                    show_outcommented_code=True, outcomment_code_comment="... outcommented code")

            elif hasattr(next_state, "message") and next_state.message:
                state_content_message += next_state.message

            state_content_message += "\n</executed_action>\n"

        state_content_message += "\n<file_context>\n"
        if next_state.workspace.file_context:
            state_content_message += next_state.workspace.file_context.create_prompt(
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            logger.warning("No file context found.")
            state_content_message += "No file context found."
        state_content_message += "\n</file_context>\n\n"

        if next_state.name == "Finished" and hasattr(executed_state, "test_results"):
            test_results = executed_state.test_results
        elif hasattr(next_state, "test_results"):
            test_results = next_state.test_results
        else:
            test_results = None

        if test_results:
            failures = [test for test in test_results if test.status in [TestStatus.FAILED, TestStatus.ERROR]]
            has_test_results = len(failures) > 0

            state_content_message += "<test_results>\n"
            if executed_state.action_request.name == "Finish" and hasattr(executed_state, "test_results"):
                state_content_message += executed_state.verification_result(verbose=True)
            elif hasattr(next_state, "test_results"):
                state_content_message += next_state.verification_result(verbose=True)
            state_content_message += "\n</test_results>\n"

        if isinstance(next_state.workspace.file_repo, GitRepository):
            diff = next_state.workspace.file_repo.diff()
            if diff:
                state_content_message += "<full_git_diff>\n"
                state_content_message += diff
                state_content_message += "\n</full_git_diff>\n\n"

        if next_state.name == "Finished":
            system_prompt = VALUE_FUNCTION_FINISH_PROMPT
        else:
            system_prompt = self.generate_plan_system_prompt(has_diff=has_diff, has_test_results=has_test_results)

        logger.debug(f"state_content_message 4: {state_content_message}")
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': state_content_message}
        ]
        return messages

    def create_message_tree(self, search_tree):
        VALUE_PROMPT = f"{ROLE_PROMPT_TREE}\n\n{MAIN_OBJECTIVE_PROMPT_TREE}"

        messages = [
            {'role': 'system', 'content': VALUE_PROMPT},
            {'role': 'user', 'content': str(search_tree)}
        ]
        return messages

    def compare_solutions(self, solutions, problem_statement,
                          debate=False):
        ROLE_PROMPT = f"""Below are a series of suggested changes to address the <Problem Statement>.   
Your task is to carefully evaluate each change and decide which one is the most appropriate to address the issue."""
        FORMAT_PROMPT = f"""Provie your answer in the following format:
<Explanation>: A comprehensive explanation and reasoning behind your decision
<ID>: The ID of the change you believe is the most appropriate"""

        SYSTEM_MESSAGE = f"{ROLE_PROMPT}\n{FORMAT_PROMPT}"
        USER_MESSAGE = f"<Problem Statement>\n{problem_statement}</Problem Statement>\n<Solutions>\n{solutions}\n</Solutions>"

        messages = [
            {'role': 'system', 'content': SYSTEM_MESSAGE},
            {'role': 'user', 'content': USER_MESSAGE}
        ]
        if debate:
            completion_func = partial(
                self.model_completion.get_completion,
            )
            _, completion_response, _  = self.debate.conduct_debate(messages=messages,
                                                                model=self.model,
                                                                completion_func=completion_func,
                                                                output_format=FORMAT_PROMPT)
        else:
            completion_response = self.model_completion.get_completion(messages)

        explanation = parse_explanation(completion_response.choices[0].message.content)
        id = parse_value(completion_response.choices[0].message.content,
                         keyword="ID")

        return id, explanation

    def create_message_compare_solutions(self, finished_solutions, include_history: bool = False, show_reward: bool = False):
        logger.info(f"Comparing {len(finished_solutions)} solutions.")

        solutions = ""
        for finished_solution in finished_solutions:
            solutions += f"\n<Solution id={finished_solution.state.id}>\n"

            if show_reward:
                visit = next((visit for visit in finished_solution.state.visits if visit.source_state_id == finished_solution.state.id), None)
                if visit:
                    solutions += f"<Explanation>{visit.explanation}</Explanation>\n"
                    solutions += f"<Reward>{visit.value}</Reward>\n"

            if include_history: 
                state_history = finished_solution.state.get_previous_states()
                if state_history:
                    formatted_history = []
                    counter = 0

                    for previous_state in state_history:
                        if previous_state.name in self.states_to_explore:
                            counter += 1
                            formatted_state = f"\n# {counter}. Action: {previous_state.action_request.name}\n\n"
                            formatted_state += previous_state.action_request.to_prompt()
                            formatted_history.append(formatted_state)

                    if formatted_history:
                        solutions += "<history>\n"
                        solutions += "\n".join(formatted_history)
                        solutions += "\n</history>\n\n"

            solutions += "<Patch>"
            solutions += finished_solution.snapshot["repository"].get("patch")
            solutions += "</Patch>"

            solutions += "\n</Solution>\n"
        return solutions

    def recursive_reward(self, state):
        if state is None:
            return []

        rewards = self.recursive_reward(state.previous_state)

        reward = next((visit.value for visit in state.visits if
                       visit.source_state_id == state.id), None)

        if reward:
            rewards.append(reward)
        return rewards

    def find_possible_solutions(self, trajectory) -> List[TrajectoryState]:
        possible_solutions = []
        patches = set()

        for transition in trajectory.transitions:
            if not transition.state.next_states and transition.snapshot["repository"].get("patch") and \
                    transition.snapshot["repository"].get("patch") not in patches:
                reward = next((visit.value for visit in transition.state.visits if
                               visit.source_state_id == transition.state.id), None)

                rewards = self.recursive_reward(transition.state)
                avg_reward = sum(rewards) / len(rewards) if rewards else None
                if avg_reward and avg_reward >= 0 and (reward is None or reward >= 0):
                    possible_solutions.append((avg_reward, transition))
                    patches.add(transition.snapshot["repository"]["patch"])

        return [item[1] for item in sorted(possible_solutions, key=lambda x: x[0], reverse=True)[:5]]

    def compare_solutions2(self, trajectory: Trajectory, include_history: bool = False, show_reward: bool = True, debate: bool = False) -> TrajectoryState | None:
        finished_solutions = [
            transition
            for transition in trajectory.transitions
            if transition.state.name == "Finished" and
               transition.snapshot["repository"] and
               transition.snapshot["repository"].get("patch")
        ]

        if not finished_solutions:
            finished_solutions = self.find_possible_solutions(trajectory)
            logger.info(
                f"No finished solutions found but found {len(finished_solutions)} possible not finished solutions")
            show_reward = False

        if len(finished_solutions) == 0:
            logger.warning(f"No finished solutions found")
            return None
        elif len(finished_solutions) == 1:
            state_id = finished_solutions[0].state.id
        else:
            solutions = self.create_message_compare_solutions(finished_solutions, include_history, show_reward)
            state_id, explanation = self.compare_solutions(solutions, trajectory.initial_message, debate=debate)

        if not state_id:
            logger.warning(f"Failed to find a valid state_id, return best_trajectory")

        return next((transition for transition in finished_solutions if transition.state.id == state_id), None)

    def get_reward(self, node: "MCTSNode", debate: bool = False,
                   message=None, n_calls=1, **kwargs) -> Tuple[float, str]:
        total_reward = 0
        completions = []
        outputs = []
        max_retries = 3  # Maximum number of retries per call

        if message:  # TODO: What's this?
            logger.info(f"get_reward() Using message: {message}")
            messages = message
        elif node.state:
            messages = self.create_message(node=node)
            if not messages:
                logger.warning(f"get_reward() No messages created for node {node}")
                return None, None
        else:
            messages = kwargs.get("messages")
            if not messages:
                raise ValueError("Either 'state', 'message', or 'messages' in kwargs must be provided")

        common_data = {
            "timestamp": datetime.now().isoformat(),
            "node_id": kwargs.get("node_id", None),
            "input": {
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens_value
            }
        }


        explanations = []

        for i in range(n_calls):
            attempts = 0
            while attempts < max_retries:
                time0 = time.time()

                response_content = None
                try:
                    if debate:
                        completion_func = partial(
                            self.model_completion.get_completion,
                        )
                        CONCLUSION_PROMPT_VALUE = f"{CONCLUSION_PROMPT}\n\n{VALUE_OUTPUT_FORMAT}"
                        _, completion_response, _  = self.debate.conduct_debate(messages=messages,
                                                                            completion_func=completion_func,
                                                                            output_format=CONCLUSION_PROMPT_VALUE,
                                                                            **kwargs)
                    else:
                        completion_response = self.model_completion.get_completion(messages)

                    completion = Completion.from_llm_completion(messages, completion_response, self.model)
                    completions.append(completion)

                    logger.debug(
                        f"number of tokens: {completion_response.usage.prompt_tokens + completion_response.usage.completion_tokens}")

                    response_content = completion_response.choices[0].message.content
                    reward = parse_value(response_content,
                                         keyword='reward', allowed_values=self.ALLOWED_REWARD_VALUES)

                    alternative_suggestion = parse_alternative_suggestion(response_content)

                    explanation = parse_explanation(response_content)

                    if reward is not None:
                        total_reward += reward
                        logger.info(f"Parsed out reward: {reward}\nExplanation: {explanation}\nALternative suggestion: {alternative_suggestion}")
                        logger.debug(f"Reward Function\n{response_content}")

                        if alternative_suggestion:
                            explanations.append(alternative_suggestion)
                        else:
                            explanations.append(explanation)

                        outputs.append({
                            "completion_response": completion_response.model_dump(),
                            "parsed_reward": reward,
                            "time_taken": time.time() - time0
                        })

                        if len(outputs) > 0:
                            logger.debug(
                                f'response {i + 1}/{n_calls} received! time taken: {outputs[-1]["time_taken"]} seconds.')

                        break  # Successfully got the reward, exit the retry loop
                    else:
                        logger.error(f"Reward not found in completion response: {response_content}")
                        return 0, ""

                except Exception as e:
                    attempts += 1
                    logger.exception(f"Attempt {attempts}/{max_retries} failed to parse {response_content}.")
                    if attempts >= max_retries:
                        logger.exception(f"Failed to get valid reward after {max_retries} attempts")
                        # You might want to handle this case, e.g., by assigning a default reward
                        reward = 0  # or some other default value
                        explanation = "Failed to get valid reward after multiple attempts"
                        total_reward += reward
                        explanations.append(explanation)
                        outputs.append({
                            "completion_response": None,
                            "parsed_reward": reward,
                            "time_taken": time.time() - time0
                        })

        average_reward = total_reward / n_calls if n_calls > 0 else None
        entry = {
            **common_data,
            "outputs": outputs,
            "average_reward": average_reward
        }

        self.interactions.append(entry)
        if self.filename_value and self.log_to_file:
            save_to_json(self.interactions, self.filename_value)

        value_function_result = ValueFunctionResult.from_completions(completions, total_reward, n_calls)
        node.state.set_value_function_result(value_function_result)

        return average_reward, explanations[0]  # TODO: Handle more than one explanation?

    def eval_tree(self, tree, **kwargs):
        tree = minimize_string(tree)

        messages = self.create_message_tree(tree)
        completion_response = self.model_completion.get_completion(messages)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens_tree
            },
            "output": {
                "completion_response": completion_response.model_dump()
            }
        }

        if self.filename_tree and self.log_to_file:
            save_to_json(entry, self.filename_tree)

        return completion_response.choices[0].message.content



# class Completion(BaseModel):
#     model: str
#     input: list[dict] | None = None
#     response: dict[str, Any] | None = None
#     usage: Usage | None = None

#     @classmethod
#     def from_llm_completion(cls, input_messages: list[dict], completion_response: Any, model: str) -> Optional['Completion']:
#         if isinstance(completion_response, BaseModel):
#             response = completion_response.model_dump()
#         elif isinstance(completion_response, dict):
#             response = completion_response
#         elif hasattr(completion_response, '__dict__'):
#             # This will handle ModelResponse and similar objects
#             response = completion_response.__dict__
#         else:
#             logger.error(f"Unexpected completion response type: {type(completion_response)}")
#             return None

#         usage = Usage.from_completion_response(completion_response, model)
#         return cls(
#             model=model,
#             input=input_messages,
#             response=response,
#             usage=usage,
#         )

# class ValueFunctionResult(BaseModel):
#     completions: List[Completion] = Field(default_factory=list, description="List of completions done in the get_reward function")
#     total_reward: float = Field(0.0, description="Sum of rewards from all completions")
#     n_calls: int = Field(0, description="Number of calls made to the value function")
#     average_reward: float = Field(0.0, description="Average reward (total_reward / n_calls)")

#     @classmethod
#     def from_completions(cls, completions: List[Completion], total_reward: float, n_calls: int):
#         average_reward = total_reward / n_calls if n_calls > 0 else 0.0
#         return cls(
#             completions=completions,
#             total_reward=total_reward,
#             n_calls=n_calls,
#             average_reward=average_reward
#         )

"""
4. Identify the most promising path(s) in the tree, taking into account:
   - The cumulative values of the nodes
   - The relevance of actions to the {ISSUE_NAME_TREE}
   - The efficiency of the solution path
   
Each node contains:

node_info: ID, number of visits, and cumulative value
trajectory: A record of the state transitions and actions taken
children: Subsequent states resulting from the current action

* Examine the Search Tree structure, paying close attention to:
   - Each node's information (ID, visits, cumulative value)
   - The trajectory of state transitions and actions taken
"""