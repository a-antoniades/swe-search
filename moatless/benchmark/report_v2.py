import hashlib
from collections import defaultdict
import json
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd

from moatless.edit.plan import Review, RequestCodeChange
from moatless.find.search import Search
from moatless.index.code_index import is_test
from moatless.benchmark.utils import (
    has_identified_spans,
    has_identified_files, count_identified_files, count_identified_spans, get_missing_files, get_moatless_instance,
)
from moatless.file_context import FileContext, RankedFileSpan, ContextFile
from moatless.trajectory import Trajectory, TrajectoryState
from moatless.schema import TestResult, TestStatus
from moatless.state import AgenticState, Content, Rejected
from pydantic import BaseModel, Field
from moatless.edit.plan_v2 import RequestCodeChange, RequestMoreContext

logger = logging.getLogger(__name__)


class Flag(BaseModel):
    state_name: Optional[str] = Field(None)
    state_id: Optional[int] = Field(None)
    message: str


class StateStats(BaseModel):
    status: str = ""
    iterations: int = 0
    rejected: int = 0
    cost: float = 0
    found_spans: int = 0
    found_files: int = 0
    result_spans: int = 0
    result_files: int = 0
    found_spans_details: Dict[str, List[str]] = {}


class SearchStats(StateStats):
    p_query: int = 0
    p_file: int = 0
    p_code: int = 0
    p_class: int = 0
    p_function: int = 0


class CodingStats(StateStats):
    review: bool = False
    edit_retries: int = 0
    plan_retries: int = 0
    edited: bool = False

    rejected: int = 0
    largest_span: Optional[int] = None
    smallest_span: Optional[int] = None
    has_diff: bool = False
    lint: bool = False
    lints: str = ""


class TrajectoryStats(BaseModel):
    """
    Stats for one finished trajectory.
    """

    state_id: int
    resolved: Optional[bool] = None
    status: Optional[str] = None
    message: Optional[str] = None
    reward: Optional[float] = None
    avg_reward: float = 0

    cost: float = 0
    iterations: int = 0
    transitions: int = 0
    rejections: int = 0
    retries: int = 0

    identify_status: str = ""
    search_status: str = ""

    has_diff: bool = False
    llm_monkey_status: Optional[str] = None

    edits: int = 0
    test_edits: int = 0
    failed_edits: int = 0

    test_files_with_spans: int = 0
    missing_test_files: int = 0

    max_tests_run: int = 0
    max_failed_tests: int = 0
    initial_failed_tests: Optional[int] = None
    final_failed_tests: Optional[int] = None

    largest_span: Optional[int] = None
    smallest_span: Optional[int] = None

    test_count: int = 0
    fail_to_pass_count: int = 0
    pass_to_pass_count: int = 0

    duplicate_children: int = 0
    duplicate_finished_children: int = 0


class BenchmarkResult(BaseModel):
    instance_id: str
    trajectory_path: str

    duration: float = 0
    total_cost: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    resolved_by: int = 0
    llmonkeys_rate: Optional[float] = None

    transitions: int = 0

    trajectories: List[TrajectoryStats] = []

    # MCTS
    all_transitions: int = 0
    solutions: int = 0
    resolved_solutions: int = 0
    failed_solutions: int = 0
    rejected_solutions: int = 0

    duplicated_search_actions: int = 0

    resolved_max_reward: float | None = None
    failed_max_reward: float | None = None

    expected_spans: int = 0
    expected_files: int = 0
    expected_spans_details: Dict[str, List[str]] = {}

    expected_test_files: List[str] = []
    found_test_files: List[str] = []
    missing_test_files: int = 0

    max_verification_issues: int = 0
    final_verification_issues: int = 0

    test_count: int = 0
    fail_to_pass_count: int = 0
    pass_to_pass_count: int = 0

    alternative_solutions: int = 0
    resolved: bool = False
    error: str = ""
    status: str = ""

    flags: List[Flag] = []

    search: SearchStats = SearchStats()
    identify: StateStats = StateStats()
    coding: CodingStats = CodingStats()
    decide: StateStats = StateStats()


def create_sha256_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()


def filter_test_code_from_diff(diff: str) -> str:
    filtered_diff = []
    in_test_file = False

    for line in diff.splitlines():
        if line.startswith("diff --git"):
            in_test_file = "tests/" in line or "test_" in line
        if not in_test_file:
            filtered_diff.append(line)

    return "\n".join(filtered_diff)


def create_trajectory_stats(trajectory_state: TrajectoryState, instance: dict, evaluation_result: dict | None = None) -> TrajectoryStats:
    result = TrajectoryStats(state_id=trajectory_state.state.id)

    trajectory_state.snapshot = trajectory_state.snapshot or {}
    if trajectory_state.snapshot.get("repository"):
        result.has_diff = bool(trajectory_state.snapshot["repository"].get("patch").strip())

        if result.has_diff:
            diff = trajectory_state.snapshot["repository"].get("patch")
            if not diff.endswith("\n"):
                diff += "\n"
            diff_hash = create_sha256_hash(diff)

            # Filter out test code from the diff
            filtered_diff = filter_test_code_from_diff(diff)
            filtered_diff_hash = create_sha256_hash(filtered_diff)

            for patch_hash in instance.get("llm_monkeys", {}).get("resolved_patches", []):
                if patch_hash == diff_hash or patch_hash == filtered_diff_hash:
                    result.llm_monkey_status = "resolved"

            if not result.llm_monkey_status:
                for patch_hash in instance.get("llm_monkeys", {}).get("unresolved_patches", []):
                    if patch_hash == diff_hash or patch_hash == filtered_diff_hash:
                        result.llm_monkey_status = "unresolved"

    terminal_state = trajectory_state.state

    if evaluation_result:
        result.resolved = evaluation_result.get("resolved")
        test_status = evaluation_result.get("tests_status")
        if test_status:
            result.fail_to_pass_count = len(test_status["fail_to_pass"]["failure"])
            result.pass_to_pass_count = len(test_status["pass_to_pass"]["failure"])
            result.test_count = (len(test_status["fail_to_pass"]["failure"])
                                 + len(test_status["pass_to_pass"]["failure"])
                                 + len(test_status["fail_to_pass"]["success"])
                                 + len(test_status["pass_to_pass"]["success"]))

    if terminal_state.visits:
        result.reward = terminal_state.visits[0].value

    if terminal_state.name == "Finished":
        result.status = "Finished"

    elif terminal_state.name == "Rejected":
        result.status = "Rejected"

        if hasattr(terminal_state, "message") and terminal_state.message:
            result.message = terminal_state.message

        if hasattr(terminal_state, "error") and terminal_state.error:
            if result.message:
                result.message += f" "
            result.message += f"Error {terminal_state.error}"
    else:
        result.status = "Abandoned"

    current_state = terminal_state
    while current_state is not None:
        result.transitions += 1

        if isinstance(current_state, AgenticState):
            state: AgenticState = current_state
            state_name = state.name

            # Update iterations and cost for the specific state
            if state_name in ["Search", "PlanToCode"]:
                result.iterations += 1

            if state.response and state.response.trigger == "reject":
                result.rejections += 1

            if len(state._actions) > 1:
                result.retries += 1

            if state._actions:
                for action in state._actions:
                    if action.completion and action.completion.usage:
                        result.cost += (
                            action.completion.usage.completion_cost
                        )

            if hasattr(state, "test_results") and state.test_results is not None:
                failed_test_count = sum(
                    1 for test in state.test_results if test.status in [TestStatus.FAILED, TestStatus.ERROR])

                result.initial_failed_tests = failed_test_count

                if len(state.test_results) > result.max_tests_run:
                    result.max_tests_run = len(state.test_results)

                if failed_test_count > result.max_failed_tests:
                    result.max_failed_tests = failed_test_count

                if result.final_failed_tests is None:
                    result.final_failed_tests = failed_test_count

            if state_name == "EditCode" and state.action_request:
                if not result.largest_span or state.end_line - state.start_line > result.largest_span:
                    result.largest_span = state.end_line - state.start_line

                if not result.smallest_span or state.end_line - state.start_line < result.smallest_span:
                    result.smallest_span = state.end_line - state.start_line

                if state.response and state.response.output:
                    output = state.response.output
                    if not output.get("diff"):
                        result.failed_edits += 1
                    else:
                        result.edits += 1
                        if is_test(state.file_path):
                            result.test_edits += 1

            if state_name == "IdentifyCode" and state.action_request:
                identified_spans = {}
                if not state.action_request.identified_spans:
                    logger.info(f"No action request found in IdentifyCode state: {state}")
                else:
                    for span in state.action_request.identified_spans:

                        if span.file_path not in identified_spans:
                            identified_spans[span.file_path] = []

                        for span_id in span.span_ids:
                            identified_spans[span.file_path].append(span_id)

                search_results_spans = {}
                for ranked_span in state.ranked_spans:
                    if isinstance(ranked_span, RankedFileSpan):
                        ranked_span = ranked_span.model_dump()

                    if ranked_span["file_path"] not in search_results_spans:
                        search_results_spans[ranked_span["file_path"]] = []
                    search_results_spans[ranked_span["file_path"]].append(
                        ranked_span["span_id"]
                    )

                test_files: List[Tuple[str, list[str]]] = [(file_path, span_ids) for file_path, span_ids in
                                                           identified_spans.items() if is_test(file_path)]

                result.test_files_with_spans = len(
                    [file for file, span_ids in test_files if len(set(span_ids) - {"imports"}) > 0])

                missing_test_files = get_missing_files(
                    instance["test_file_spans"], [file for file, span_ids in test_files]
                )

                result.missing_test_files = len(missing_test_files)
                result.identify_status = get_span_status(instance, identified_spans)
                result.search_status = get_span_status(instance, search_results_spans)

            # Add check for duplicate children
            duplicate_actions, duplicate_finished = find_duplicate_children(state)
            result.duplicate_children += duplicate_actions
            result.duplicate_finished_children += duplicate_finished

        current_state = current_state.previous_state

    return result


def get_span_status(instance: Dict, identified_spans: Dict[str, List[str]]):
    expected_spans = instance.get("expected_spans", {})
    alternative_solutions = [expected_spans]
    for resolved_by in instance.get("resolved_by", []):
        if (
                "alternative_spans" in resolved_by
                and resolved_by["alternative_spans"] not in alternative_solutions
        ):
            alternative_solutions.append(resolved_by["alternative_spans"])

    if has_identified_spans(alternative_solutions, identified_spans):
        return "found_spans"
    elif has_identified_files(alternative_solutions, identified_spans):
        return "found_files"
    else:
        return "missing_files"


def to_result(
    instance: Dict, trajectory: Trajectory, **kwargs
) -> BenchmarkResult:
    info = trajectory._info

    if instance is None:
        instance = get_moatless_instance(info["instance_id"])

    selected_transition_ids = []
    current_state = trajectory.get_current_state()
    while current_state:
        selected_transition_ids.append(current_state.id)
        current_state = current_state.previous_state

    logger.debug(f"Selected transitions: {selected_transition_ids}")

    try:
        expected_spans = instance.get("expected_spans", {})
        expected_files = list(expected_spans.keys())
        expected_spans_details = expected_spans

        alternative_solutions = []
        for resolved_by in instance.get("resolved_by", []):
            if (
                "alternative_spans" in resolved_by
                and resolved_by["alternative_spans"] not in alternative_solutions
            ):
                alternative_solutions.append(resolved_by["alternative_spans"])

        if info.get("error"):
            status = "error"
        elif info.get("resolved", False):
            status = "resolved"
        elif isinstance(trajectory.get_current_state(), Rejected):
            status = "rejected"
        elif info.get("status") == "finished":
            status = "failed"
        else:
            status = "running"

        total_usage = trajectory.total_usage()

        result = BenchmarkResult(
            instance_id=instance["instance_id"],
            trajectory_path=trajectory._persist_path,
            duration=info.get("duration", 0),
            total_cost=total_usage.completion_cost,
            prompt_tokens=total_usage.prompt_tokens,
            completion_tokens=total_usage.completion_tokens,
            resolved_by=len(instance.get("resolved_by", [])),
            llmonkeys_rate=instance.get("llm_monkeys", {}).get("resolved_rate", 0),
            transitions=len(selected_transition_ids),
            all_transitions=len(trajectory.transitions),
            solutions=0,
            resolved_solutions=0,
            expected_spans=sum(len(spans) for spans in expected_spans.values()),
            expected_files=len(expected_files),
            expected_spans_details=expected_spans_details,
            alternative_solutions=len(alternative_solutions),
            duplicated_search_actions=len(find_duplicated_actions(trajectory.transitions)),
            status=status,
            search=SearchStats(),
            identify=StateStats(),
            coding=CodingStats(),
            decide=StateStats(),
        )

        search_results_spans: Dict[str, List[str]] = {}
        identified_spans: Dict[str, List[str]] = {}
        pinned_spans: Dict[str, List[str]] = {}
        test_files: List[str] = []

        for transition in trajectory.transitions:
            if not transition.state.next_states:
                if hasattr(transition.state, "output") and transition.state.output and transition.state.output.get("evaluation_result"):
                    evaluation_result = transition.state.output["evaluation_result"]
                elif transition.state.id == trajectory.get_current_state().id:
                    # Set evaluation_result from final submission if none is set on the finished node for non MCTS
                    evaluation_result = info.get("evaluation_result")
                else:
                    evaluation_result = None

                result.trajectories.append(create_trajectory_stats(transition, instance, evaluation_result))

            if transition.state.name == "Finished":
                result.solutions += 1

                if transition.state.output and transition.state.output.get("resolved") is not None:
                    if transition.state.output.get("resolved"):
                        result.resolved_solutions += 1
                        if transition.state.visits and (result.resolved_max_reward is None or transition.state.visits[0].value > result.resolved_max_reward):
                            result.resolved_max_reward = transition.state.visits[0].value

                    else:
                        result.failed_solutions += 1
                        if transition.state.visits and (result.failed_max_reward is None or transition.state.visits[0].value > result.failed_max_reward):
                            result.failed_max_reward = transition.state.visits[0].value

            elif transition.state.name == "Rejected":
                result.rejected_solutions += 1

            if expected_spans:
                if (
                    selected_transition_ids
                    and transition.id not in selected_transition_ids
                ):
                    continue

                state: AgenticState = transition.state
                state_name = state.name

                # Update iterations and cost for the specific state
                if state_name in ["search", "identify", "decide", "plan", "edit"]:
                    current_state_stats = getattr(result, state_name)
                    current_state_stats.iterations += 1

                    if current_state.response.trigger == "reject":
                        current_state_stats.rejected += 1

                    if state._actions:
                        for action in state._actions:
                            if action.completion and action.completion.usage:
                                current_state_stats.cost += (
                                    action.completion.usage.completion_cost
                                )

                    # Update the state stats in the result object
                    setattr(result, state_name, current_state_stats)

                if state_name == "SearchCode":
                    if state.action_request:
                        for search_request in state.action_request.search_requests:
                            if search_request.query:
                                result.search.p_query += 1
                            if search_request.file_pattern:
                                result.search.p_file += 1
                            if search_request.code_snippet:
                                result.search.p_code += 1
                            if search_request.class_names:
                                result.search.p_class += 1
                            if search_request.function_names:
                                result.search.p_function += 1

                    if state.outcome and "ranked_spans" in state.outcome:
                        for ranked_span in state.outcome["ranked_spans"]:
                            if isinstance(ranked_span, RankedFileSpan):
                                ranked_span = ranked_span.model_dump()

                            result.search.result_spans += 1
                            if ranked_span["file_path"] not in search_results_spans:
                                search_results_spans[ranked_span["file_path"]] = []
                                result.search.result_files += 1
                            search_results_spans[ranked_span["file_path"]].append(
                                ranked_span["span_id"]
                            )

                    result.search.found_spans = sum(len(spans) for spans in search_results_spans.values())
                    result.search.found_files = len(search_results_spans)
                    result.search.found_spans_details = search_results_spans
                    set_found_status(
                        expected_spans,
                        alternative_solutions,
                        search_results_spans,
                        result.search,
                    )

                if state_name == "IdentifyCode" and state.action_request:
                    if not state.action_request.identified_spans:
                        logger.info(f"No action request found in IdentifyCode state: {state}")
                    else:
                        for span in state.action_request.identified_spans:
                            result.identify.result_spans += 1

                            if span.file_path not in identified_spans:
                                identified_spans[span.file_path] = []
                                result.identify.result_files += 1

                            for span_id in span.span_ids:
                                identified_spans[span.file_path].append(span_id)

                        set_found_status(
                            expected_spans,
                            alternative_solutions,
                            identified_spans,
                            result.identify,
                        )

                if state_name == "PlanToCode" and state.action_request:
                    if isinstance(state.action_request.action, Review):
                        result.coding.review = True

                    if len(state._actions) > 1:
                        result.coding.plan_retries += 1

                    if state.test_results:
                        failed_test_count = sum(
                            1 for test in state.test_results if test.status in [TestStatus.FAILED, TestStatus.ERROR])

                        if failed_test_count > result.max_verification_issues:
                            result.max_verification_issues = failed_test_count

                        result.final_verification_issues = failed_test_count

                if state_name == "EditCode" and state.action_request:
                    if len(state._actions) > 1:
                        result.coding.edit_retries += 1

                    if not result.coding.largest_span or state.end_line - state.start_line > result.coding.largest_span:
                        result.coding.largest_span = state.end_line - state.start_line

                    if not result.coding.smallest_span or state.end_line - state.start_line < result.coding.smallest_span:
                        result.coding.smallest_span = state.end_line - state.start_line

                    if transition.state.response.trigger == "reject":
                        result.coding.rejected += 1
                        result.flags.append(Flag(state_name=state_name, state_id=state.id, message=f"Change rejected with reason: {state.response.output.get('message')}"))

                    if state.response and state.response.output:
                        output = state.response.output
                        if output.get("diff"):
                            result.coding.has_diff = True

                if state_name in ["Finished", "Rejected"]:
                    for file in transition.snapshot["file_context"]["files"]:
                        if is_test(file["file_path"]):
                            test_files.append(file["file_path"])
                            continue

                        pinned_spans = {}
                        for span in file["spans"]:
                            if not span.get("pinned", False):
                                continue

                            if file["file_path"] not in pinned_spans:
                                pinned_spans[file["file_path"]] = []

                            pinned_spans[file["file_path"]].append(span["span_id"])

        missing_tests = get_missing_files(
            instance["test_file_spans"], test_files
        )
        result.missing_test_files = len(missing_tests)
        result.expected_test_files = list(instance["test_file_spans"].keys())

        if missing_tests:
            found_tests_str = ", ".join(test_files)
            expected_tests_str = ", ".join(instance["test_file_spans"].keys())
            result.flags.append(Flag(message=f"Missing tests. Found {found_tests_str}. Expected {expected_tests_str}."))

        result.found_test_files = test_files

        if "evaluation_result" in info:
            test_status = info["evaluation_result"]["tests_status"]
            result.fail_to_pass_count = len(test_status["fail_to_pass"]["failure"])
            result.pass_to_pass_count = len(test_status["pass_to_pass"]["failure"])
            result.test_count = (len(test_status["fail_to_pass"]["failure"])
                                 + len(test_status["pass_to_pass"]["failure"])
                                 + len(test_status["fail_to_pass"]["success"])
                                 + len(test_status["pass_to_pass"]["success"]))

        if result.pass_to_pass_count > 0 and result.final_verification_issues == 0:
            result.flags.append(Flag(message="Missed failing Pass to pass tests."))

        set_found_status(
            expected_spans,
            alternative_solutions,
            pinned_spans,
            result.coding,
        )

        if "error" in info:
            result.error = info["error"].split("\n")[0]
        else:
            result.error = ""

    except Exception as e:
        raise e

    return result


def find_duplicated_actions(transitions: List[TrajectoryState]) -> List[List[int]]:
    state_by_action = defaultdict(list)
    duplicates = []
    finished_states_by_parent = defaultdict(list)

    for transition in transitions:
        action_key = None
        if transition.state.name == "SearchCode" and isinstance(transition.state.action_request, Search):
            action_key = generate_search_action_key(transition.state.action_request.search_requests)
        elif transition.state.name == "PlanToCode" and transition.state.action_request:
            action_key = generate_plan_action_key(transition.state.action_request.action)

        if action_key:
            state_by_action[action_key].append(transition.state.id)

        # Check for multiple Finished states with the same parent
        if transition.state.name == "Finished" and transition.state.previous_state:
            parent_id = transition.state.previous_state.id
            finished_states_by_parent[parent_id].append(transition.state.id)

    for states in state_by_action.values():
        if len(states) > 1:
            duplicates.append(states)

    # Add duplicates for multiple Finished states with the same parent
    for parent_id, finished_states in finished_states_by_parent.items():
        if len(finished_states) > 1:
            duplicates.append(finished_states)

    return duplicates


def find_duplicate_children(state: AgenticState) -> Tuple[int, int]:
    child_actions = defaultdict(list)
    finished_children = []
    
    for child in state.next_states:
        if child.name == "Finished":
            finished_children.append(child.id)
        elif child.name in ["SearchCode", "PlanToCode"]:
            action_key = None
            if child.name == "SearchCode" and isinstance(child.action_request, Search):
                action_key = generate_search_action_key(child.action_request.search_requests)
            elif child.name == "PlanToCode" and child.action_request:
                action_key = generate_plan_action_key(child.action_request.action)
            
            if action_key:
                child_actions[action_key].append(child.id)
    
    duplicate_actions = sum(1 for actions in child_actions.values() if len(actions) > 1)
    duplicate_finished = len(finished_children) > 1
    
    return duplicate_actions, int(duplicate_finished)

def generate_search_action_key(search_requests: List) -> str:
    key_parts = []
    for request in search_requests:
        if request.class_names or request.function_names or request.file_pattern or request.code_snippet:
            key_part = (
                f"c:{','.join(sorted(request.class_names or []))}|"
                f"f:{','.join(sorted(request.function_names or []))}|"
                f"p:{request.file_pattern or ''}|"
                f"s:{request.code_snippet or ''}"
            )
        else:
            key_part = f"q:{request.query or ''}"
        key_parts.append(key_part)
    return "|".join(sorted(key_parts))

def generate_plan_action_key(action) -> str:
    if isinstance(action, RequestCodeChange):
        return f"RequestCodeChange:{action.file_path}:{action.start_line}:{action.end_line}"
    elif isinstance(action, RequestMoreContext):
        code_spans = []
        for file in action.files:
            span = f"{file.file_path}:{file.start_line or ''}:{file.end_line or ''}:{','.join(sorted(file.span_ids))}"
            code_spans.append(span)
        return f"RequestMoreContext:{';'.join(sorted(code_spans))}"
    return None

def set_found_status(
    expected_spans, alternative_solutions, identified_spans, result_stats: StateStats
):
    result_stats.result_spans = sum(len(spans) for spans in identified_spans.values())
    result_stats.result_spans = len(identified_spans)
    result_stats.found_files = count_identified_files(expected_spans, identified_spans)
    result_stats.found_spans = count_identified_spans(expected_spans, identified_spans)
    result_stats.found_spans_details = identified_spans

    expected_files = list(expected_spans.keys())
    if result_stats.found_spans == sum(len(spans) for spans in expected_spans.values()):
        result_stats.status = "expected_spans"
    elif has_identified_spans(alternative_solutions, identified_spans):
        result_stats.status = "alternative_spans"
    elif result_stats.found_files == len(expected_files):
        result_stats.status = "expected_files"
    elif has_identified_files(alternative_solutions, identified_spans):
        result_stats.status = "alternative_files"
    else:
        result_stats.status = "missing_spans"

def read_reports(report_path: str) -> List[BenchmarkResult]:
    with open(report_path, "r") as f:
        data = json.load(f)

    results = [BenchmarkResult.model_validate(item) for item in data]
    return results



def trajs_to_df(trajectories: List[Trajectory], report_mode: str | None = None) -> pd.DataFrame:
    results = [to_result(None, trajectory) for trajectory in trajectories]
    return to_dataframe(results, report_mode)


def to_trajectory_dataframe(results: List[BenchmarkResult]):
    result_dicts = []
    for result in results:
        for traj_result in result.trajectories:
            result_dict = {"instance_id": result.instance_id, "resolved_instance": result.resolved, "resolved_by": result.resolved_by, "llmonkeys_rate": result.llmonkeys_rate}
            result_dict.update(traj_result.model_dump())
            result_dicts.append(result_dict)

    return pd.DataFrame(result_dicts)

def to_dataframe(results: list[BenchmarkResult], report_mode: str | None = None, previous_report: dict = None) -> pd.DataFrame:
    state_keys = ["search", "identify", "decide", "coding"]
    rename_columns = False
    if report_mode == "code":
        state_keys = ["coding"]
    elif report_mode == "search_and_identify":
        state_keys = ["search", "identify"]
    elif report_mode in state_keys:
        state_keys = [report_mode]
        rename_columns = True

    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        general_keys = ["instance_id", "duration", "total_cost", "prompt_tokens", "completion_tokens", "resolved_by", "llmonkeys_rate", "status",
                        "transitions", "all_transitions", "solutions", "resolved_solutions", "failed_solutions", "rejected_solutions", "resolved_max_reward", "failed_max_reward",
                        "alternative_solutions", "resolved", "duplicated_search_actions", "expected_spans", "expected_files", "error", "trajectory_path"]

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if new_key.split(sep)[0] in state_keys or new_key in general_keys:
                if new_key in state_keys and isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))

            if k.endswith('_spans_details'):
                items.append((new_key, json.dumps(v)))

        if previous_report:
            items.append(("previously_resolved", d.get("instance_id", None) in previous_report["resolved"]))
        return dict(items)

    flattened_results = [flatten_dict(result.model_dump()) for result in results]

    df = pd.DataFrame(flattened_results)

    if rename_columns:
        df.columns = [
            col.replace(f"{report_mode}_", "")
            if col.startswith(f"{report_mode}_")
            else col
            for col in df.columns
        ]

    if report_mode == "mcts":
        mcts_cols = ["instance_id", "resolved_by", "llmonkeys_rate", "duration", "total_cost", "prompt_tokens", "completion_tokens", "status", "transitions",
                     "all_transitions", "solutions", "resolved_solutions", "failed_solutions", "rejected_solutions", "resolved_max_reward", "failed_max_reward", "duplicated_search_actions", "trajectory_path"]

        if previous_report:
            mcts_cols.append("previously_resolved")
        
        # Only select columns that exist in the DataFrame
        existing_cols = [col for col in mcts_cols if col in df.columns]
        df = df[existing_cols]
        
        # Add missing columns with NaN values
        missing_cols = set(mcts_cols) - set(existing_cols)
        for col in missing_cols:
            df[col] = pd.NA

    elif report_mode == "summary":
        summary_cols = ["instance_id", "duration", "total_cost", "status", "transitions", "expected_spans",
                        "expected_files", "search_status", "search_iterations", "identify_status",
                        "identify_iterations", "decide_status", "decide_iterations", "coding_status",
                        "coding_iterations", "coding_edit_retries", "coding_plan_retries"]
        df = df[summary_cols]

    # Reorder columns
    column_order = [
        "instance_id", "duration", "total_cost", "prompt_tokens", "completion_tokens", "resolved_by", "status", "resolved",
        "transitions", "all_transitions", "expected_spans", "expected_files", "alternative_solutions",
        "expected_spans_details", "error"
    ]

    state_columns = ["status", "iterations", "rejected", "cost", "found_spans", "found_files",
                     "result_spans", "result_files", "found_spans_details"]

    for state in state_keys:
        column_order.extend([f"{state}_{col}" for col in state_columns])

    # Add any remaining columns
    remaining_columns = [col for col in df.columns if col not in column_order]
    column_order.extend(remaining_columns)

    # Reorder the dataframe columns
    df = df.reindex(columns=[col for col in column_order if col in df.columns])
    return df

def read_results_from_json(file_path: str) -> List[BenchmarkResult]:
    with open(file_path, "r") as f:
        data = json.load(f)

    results = [BenchmarkResult.validate(item) for item in data]
    return results
