import logging

import networkx as nx

from moatless.benchmark.swebench import found_in_expected_spans, found_in_alternative_spans, found_in_alternative_files
from moatless.benchmark.utils import get_missing_files, get_moatless_instance
from moatless.edit.plan_v2 import RequestCodeChange, RequestMoreContext
from moatless.schema import TestStatus
from moatless.state import State
from moatless.trajectory import Trajectory

logger = logging.getLogger(__name__)

node_state_types = ["PlanToCode", "SearchCode", "Finished"]

def get_next_node_state(state: State):

    for next_state in state.next_states:
        if next_state.name in node_state_types:
            return next_state
        
        next_state = get_next_node_state(next_state)
        if next_state:
            return next_state
        
    return None


def build_graph(trajectory: Trajectory, hide_unvisited_nodes: bool = True):
    if "instance_id" in trajectory.info:
        instance = get_moatless_instance(trajectory.info.get("instance_id"))
    else:
        logger.warning("Instance ID not found in trajectory info")
        instance = None

    G = nx.DiGraph()

    mcts_mode = bool(trajectory.transitions[1].state.visits)
    if not mcts_mode:
        hide_unvisited_nodes = False

    node_index = 0

    for transition in trajectory.transitions:
        state = transition.state

        properties = {}

        if state.name == "Pending":
            G.add_node("Node0",
                       label=f"Node{node_index}",
                       name="Start",
                       visits=0,
                       value=0,
                       avg_reward=0,
           )
            continue

        if hide_unvisited_nodes:
            if state.name in node_state_types:
                if not state.visits:
                    continue
            else:
                node_state = get_next_node_state(state)
                if node_state and not node_state.visits:
                    continue

        if state.name in ["PlanToCode", "SearchCode"] and state.origin_state is None:
            # Use first created state as node
            state_params = []

            if transition.snapshot and transition.snapshot.get("file_context", {}).get("files"):
                identified_spans = {}
                for file_with_spans in transition.snapshot["file_context"]["files"]:
                    identified_spans[file_with_spans["file_path"]] = []
                    for span in file_with_spans["spans"]:
                        identified_spans[file_with_spans["file_path"]].append(span["span_id"])
                properties["expected_span_identified"] = found_in_expected_spans(instance, identified_spans)
                missing_files = get_missing_files(
                    instance["expected_spans"],
                    list(identified_spans.keys()),
                )
                if not missing_files:
                    properties["file_identified"] = True
                properties["alternative_span_identified"] = found_in_alternative_spans(instance, identified_spans)

            if hasattr(state, "test_results") and state.test_results is not None:
                failed_tests = len([test for test in state.test_results if test.status in [TestStatus.FAILED, TestStatus.ERROR]])
                properties["failed_tests"] = failed_tests
                properties["tests_run"] = len(state.test_results)

                state_params.append(f"Tests run: {len(state.test_results)}")
                state_params.append(f"Failed tests: {failed_tests}")

            sum_value = 0
            value = None
            for visit in state.visits:
                if visit.source_state_id == state.id:
                    value = visit.value
                
                sum_value += visit.value

            visits = len(state.visits) if state.visits else 0

            node_index += 1
            node_id = f"Node{state.id}"
            G.add_node(node_id,
                       type="node",
                       label=f"Node{state.id}",
                       name=state.name,
                       visits=visits,
                       state_params=state_params,
                       value=value,
                       sum_value=sum_value,
                       avg_reward=sum_value / visits if visits > 0 else 0,
                       **properties)
            if state.previous_state.id == 0:
                G.add_edge("Node0", node_id)
            else:
                G.add_edge(f"Action{state.previous_state.id}", node_id)
        elif state.name == "Finished":
            state_params = []

            resolved = None
            if state.output:
                if state.output.get("diff"):
                    diff_lines = state.output.get("diff").split("\n")
                    plus_lines = [line for line in diff_lines if line.startswith("+")]
                    minus_lines = [line for line in diff_lines if line.startswith("-")]
                    state_params.append(f"Diff: +{len(plus_lines)}, -{len(minus_lines)}")

                resolved = state.output.get("resolved", None)
                if state.output.get("evaluation_result", None):
                    if state.output.get("resolved"):
                        state_params.append("Status: Resolved")
                    else:
                        state_params.append("Status: Failed")

                    tests_status = state.output["evaluation_result"].get("tests_status", {})
                    f2p_fails = len(tests_status.get("fail_to_pass", {}).get("failure", []))
                    f2p_passes = len(tests_status.get("fail_to_pass", {}).get("success", []))
                    p2p_fails = len(tests_status.get("pass_to_pass", {}).get("failure", []))
                    p2p_passes = len(tests_status.get("pass_to_pass", {}).get("success", []))
                    state_params.append(f"f2p_passes: {f2p_passes}")
                    state_params.append(f"f2p_fails: {f2p_fails}")
                    state_params.append(f"p2p_passes: {p2p_passes}")
                    state_params.append(f"p2p_fails: {p2p_fails}")

            value = None
            for visit in state.visits:
                if visit.source_state_id == state.id:
                    value = visit.value
                
                sum_value += value

            visits = len(state.visits) if state.visits else 0

            node_index += 1
            node_id = f"Node{state.id}"
            G.add_node(node_id,
                       id=state.id,
                       type="node",
                       label=f"Node{state.id}",
                       name=state.name,
                       visits=visits,
                       sum_value=sum_value,
                       value=int(value) if value is not None else None,
                       resolved=resolved,
                       state_params=state_params,
                       avg_reward=sum_value / visits if visits > 0 else 0,
                       **properties)

            if state.previous_state.id == 0:
                G.add_edge("Node0", node_id)
            else:
                G.add_edge(f"Action{state.previous_state.id}", node_id)

        if hasattr(state, "action_request") and state.action_request:

            if hide_unvisited_nodes:
                node_state = get_next_node_state(state)
                if node_state and not node_state.visits:
                    continue
                    
            info_params = []
            warnings = []

            if len(state.actions) > 1:
                warnings.append(f"{len(state.actions)} retries")

            if state.name == "PlanToCode":
                action = state.action_request.action
                name = action.__class__.__name__

                if isinstance(action, RequestCodeChange):
                    info_params.append(f"File: {action.file_path}")
                    info_params.append(f"Lines {action.start_line}-{action.end_line}")
                    info_params.append(f"Instructions: {action.instructions}")

                    if state.outcome and state.outcome.get("message"):
                        warnings.append(f"Message: {state.outcome.get('message').strip()[:100]}")

                if isinstance(action, RequestMoreContext):
                    for file in action.files:
                        if file.start_line:
                            info_params.append(f"{file.file_path}: {file.start_line}-{file.end_line}")

                        if file.span_ids:
                            info_params.append(f"{file.file_path}: {file.span_ids}")

                    if state.outcome and state.outcome.get("message") and not state.outcome.get("updated_file_context", []):
                        warnings.append(f"Message: {state.outcome.get('message').strip()[:100]}")

            else:
                name = state.name

            if state.name == "EditCode":
                info_params.append(f"File: {state.file_path}")
                info_params.append(f"Lines {state.start_line}-{state.end_line}")

                if state.outcome.get("message"):
                    info_params.append(f"Message: {state.outcome.get('message').strip()[:100]}")

                if state.outcome.get("diff"):
                    diff_lines = state.outcome.get("diff").split("\n")
                    plus_lines = [line for line in diff_lines if line.startswith("+")]
                    minus_lines = [line for line in diff_lines if line.startswith("-")]
                    info_params.append(f"Diff: +{len(plus_lines)}, -{len(minus_lines)}")
                else:
                    warnings.append("No diff")

                if state.outcome.get("updated_file_context"):
                    spans = state.outcome.get("updated_file_context")[0].get("span_ids", [])
                    if len(spans) > 1:
                        warnings.append(f"Multiple spans edited: {spans}")

            if state.name == "SearchCode":

                for i, search_request in enumerate(state.action_request.search_requests):
                    info_params.append(f"Search {i + 1}:")

                    if search_request.file_pattern:
                        info_params.append(f"File pattern: {search_request.file_pattern}")

                    if search_request.query:
                        info_params.append(f"Query: {search_request.query}")

                    if search_request.code_snippet:
                        info_params.append(f"Code snippet: {search_request.code_snippet}")

                    if search_request.class_names:
                        info_params.append(f"Class names: {search_request.class_names}")

                    if search_request.function_names:
                        info_params.append(f"Function names: {search_request.function_names}")

            if state.name == "IdentifyCode":
                info_params.append(f"Search results: {len(state.ranked_spans)}")

                identified_spans = {}
                if not state.action_request.identified_spans:
                    warnings.append("No spans identified")
                else:
                    for span in state.action_request.identified_spans:

                        if span.file_path not in identified_spans:
                            identified_spans[span.file_path] = []

                        for span_id in span.span_ids:
                            identified_spans[span.file_path].append(span_id)

                    missing_files = get_missing_files(
                        instance["expected_spans"],
                        list(identified_spans.keys()),
                    )
                    if missing_files:
                        if found_in_alternative_files(instance, list(identified_spans.keys())):
                            info_params.append(f"Alternative files identified")
                        else:
                            warnings.append(f"Missing {missing_files} expected files")

                    if found_in_expected_spans(instance, identified_spans):
                        info_params.append("Expected spans identified")
                        properties["expected_span_identified"] = True
                    elif found_in_alternative_spans(instance, identified_spans):
                        info_params.append("Alternative spans identified")
                        properties["alternative_span_identified"] = True

            if state.outcome and state.outcome.get("updated_file_context", []):
                info_params.append(f"Spans:")
                for file_with_spans in state.outcome.get("updated_file_context", []):
                    if isinstance(file_with_spans, dict):
                        info_params.append(f"{file_with_spans['file_path']}: {len(file_with_spans['span_ids'])}")
                    else:
                        info_params.append(f"{file_with_spans.file_path}: {len(file_with_spans.span_ids)}")

            node_id = f"Action{state.id}"
            G.add_node(node_id,
                       type="action",
                       name=name,
                       info_params=info_params,
                       warnings=warnings,
                       **properties)

            if state.name in ["PlanToCode", "SearchCode"]:
                if state.origin_state:
                    G.add_edge(f"Node{state.origin_state.id}", node_id)
                else:
                    G.add_edge(f"Node{state.id}", node_id)
            else:
                G.add_edge(f"Action{state.previous_state.id}", node_id)


    return G


def print_graph(G):
    print("Graph Summary:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("\nNodes:")
    for node, data in G.nodes(data=True):
        print(f"  Node {node}:")
        for key, value in data.items():
            print(f"    {key}: {value}")
    print("\nEdges:")
    for from_node, to_node, data in G.edges(data=True):
        print(f"  {from_node} -> {to_node}")
        for key, value in data.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    instance = get_moatless_instance("django__django-14238")
    trajectory = Trajectory.load("/home/albert/repos/albert/swe-planner/trajs/evaluations/20240910_mcts_plan_v2_2/django__django-14238/trajectory.json", skip_workspace=True)

    G = build_graph(trajectory)
    print_graph(G)