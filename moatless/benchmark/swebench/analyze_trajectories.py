import logging
import os
import json

from moatless.benchmark.report_v2 import to_result
from moatless.benchmark.utils import get_moatless_instance, file_spans_to_dict, has_identified_spans, get_missing_spans, \
    get_moatless_instances, has_identified_files
from moatless.benchmark.swebench.utils import create_workspace
from moatless.benchmark.swebench.index_instances import evaluate_index
from moatless.trajectory import Trajectory


def load_trajectory(root_directory, instance_id):
    trajectory_path = os.path.join(root_directory, instance_id, "trajectory.json")
    if os.path.exists(trajectory_path):
        try:
            trajectory = Trajectory.load(trajectory_path, skip_workspace=False)
            return trajectory
        except Exception as e:
            print(f"Error loading trajectory from {trajectory_path}: {str(e)}")
    return None

def load_trajectories(root_directory):
    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    
    trajectories = []
    for subdir in subdirectories:
        trajectory_path = os.path.join(root_directory, subdir, "trajectory.json")
        if os.path.exists(trajectory_path):
            try:
                trajectory = Trajectory.load(trajectory_path, skip_workspace=True)
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error loading trajectory from {trajectory_path}: {str(e)}")
    
    return trajectories

def analyze(root_dir: str):
    instances = get_moatless_instances("verified")


    no_trajectory = []
    resolved_instances = []

    too_hard = []
    too_many_files = []

    failures = {
        "ingestion_failure": [],
        "missing_file_in_search_result": [],
        "missing_spans_in_search_result": [],
        "missing_file_in_identified_spans": [],
        "missing_spans_in_identified_spans": []
    }

    solved = {
        "solved_file_in_search_result": []
    }

    result_file = f"{root_dir}/result.json"
    if os.path.exists(result_file):
        with open(os.path.join(result_file)) as f:
            report = json.load(f)
    else:
        report = {"resolved_ids": []}

    for instance_id, instance in instances.items():
        if len(instance["resolved_by"]) == 0:
            too_hard.append(instance_id)
            continue

        if len(instance["expected_spans"].keys()) > 1:
            too_many_files.append(instance_id)
            continue

        if instance_id in report["resolved_ids"]:
            resolved_instances.append(instance_id)
            continue

        trajectory = load_trajectory(root_dir, instance_id)
        if not trajectory:
            print(f"Could not load trajectory for {instance_id}")
            no_trajectory.append(instance_id)
            continue
        trajectory.update_workspace_to_current_state()

        identified_spans = file_spans_to_dict(trajectory.workspace.file_context.to_files_with_spans())
        all_solutions = [instance["expected_spans"]]
        all_solutions.extend(instance["alternative_spans"])

        if not has_identified_spans(all_solutions, identified_spans):
            result = to_result(instance, trajectory)

            if not has_identified_files(all_solutions, result.search.found_spans_details):
                workspace = create_workspace(instance)
                missing_spans_in_search = get_missing_spans(instance["expected_spans"],
                                                            result.search.found_spans_details)

                expected_changes, all_matching_context_window, any_matching_context_window = evaluate_index(workspace.code_index, instance)

                print(f"\n\nInstance {instance['instance_id']}: Search fail")
                print(f"Expected files: {','.join(instance['expected_spans'].keys())}")
                print(f"All matching context window: {all_matching_context_window}, Any matching context window: {any_matching_context_window}")

                if not any_matching_context_window:
                    failures["ingestion_failure"].append(instance_id)
                else:
                    print(f"Missing: {json.dumps(missing_spans_in_search, indent=2)}")

                    found_in_search = {}

                    states = trajectory.get_states_by_name("SearchCode")
                    for i, search_state in enumerate(states):
                        print(f"Search Request {i}")

                        for search_request in search_state.action_request.search_requests:
                            search_response = workspace.code_index.search(
                                file_pattern=search_request.file_pattern,
                                query=search_request.query,
                                code_snippet=search_request.code_snippet,
                                class_names=search_request.class_names,
                                function_names=search_request.function_names,
                                max_tokens=16000 / len(search_state.action_request.search_requests),
                                max_results=250 / len(search_state.action_request.search_requests),
                            )
                            print(search_request.model_dump_json(indent=2, exclude_unset=True))
                            print("Search results")
                            print(search_response.model_dump_json(indent=2, exclude_unset=True))

                            for hit in search_response.hits:
                                file_path = hit.file_path
                                if file_path not in found_in_search:
                                    found_in_search[file_path] = []

                                for span in hit.spans:
                                    if span.span_id not in found_in_search[file_path]:
                                        found_in_search[file_path].append(span.span_id)

                    if has_identified_spans(all_solutions, found_in_search):
                        print(f"Found in search")
                        solved["file_in_search_result"].append(instance_id)
                    else:
                        print(f"Search results: {json.dumps(found_in_search, indent=2)}")

                    #print(f"\n\nInstance {instance['instance_id']}: File fail")
                    #print(f"Expected: {instance['expected_spans']}")
                    #print(f"Identified: {result.search.found_spans_details}")
                    #print(f"Missing files: {missing_spans_in_search.keys()}")

                    failures["missing_file_in_search_result"].append(instance_id)

            if not has_identified_spans(all_solutions, result.search.found_spans_details):
                missing_spans_in_search = get_missing_spans(instance["expected_spans"], result.search.found_spans_details)

                if False:
                    print(f"\n\nInstance {instance['instance_id']}: Search fail")
                    print(f"Expected spans: {instance['expected_spans']}")
                    print(f"Missing: {missing_spans_in_search}")

                    states = trajectory.get_states_by_name("SearchCode")
                    for i, search_state in enumerate(states):
                        print(f"Search Request {i}")
                        print(search_state.request.model_dump_json(indent=2))

                print(f"Search result: {result.search.found_spans_details}")
                failures["missing_spans_in_search_result"].append(instance_id)
            elif not has_identified_files(all_solutions, identified_spans):
                missing_spans = get_missing_spans(instance["expected_spans"], identified_spans)
                failures["missing_file_in_identified_spans"].append(instance_id)
            #    print(f"\n\nInstance {instance['instance_id']}: Identify fail")
            #    print(f"Expected: {instance['expected_spans']}")
            #    print(f"Identified: {identified_spans}")
            #    print(f"Missing: {missing_spans}")
            else:
                missing_spans = get_missing_spans(instance["expected_spans"], identified_spans)
                failures["missing_spans_in_identified_spans"].append(instance_id)

    print(f"Resolved instances: {len(resolved_instances)}")
    print(f"Instances too hard: {len(too_hard)}")
    print(f"Instances with too many expected files: {len(too_many_files)}")

    for key, value in failures.items():
        print(f"{key}: {len(value)}")

    print("Solved")
    for key, value in solved.items():
        print(f"{key}: {len(value)}")

# logging.basicConfig(level=logging.INFO)

root_dir = "/home/albert/repos/albert/moatless/evaluations/20240818_moatless_tools_verified"
analyze(root_dir)
