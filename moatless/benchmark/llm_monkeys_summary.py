import json
import logging
import traceback

from moatless import AgenticLoop
from moatless.benchmark.report_v2 import to_result
from moatless.benchmark.swebench import load_instance, create_workspace, get_moatless_instance
from moatless.repository import GitRepository
from moatless.search.reward import LLM_Value_Function
from moatless.settings import Settings
from moatless.transitions import search_and_code_transitions


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


"""
"gpt-4o-2024-08-06"
"""

global_params = {
    "model": "gpt-4o-2024-08-06",
    "temperature": 1.6,
    "max_tokens": 1000
}

with open("/home/albert/repos/stuffs/swe-bench-lite-samples/summary.json", "r") as f:
    summary = json.load(f)

value_function = LLM_Value_Function(log_to_file=False)


def run_instances(max_runs: int = 5):
    all_results = []
    for key, value in summary["instances_without_flaky_tests"].items():
        resolved_count = 0
        resolved_patches = set()
        all_patches = set()
        for run in value.values():
            if run["resolved"]:
                resolved_count += 1
                resolved_patches.add(run["patch_hash"])

            all_patches.add(run["patch_hash"])

        with open(f"llm_monkeys_classification.csv", "w") as f:
            f.write("instance_id,run_id,resolved,status,has_diff,reward\n")
            for res in all_results:
                f.write(
                    f"{res['instance_id']},{res['run_id']},{res['resolved']},{res['status']},{res['has_diff']},{res['reward']}\n")

        print(f"{key}: {resolved_count}/{len(value)} ({len(resolved_patches)}/{len(all_patches)})")

        if len(resolved_patches) > max_runs and len(all_patches) - len(resolved_patches) > max_runs:
            print(f"Running tests for {key}")
            result = run_test(key, max_runs=max_runs)
            all_results.extend(result)


def run_test(instance_id: str, max_runs: int = 5):
    test_runs = summary["instances_without_flaky_tests"][instance_id]

    resolved_patches = set()
    unresolved_patches = set()

    result = []

    for run_id, run in test_runs.items():
        if run["resolved"]:
            if len(resolved_patches) >= max_runs or run["patch_hash"] in resolved_patches:
                continue
        else:
            if len(unresolved_patches) >= max_runs or run["patch_hash"] in unresolved_patches:
                continue

        instance = get_moatless_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance {instance_id} not found")
        workspace = create_workspace(instance, repo_base_dir="/tmp/repos2")
        assert isinstance(workspace.file_repo, GitRepository)
        Settings.cheap_model = None  # To not use an LLM when generating commit messages

        trajectory_path = f"/home/albert/repos/stuffs/swe-bench-lite-samples/trajectories/{instance_id}/{run_id}.json"

        with open(trajectory_path, "r") as f:
            trajectory_dict = json.load(f)

        mocked_actions = []
        for transition in trajectory_dict["transitions"]:
            if "actions" in transition:
                for action in transition["actions"]:
                    mocked_actions.append(action["action"])

        try:
            loop = AgenticLoop(
                search_and_code_transitions(global_params=global_params), initial_message=instance["problem_statement"], workspace=workspace, mocked_actions=mocked_actions
            )
            response = loop.run()
            diff = loop.workspace.file_repo.diff()
            print(f"Did run trajectory with resulting diff:\n{diff}")

            reward = value_function.get_reward(
                state=loop.state,
                problem_statement=instance["problem_statement"],
                node_id=loop.state.id
            )

            print(f"Reward: {reward}, Resolved: {run['resolved']}")

            report = to_result(instance, loop.trajectory)

            result.append({
                "instance_id": instance_id,
                "run_id": run_id,
                "resolved": run["resolved"],
                "status": report["status"],
                "has_diff": bool(diff and diff.strip()),
                "reward": reward
            })
            if run["resolved"]:
                resolved_patches.add(run["patch_hash"])
            else:
                unresolved_patches.add(run["patch_hash"])
        except Exception as e:
            traceback.print_exc()

    return result

# analyze_instances()
run_instances(max_runs=3)