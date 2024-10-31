import json
import logging
import os
import time

from moatless.benchmark.swebench import create_workspace
from moatless.index.code_index import is_test
from moatless.testbed.sdk import TestbedSDK
from moatless.trajectory import Trajectory

from moatless.benchmark.utils import get_trajectories, get_moatless_instance
from moatless.verify.testbed import TestbedVerifier


failing_tests = []

def run_tests_instance(traj_dir):
    trajectory = Trajectory.load(f"{traj_dir}/trajectory.json", skip_workspace=False)
    instance_id = trajectory.info.get("instance_id")
    instance = get_moatless_instance(instance_id)

    print(f"\nSetting up {instance_id}")
    file_paths = [file["file_path"] for file in trajectory.transitions[-1].snapshot["file_context"]["files"]]
    test_files = [file_path for file_path in file_paths if is_test(file_path)]
    if not test_files:
        print(f"No test files found for {instance_id}. Files: {file_paths}")
        return
    print(f"Found {len(test_files)} test files")

    watch = time.time()
    testbed_sdk = TestbedSDK()
    testbed = testbed_sdk.get_or_create_testbed(instance_id=instance_id)

    log_dir = os.path.join(traj_dir, f"logs_{testbed.testbed_id}")
    os.makedirs(log_dir, exist_ok=True)
    verifier = TestbedVerifier(testbed=testbed_sdk, testbed_id=testbed.testbed_id, repository=trajectory.workspace.file_repo,
                               instance=instance, log_dir=log_dir)

    startup_time = time.time() - watch

    watch = time.time()

    test_runs = {}
    state = trajectory._transitions.get(5)
    trajectory.restore_from_snapshot(state)

    result = verifier.testbed.run_evaluation(testbed_id=verifier.testbed_id, patch=None)
    print(result.output)

    return
    for i, test_file in enumerate(test_files):
        print(f"Running tests from {test_file} ({i+1}/{len(test_files)})")
        # result = verifier.run_tests([test_file])

        result = verifier.testbed.run_evaluation(testbed_id=verifier.testbed_id, patch=None)
        print(result.output)

    test_time = time.time() - watch


def generate_csv(runs):
    with open("test_results.csv", "w") as f:
        f.write("instance_id,startup_time,test_time,total_tests,failed_tests,errors\n")

        for run in runs:
            if not run:
                continue

            errors = [
                test
                for tests in run["tests"].values()
                for test in tests
                if test["status"] == "ERROR"
            ]
            failures = [
                test
                for tests in run["tests"].values()
                for test in tests
                if test["status"] == "ERROR"
            ]

            f.write(
                f"{run['instance_id']},{run['startup_time']},{run['test_time']},{run['total_tests']},{len(failures)},{len(errors)}\n"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    run_tests_instance("/home/albert/repos/albert/sw-planner-2/trajs/evaluations/20240917_one_easy_per_repo_gpt-4o-mini/sphinx-doc__sphinx-8713")

def run_dir(traj_dir):
    for root, _, files in os.walk(traj_dir):
        if "trajectory.json" in files:
            trajectory_path = os.path.join(root, "trajectory.json")

    trajectories = get_trajectories(
        "/home/albert/repos/albert/swe-planner/trajs/evaluations/20240906_moatless_claude-3.5-sonnet",
        skip_workspace=True,
    )
    for trajectory in trajectories:
        run = run_tests_instance(trajectory)

        try:
            if trajectory.info.get("status") != "resolved":
                runs.append(run)
                # Write results to JSON file after each completed run
                with open("tests.json", "w") as f:
                    json.dump(runs, f, indent=2)
                logging.info(
                    f"Completed and saved results for {trajectory.info.get("instance_id")}"
                )
        except Exception as exc:
            logging.exception(
                f"{trajectory.info.get('instance_id')} generated an exception"
            )

    logging.info("All instances completed")
