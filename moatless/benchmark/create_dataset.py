import json
import os

import pandas as pd

from moatless.benchmark.swebench import setup_swebench_repo, sorted_instances
from moatless.benchmark.utils import get_file_spans_from_patch
from moatless.repository import FileRepository

dataset_path = "/home/albert/repos/albert/swe-planner/moatless/benchmark/swebench_lite_all_evaluations.json"
#dataset_path = "/home/albert/repos/albert/swe-planner/moatless/benchmark/swebench_verified_all_evaluations.json"


def read_predictions(pred_path: str):
    predictions = {}
    with open(pred_path) as f:
        for line in f.readlines():
            prediction = json.loads(line)
            predictions[prediction["instance_id"]] = prediction["model_patch"]
    return predictions


def generate_report(dataset_name: str = "princeton-nlp/SWE-bench_Lite"):
    results = {}

    experiments_dir = "/home/albert/repos/stuffs/experiments/evaluation/lite"

    runs = []
    for run_name in os.listdir(experiments_dir):
        if not os.path.exists(f"{experiments_dir}/{run_name}/all_preds.jsonl"):
            print(f"Missing {experiments_dir}/{run_name}/all_preds.jsonl")
            continue

        runs.append(
            (
                run_name,
                f"{experiments_dir}/{run_name}/all_preds.jsonl",
                f"{experiments_dir}/{run_name}/results/results.json",
            )
        )

    print(f"Found {len(runs)} runs")

    for run_name, prediction_file, result_file in runs:

        with open(result_file) as file:
            final_report = json.load(file)

        resolved_tasks = final_report["resolved"]
        predictions_by_id = read_predictions(prediction_file)

        results[run_name] = {
            "resolved_tasks": resolved_tasks,
            "predictions": predictions_by_id,
        }

    evaluation_dataset = []

    report = []

    instances = sorted_instances(split="test", dataset_name=dataset_name)
    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        print(f"Processing {instance_id} ({i}/{len(instances)})")
        repo_dir = setup_swebench_repo(instance, repo_base_dir="/tmp/repos_2")
        file_repo = FileRepository(repo_dir)

        expected_file_spans = get_file_spans_from_patch(file_repo, instance["patch"])
        test_file_spans = get_file_spans_from_patch(file_repo, instance["test_patch"])

        evaluation_instance = {
            "instance_id": instance_id,
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "problem_statement": instance["problem_statement"],
            "golden_patch": instance["patch"],
            "test_patch": instance["test_patch"],
            "fail_to_pass": instance["FAIL_TO_PASS"],
            "pass_to_pass": instance["PASS_TO_PASS"],
            "expected_spans": expected_file_spans,
            "test_file_spans": test_file_spans,
            "resolved_by": [],
            "alternative_spans": [],
        }

        for run_name, _, _ in runs:
            prediction = results[run_name]["predictions"].get(instance_id)

            if instance_id not in results[run_name]["resolved_tasks"]:
                continue

            file_spans = get_file_spans_from_patch(file_repo, prediction)

            is_different = False
            alternative_spans = {}
            for file_path, span_ids in file_spans.items():
                if file_path in expected_file_spans:
                    alternative_spans[file_path] = span_ids

                    if set(expected_file_spans[file_path]).difference(set(span_ids)):
                        is_different = True

            if is_different:
                evaluation_instance["alternative_spans"].append(
                    {"run_name": run_name, "spans": alternative_spans}
                )

            resolved = {
                "name": run_name,
                "updated_spans": file_spans,
                "alternative_spans": alternative_spans,
            }

            evaluation_instance["resolved_by"].append(resolved)

        report.append(
            {
                "instance_id": instance_id,
                "resolved_by": len(evaluation_instance["resolved_by"]),
            }
        )

        evaluation_dataset.append(evaluation_instance)

        with open(dataset_path, "w") as f:
            json.dump(evaluation_dataset, f, indent=2)

    return pd.DataFrame(report)


if __name__ == "__main__":
    df = generate_report("princeton-nlp/SWE-bench_Lite")
