import concurrent.futures
import gc
import json
import logging
import os
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Tuple, Callable, Any, List
import threading

import litellm
from tqdm.auto import tqdm

from moatless.benchmark.report_v2 import to_result, BenchmarkResult, to_dataframe
from moatless.benchmark.solution_report import generate_solution_report
from moatless.trajectory import Trajectory
from moatless.transition_rules import TransitionRules, TransitionRule
from moatless.benchmark.swebench import (
    load_instance,
    create_workspace,
)
from moatless.benchmark.utils import (
    trace_metadata, get_moatless_instance,
)
from moatless.loop import AgenticLoop
from moatless.repository import GitRepository
from moatless.verify.testbed import TestbedVerifier
from testbed.sdk import TestbedSDK

logger = logging.getLogger(__name__)



class Evaluation:
    def __init__(
        self,
        evaluations_dir: str,
        evaluation_name: str,
        transitions: TransitionRules,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        repo_base_dir: str | None = None,
        report_mode: str | None = None,
        max_cost: float = 0.5,
        max_transitions: int = 25,
        use_perfect_file_context: bool = False,
        max_file_context_tokens: int = 16000,
        litellm_callback: Optional[str] = None,
        previous_trajectory_dir: Optional[str] = None,
        use_previous_trajectory_context: bool = False,
        retry_state: Optional[str] = None,
        retry_trajectory: bool = False,
        num_workers: int = 1,
        enable_mcts: bool = False,
        use_testbed: bool = False,
        eval_func: Callable[[dict, Trajectory], bool] = None,
        use_local_git_upstream: bool = False,
        **kwargs,
    ):
        self.evaluations_dir = evaluations_dir
        self.num_workers = num_workers
        self.report_mode = report_mode
        self.dataset_name = dataset_name
        self.evaluation_name = evaluation_name
        self.retry_trajectory = retry_trajectory
        self.use_local_git_upstream = use_local_git_upstream
        self.eval_func = eval_func

        self.use_testbed = use_testbed

        self.use_perfect_file_context = use_perfect_file_context

        self.max_file_context_tokens = max_file_context_tokens
        self.max_cost = max_cost
        self.max_transitions = max_transitions
        self.transitions = transitions

        self.enable_mcts = enable_mcts

        self.evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
        logger.info(f"Evaluation directory: {self.evaluation_dir}")
        if not os.path.exists(self.evaluation_dir):
            os.makedirs(self.evaluation_dir)

        self.predictions_path = f"{self.evaluation_dir}/all_preds.jsonl"

        self.repo_base_dir = repo_base_dir or os.getenv("REPO_DIR", "/tmp/repos")

        self.previous_trajectory_dir = previous_trajectory_dir
        self.use_previous_trajectory_context = use_previous_trajectory_context

        self.retry_state = retry_state
        self.retry_trajectory = retry_trajectory

        if litellm_callback:
            litellm.success_callback = [litellm_callback]
            litellm.failure_callback = [litellm_callback]

        self.status_file = f"{self.evaluation_dir}/status_summary.json"
        self.event_file = f"{self.evaluation_dir}/event_log.json"
        self.file_lock = threading.Lock()
        self.statuses = defaultdict(dict)
        self.events = defaultdict(list)

        self.kwargs = kwargs

    def update_status(self, instance_id: str, status: str):
        with self.file_lock:
            if instance_id not in self.statuses:
                self.statuses[instance_id] = {
                    "created": datetime.now().isoformat(),
                }

            self.statuses[instance_id].update({
                "last_updated": datetime.now().isoformat(),
                "status": status
            })
            self._save_statuses()

    def log_event(self, instance_id: str, event: str):
        with self.file_lock:
            self.events[instance_id].append({
                "timestamp": datetime.now().isoformat(),
                "event": event
            })
            self._save_events()

    def _save_statuses(self):
        with open(self.status_file, "w") as f:
            json.dump(self.statuses, f, indent=2)

    def _save_events(self):
        with open(self.event_file, "w") as f:
            json.dump(self.events, f, indent=2)

    def run_evaluation(
        self,
        split: str = "lite",
        resolved_by: Optional[int] = None,
        instance_ids: list[str] | None = None,
        ignore_repos: list[str] | None = None
    ):
        file_path = os.path.join(
            os.path.dirname(__file__), f"swebench_{split}_all_evaluations.json"
        )
        with open(file_path) as f:
            instances = json.load(f)

        instances = sorted(instances, key=lambda x: len(x["resolved_by"]), reverse=True)
        logger.info(f"Loaded {len(instances)} instances from {file_path}")

        if instance_ids:
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in instance_ids
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by instance_ids"
            )

        if resolved_by:
            instances = [
                instance
                for instance in instances
                if len(instance["resolved_by"]) >= resolved_by or (resolved_by == 1 and instance.get("llm_monkeys", {}).get("resolved_rate", 0) > 0)
            ]

            logger.info(
                f"Running evaluation for {len(instances)} instances filtered by resolved_by >= {resolved_by}"
            )
        
        if ignore_repos:
            instances = [
                instance
                for instance in instances
                if instance["repo"] not in ignore_repos
            ]

            if instances:
                logger.info(
                    f"Running evaluation for {len(instances)} instances after filtering by ignore_repos")

        return self._run_evaluation(instances)

    def run_single_instance(
        self,
        instance_id: str,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split="test",
    ) -> BenchmarkResult:
        instance = load_instance(instance_id, dataset, split)
        return self.evaluate_instance(instance)

    def evaluate_instance(self, instance: dict):       
        instance_id = instance["instance_id"]
        instance_dir = os.path.join(self.evaluation_dir, f"{instance_id}")
        trajectory_path = os.path.join(instance_dir, "trajectory.json")
        
        if not os.path.exists(self.evaluation_dir):
            os.makedirs(trajectory_path)

        if os.path.exists(trajectory_path) and not self.retry_trajectory:
            trajectory = Trajectory.load(trajectory_path, skip_workspace=True)
            status = trajectory.info.get("status")
            if status and status != "error":
                logger.info(f"Skipping {instance_id} because it has already been evaluated with status {status}")
                result = to_result(instance, trajectory)
                del trajectory
                gc.collect()
                return result

        logger.info(f"Evaluating {instance_id}")
        problem_statement = instance["problem_statement"]

        workspace = None
        testbed_sdk = None
        testbed = None
        loop = None
        trajectory = None
        result = None

        self.update_status(instance_id, "started")
        self.log_event(instance_id, "evaluate_instance_initiated")

        try:
            self.log_event(instance_id, "workspace_creation_started")
            workspace = create_workspace(
                instance,
                repo_base_dir=self.repo_base_dir,
                create_instance_dir=True,
                use_perfect_file_context=self.use_perfect_file_context,
                max_file_context_tokens=self.max_file_context_tokens,
                use_local_git_upstream=self.use_local_git_upstream,
            )
            self.log_event(instance_id, "workspace_created")

            if self.use_testbed:
                self.log_event(instance_id, "testbed_creation_started")
                workspace.verifier = TestbedVerifier(testbed_sdk=TestbedSDK(), repository=workspace.file_repo, instance=instance)
                self.log_event(instance_id, "testbed_created")

            previous_actions = self.get_previous_actions(instance_id)

            metadata = trace_metadata(
                instance_id=instance_id,
                session_id=self.evaluation_name,
                trace_name="moatless",
            )

            loop = AgenticLoop(
                transition_rules=self.transitions,
                initial_message=problem_statement,
                workspace=workspace,
                metadata=metadata,
                reset_mocks_at_state=self.retry_state,
                mocked_actions=previous_actions,
                continue_after_mocks=True,
                trajectory_path=trajectory_path,
                max_cost=self.max_cost,
                max_transitions=self.max_transitions,
                **self.kwargs,
            )

            info: dict[str, Any] = {
                "evaluation_name": self.evaluation_name,
                "instance_id": instance["instance_id"],
            }

            loop.trajectory.save_info(info)

            start_time = time.time()
            try:
                self.log_event(instance_id, "loop_execution_started")
                if self.enable_mcts:
                    response = loop.run_search()
                else:
                    response = loop.run()
                self.log_event(instance_id, "loop_execution_completed")

                info["status"] = response.status
            except Exception:
                info["error"] = traceback.format_exc()
                info["status"] = "error"
                logging.exception(f"Error in evaluation of {instance['instance_id']} ")
            finally:
                info["duration"] = time.time() - start_time
                usage = loop.total_usage()
                info["total_cost"] = usage.completion_cost
                info["prompt_tokens"] = usage.prompt_tokens
                info["completion_tokens"] = usage.completion_tokens

                if self.eval_func:
                    try:
                        info["eval_func"] = self.eval_func(instance, loop.trajectory)
                    except Exception:
                        logging.exception(f"Error in evaluation of {instance['instance_id']} ")

                diff = self.get_diff(workspace, instance)
                info["submission"] = diff

                self.process_evaluation_result(info, loop, diff, testbed)

                if info.get("resolved"):
                    logger.info(f"Resolved {instance['instance_id']} in {info['duration']} seconds")
                else:
                    logger.info(f"Could not resolve {instance['instance_id']} in {info['duration']} seconds")

                loop.trajectory.save_info(info)

            trajectory = loop.trajectory
            result = to_result(instance, trajectory)

            trajectory_light_path = os.path.join(
                self.evaluation_dir, instance_dir, "trajectory_light.json"
            )
            trajectory.persist(trajectory_light_path, exclude={"completion", "completions"})

            self.save_solution_report(trajectory, instance_id)
            self.save_prediction(instance_id, info.get("submission", ""))

            self.log_event(instance_id, "evaluation_completed")
            self.update_status(instance_id, info["status"])

            return result

        except Exception:
            logger.exception(f"Error in processing instance {instance_id}")
            self.log_event(instance_id, "evaluation_error")
            self.update_status(instance_id, "error")
            return None

        finally:
            # Clean up
            if workspace and workspace.file_repo:
                shutil.rmtree(workspace.file_repo.repo_dir, ignore_errors=True)
            if testbed:
                try:
                    testbed.destroy()
                except Exception:
                    logger.exception("Error deleting testbed")
            
            del workspace
            del testbed_sdk
            del testbed
            del loop
            del trajectory
            del result
            gc.collect()

    def get_previous_actions(self, instance_id):
        if not self.previous_trajectory_dir:
            return None

        previous_trajectory_path = os.path.join(self.previous_trajectory_dir, f"{instance_id}/trajectory.json")
        if os.path.exists(previous_trajectory_path):
            previous_trajectory = Trajectory.load(previous_trajectory_path)
            if self.use_previous_trajectory_context:
                return None
            else:
                return previous_trajectory.get_mocked_actions()
        else:
            # Version 1
            previous_trajectory_path = os.path.join(self.previous_trajectory_dir, f"{instance_id}.json")
            previous_trajectory = self.read_trajectory(previous_trajectory_path)
            if previous_trajectory:
                if self.use_previous_trajectory_context:
                    return None
                else:
                    return self.get_actions(previous_trajectory)
        return None

    def get_diff(self, workspace, instance):
        if isinstance(workspace.file_repo, GitRepository):
            test_patch_files = instance.get("test_file_spans", {}).keys()
            return workspace.file_repo.diff(ignore_paths=test_patch_files)
        else:
            output = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                cwd=workspace.file_repo.repo_dir,
            )
            diff = output.stdout if output else None
            if diff and not diff.endswith("\n"):
                diff += "\n"
            return diff

    def process_evaluation_result(self, info, loop, diff, testbed):
        if hasattr(loop.state, "output") and loop.state.output and loop.state.output.get("evaluation_result"):
            info["evaluation_result"] = loop.state.output["evaluation_result"]
            info["resolved"] = loop.state.output.get("resolved", None)
        elif diff and testbed:
            result = testbed.run_evaluation(patch=diff)
            info["resolved"] = result.resolved
            info["evaluation_result"] = result.model_dump()

    def save_solution_report(self, trajectory, instance_id):
        try:
            solution_report = generate_solution_report(trajectory)
            with open(f"{self.evaluation_dir}/{instance_id}/solution_report.json", "w") as f:
                json.dump(solution_report, f, indent=2)
        except Exception:
            logging.exception(f"Error when generating solution report for instance {instance_id}")

    def save_prediction(self, instance_id, submission):
        prediction = {
            "model_name_or_path": self.evaluation_name,
            "instance_id": instance_id,
            "model_patch": submission,
        }
        with open(self.predictions_path, "a") as file:
            json_string = json.dumps(prediction)
            file.write(json_string + "\n")

    def _to_csv_report(self, results: list[BenchmarkResult]):
        df = to_dataframe(results, self.report_mode)
        df.to_csv(
            f"{self.evaluation_dir}/result.csv",
            index=False,
            sep=",",
            decimal=",",
            quoting=1,
        )

    def _run_evaluation(self, instances: list[dict]):
        error = 0

        with open(self.predictions_path, "w") as file:
            file.write("")

        results = []

        logger.info(f"Processing {len(instances)} instances with {self.num_workers} workers")
        logger.info(self.transitions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.evaluate_instance, instance) for instance in instances]

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

            for future in pbar:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self._to_csv_report(results)
                        self._save_json_report(results)
                    else:
                        error += 1

                    stats = self._create_stats(results)
                    pbar.set_postfix(stats)

                except Exception:
                    error += 1
                    logger.exception("Error in processing instance")

        logger.info(f"Completed processing with {error} errors")
        self.update_status("all", "evaluation_completed")

    def _create_stats(self, results):
        stats = {}
        if results:
            stats["avg_duration"] = sum(r.duration for r in results) / len(results)
            stats["avg_cost"] = sum(r.total_cost for r in results) / len(results)
            stats["total_cost"] = sum(r.total_cost for r in results)

            identified = sum(
                1
                for r in results
                if r.status in ["identified", "planned", "edited", "resolved"]
            )
            resolved = sum(1 for r in results if r.status in ["resolved"])
            error = sum(1 for r in results if r.status == "error")

            if identified > 0:
                stats["identified"] = f"{(identified / len(results)) * 100:.2f}%"
            if resolved > 0:
                stats["resolved"] = f"{(resolved / len(results)) * 100:.2f}%"
            stats["error"] = error

        return stats

    def _save_json_report(self, results: list[BenchmarkResult]):
        json_results = [result.model_dump() for result in results]
        with open(f"{self.evaluation_dir}/report.json", "w") as f:
            json.dump(json_results, f, indent=2)

    def read_trajectory(self, path) -> dict | None:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        else:
            return None

    def get_actions(self, trajectory: dict):
        actions = []
        for transition in trajectory["transitions"]:
            for action in transition["actions"]:
                actions.append(action["action"])
        return actions


def create_evaluation_name(
    model: str,
    date,
    max_expansions = None,
    **kwargs,
):  
    if date:
        date_str = date
    else:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    model_name = model.split("/")[-1]
    model_name = f"{date_str}_{model_name}"
    if max_expansions:
        model_name += f"_max_exp{max_expansions}"
    for key, value in kwargs.items():
        model_name += f"_{key}_{value}"
    return model_name