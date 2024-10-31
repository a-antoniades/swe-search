import logging
import logging
import os
from typing import Tuple

from moatless.benchmark.report_v2 import to_result, to_dataframe, to_trajectory_dataframe
from moatless.benchmark.swebench import create_workspace
from moatless.benchmark.utils import get_moatless_instances, has_identified_files
from moatless.schema import RankedFileSpan
from moatless.trajectory import Trajectory

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def get_trajectories(dir: str) -> list[Tuple[Trajectory, str]]:
    trajectories = []
    for root, _, files in os.walk(dir):
        if "solutions" in root.split(os.path.sep) or "rews" in root.split(os.path.sep):
            continue

        if "trajectory.json" in files:
            trajectory_path = os.path.join(root, "trajectory.json")
            try:
                rel_path = os.path.relpath(root, dir)
                trajectory = Trajectory.load(trajectory_path, skip_workspace=True)
                trajectories.append((trajectory, rel_path))
            except Exception as e:
                logger.exception(f"Failed to load trajectory from {trajectory_path}: {e}")
    return trajectories

def generate_report(dir: str):
    trajectories = get_trajectories(dir)
    print(f"Trajectories: {len(trajectories)}")
    instances = get_moatless_instances()

    duplicted_sarch = 0
    results = []
    for trajectory, rel_path in trajectories:
        instance_id = trajectory.info["instance_id"]

        instance = instances.get(instance_id)
        if not instance:
            logger.error(f"Instance {instance_id} not found")
            continue

        expected_spans = instance.get("expected_spans", {})
        solutions = [expected_spans]
        for resolved_by in instance.get("resolved_by", []):
            if (
                    "alternative_spans" in resolved_by
                    and resolved_by["alternative_spans"] not in solutions
            ):
                solutions.append(resolved_by["alternative_spans"])

        for transition in trajectory.transitions:
            state = transition.state

            if state.name == "SearchCode":
                search_results_spans = {}
                if state.outcome and "ranked_spans" in state.outcome:
                    if not state.outcome["ranked_spans"]:
                        print(f"{instance_id} {state.id} search_status: no_spans")

                    for ranked_span in state.outcome["ranked_spans"]:
                        if isinstance(ranked_span, RankedFileSpan):
                            ranked_span = ranked_span.model_dump()

                        if ranked_span["file_path"] not in search_results_spans:
                            search_results_spans[ranked_span["file_path"]] = []
                        search_results_spans[ranked_span["file_path"]].append(
                            ranked_span["span_id"]
                        )

                if not has_identified_files(solutions, search_results_spans):
                    print(f"{instance_id} {state.id} search_status: no_files")

                    workspace = create_workspace(instance)

                    cloned_state = state.clone()
                    cloned_state._workspace = workspace
                    outcome = cloned_state.execute(mocked_action_request=state.action_request)

                    print(f"Action: {state.action_request.model_dump_json(indent=2)}")
                    print(f"Outcome: {outcome.model_dump_json(indent=2)}")




logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

#    df = to_dataframe(results)
#    print(df)
# directory = "/home/albert/repos/albert/swe-planner/trajs/evaluations/20240906_moatless_claude-3.5-sonnet"
directory = "/home/albert/repos/albert/sw-planner-2/trajs/evaluations/django_mcts_no_feedback/20240919_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_use_testbed_True"
generate_report(directory)
