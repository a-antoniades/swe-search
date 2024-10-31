import os
import logging
import glob
import json
import argparse
import tempfile
from pathlib import Path
import resource

from moatless.trajectory import Trajectory


def run_swebench_multi_solutions(
    dataset_name: str,
    split: str,
    instance_id: str,
    solutions_file: str,
    max_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    timeout: int
):
    """
    Run SWEBench evaluation for multiple proposed solutions for a specific instance.

    Args:
        dataset_name (str): Name of the dataset or path to JSON file.
        split (str): Split of the dataset.
        instance_id (str): The specific instance ID to evaluate.
        solutions_file (str): Path to the JSON file containing multiple solutions.
        max_workers (int): Maximum number of workers.
        force_rebuild (bool): Force rebuild of all images.
        cache_level (str): Cache level - remove images above this level.
        clean (bool): Clean images above cache level.
        open_file_limit (int): Open file limit.
        timeout (int): Timeout (in seconds) for running tests for each instance.
    """

    REQUIRED_ENTRIES = [
    "submitted_instances",
    "completed_instances",
    "resolved_instances",
    "solution"
    ]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set up the environment
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    # Load the solutions
    try:
        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: The solutions file {solutions_file} is not a valid JSON file.")
        return
    except FileNotFoundError:
        logging.error(f"Error: The solutions file {solutions_file} was not found.")
        return

    # Prepare the dataset
    dataset = load_swebench_dataset(dataset_name, split)
    instance = next((item for item in dataset if item['instance_id'] == instance_id), None)
    if not instance:
        logging.error(f"Error: Instance {instance_id} not found in the dataset.")
        return

    # Prepare for evaluation
    build_env_images(client, [instance], force_rebuild, max_workers)

    results = {}

    for node_id, solution in solutions.items():
        logging.info(f"Evaluating solution for node {node_id}")

        # Prepare prediction
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": f"solution_{node_id}",
            "model_patch": solution
        }

        # Create a temporary file instead of using StringIO
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump([prediction], temp_file)
            temp_file_path = temp_file.name

        # Run the evaluation
        run_id = f"run_{instance_id}_{node_id}"
        try:
            run_swebench(
                dataset_name=dataset_name,
                split=split,
                instance_ids=[instance_id],
                predictions_path=temp_file_path,  # Pass the temporary file path
                max_workers=max_workers,
                force_rebuild=force_rebuild,
                cache_level=cache_level,
                clean=clean,
                open_file_limit=open_file_limit,
                run_id=run_id,
                timeout=timeout
            )

            # Read the report
            report_file = Path(f"/tmp/{run_id}_1.json")
            if report_file.exists():
                logging.info(f"Report file found at {report_file}")
                with open(report_file, 'r') as f:
                    report = json.load(f)
                # keep only required entries and add the solution
                report = {k: v for k, v in report.items() if k in REQUIRED_ENTRIES}
                report["solution"] = solution
                results[node_id] = report
                # Don't delete the report file, it might be needed later
            else:
                logging.error(f"Report file not found at {report_file}")
                # Check other possible locations
                possible_locations = [
                    Path(f"{run_id}.json"),
                    Path(f"{run_id}_1.json"),
                    Path(f"/tmp/{run_id}.json")
                ]
                for loc in possible_locations:
                    if loc.exists():
                        logging.info(f"Report file found at alternative location: {loc}")
                        with open(loc, 'r') as f:
                            report = json.load(f)
                        # keep only required entries and add the solution
                        report = {k: v for k, v in report.items() if k in REQUIRED_ENTRIES}
                        report["solution"] = solution
                        results[node_id] = report
                        break
                else:
                    results[node_id] = {"error": "Report file not found", "solution": solution}

        except Exception as e:
            logging.exception(f"Error in evaluation for node {node_id}: {str(e)}")
            results[node_id] = {"error": str(e), "traceback": traceback.format_exc(), "solution": solution}
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    # Write final results
    final_report_file = os.path.join(os.path.dirname(solutions_file), f"solutions_report.json")
    with open(final_report_file, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"Final report written to {final_report_file}")

    # Clean up images
    existing_images = list_images(client)
    clean_images(client, existing_images, cache_level, clean)

    return results

def get_unique_solutions(trajectory: Trajectory):
    unique_diffs = set()
    for transition in trajectory.transitions:
        if transition.state.name in ["Finished"]:
            if transition.state.output and transition.state.output.get("diff"):
                unique_diffs.add(transition.state.output["diff"])

    # Create a new dictionary with unique solutions
    unique_solutions = {i: diff for i, diff in enumerate(unique_diffs)}
    print(f"Found {len(unique_solutions)} unique solutions")
    return unique_solutions

def main(solution_dir, instance_ids=None):
    if isinstance(instance_ids, str):
        instance_ids = [instance_ids]
    TRAJECTORY_FILE_NAME = "trajectory.json"
    trajectory_files = glob.glob(os.path.join(solution_dir, f"*/{TRAJECTORY_FILE_NAME}"))
    print(f"solution_files: {trajectory_files}")

    for trajectory_file_path in trajectory_files:
        trajectory = Trajectory.load(trajectory_file_path)

        instance_id = trajectory.info.get("instance_id")

        print(f"Processing solutions for instance {instance_id}")
        try:
            unique_solutions = get_unique_solutions(trajectory)
            # Create a temporary file with unique solutions
            temp_file_path = f"{trajectory_file_path}_unique.json"
            with open(temp_file_path, 'w') as f:
                json.dump(unique_solutions, f, indent=2)
            print(f"Running evaluation for instance {instance_id} with {len(unique_solutions)} unique solutions")
            run_swebench_multi_solutions(
                dataset_name="princeton-nlp/SWE-bench_Lite",
                split="test",
                instance_id=instance_id,
                solutions_file=temp_file_path,
                max_workers=4,
                force_rebuild=False,
                cache_level="env",
                clean=False,
                open_file_limit=4096,
                timeout=1800
            )
        except Exception as e:
            print(f"Error processing {trajectory_file_path}: {str(e)}")
        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals():
                os.remove(temp_file_path)
    print("Evaluation completed for all instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWEBench evaluation with optional solution directory.")
    parser.add_argument("--solution_dir", type=str, default="/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/search_and_code/20240906_search_and_code_gpt-4o-mini-2024-07-18_max_exp2_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_eval_name_None",
                        help="Directory containing solution files (default: %(default)s)")
    parser.add_argument("--instance_ids", type=str, nargs='+', default=None,
                        help="Instance IDs to evaluate (can specify multiple, space-separated)")
    args = parser.parse_args()

    main(args.solution_dir)