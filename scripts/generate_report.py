import logging
import logging
import os
from typing import Tuple
import json
from tabulate import tabulate
from colorama import Fore, Back, Style, init
import shutil

from moatless.benchmark.report_v2 import to_result, to_dataframe, to_trajectory_dataframe, read_reports
from moatless.benchmark.utils import get_moatless_instances
from moatless.trajectory import Trajectory

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.WARNING,
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
                # Check if file is empty
                if os.stat(trajectory_path).st_size == 0:
                    logger.warning(f"Empty trajectory file: {trajectory_path}")
                    continue

                trajectory = Trajectory.load(trajectory_path, skip_workspace=True)
                trajectories.append((trajectory, rel_path))

                trajectory_path = os.path.join(root, "trajectory_light.json")
                trajectory.persist(trajectory_path, exclude={"completion", "completions"})
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

        result = to_result(instance, trajectory)
        if result.duplicated_search_actions > 0:
            duplicted_sarch += 1
        results.append(result)

    print(f"Results: {len(results)}, duplicated search actions: {duplicted_sarch}")

    # Save the results as report.json
    report_path = os.path.join(dir, "report.json")
    with open(report_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)

    # Use the saved results directly
    df = to_dataframe(results, report_mode="mcts")

    # Add a new column for success status
    df['success_status'] = df.apply(lambda row: 
        'Resolved' if row['status'] == 'resolved' else
        'Running w/ Resolved' if row['status'] == 'running' and row['resolved_solutions'] > 0 else
        'Partially Resolved' if row['resolved_solutions'] > 0 else
        'Running' if row['status'] == 'running' else
        'Rejected' if row['status'] == 'rejected' else
        'Error' if row['status'] == 'error' else
        'Failed', axis=1)

    # Format llmonkeys_rate as a percentage
    df['llmonkeys_rate'] = df['llmonkeys_rate'].apply(lambda x: f"{x * 100:.1f}%")

    # Select columns to display
    display_columns = ['instance_id', 'resolved_by', 'llmonkeys_rate', 'success_status', 'resolved_solutions', 'failed_solutions', 'resolved_max_reward', 'failed_max_reward', 'all_transitions', 'duration', 'total_cost']

    # Prepare data for tabulate
    table_data = df[display_columns].values.tolist()
    headers = display_columns

    # Initialize colorama
    init()

    # Define color mapping for success_status
    color_map = {
        'Resolved': Fore.GREEN,
        'Running w/ Resolved': Fore.CYAN,
        'Partially Resolved': Fore.YELLOW,
        'Running': Fore.BLUE,
        'Rejected': Fore.MAGENTA,
        'Failed': Fore.RED
    }

    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Generate the table
    table = tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[None, None, None, None, None, None, 8, 8, None, None, None])
    
    # Apply colors to the table after generation
    colored_lines = []
    for line in table.split('\n'):
        for status, color in color_map.items():
            if status in line:
                line = line.replace(status, f"{color}{status}{Style.RESET_ALL}")
        colored_lines.append(line)

    # Print the table, truncating if necessary
    for line in colored_lines:
        if len(line) > terminal_width:
            print(line[:terminal_width - 3] + '...')
        else:
            print(line)

    # Calculate summary statistics
    total_trajectories = len(df)
    status_counts = df['success_status'].value_counts()
    total_cost = df['total_cost'].sum()
    total_prompt_tokens = df['prompt_tokens'].sum()
    total_completion_tokens = df['completion_tokens'].sum()
    total_transitions = df['all_transitions'].sum()
    avg_cost = df['total_cost'].mean()
    filtered_df = df[~df['status'].isin(['running', 'error', 'rejected'])]
    avg_duration = filtered_df['duration'].mean()

    # Print summary statistics
    print("\nSummary:")
    print(f"Total Trajectories: {total_trajectories}")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Total Transitions: {total_transitions}")
    print(f"Average Cost per Trajectory: ${avg_cost:.2f}")
    print(f"Average Duration: {avg_duration:.2f} s")
    print("\nTrajectory Status Distribution:")
    for status, count in status_counts.items():
        color = color_map.get(status, '')
        print(f"{color}{status}: {count}{Style.RESET_ALL}")

    # to csv
    df = to_trajectory_dataframe(results)
    df.to_csv(os.path.join(dir, "trajectories.csv"), index=False)

    # to json
    with open(os.path.join(dir, "report.json"), "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)


#    df = to_dataframe(results)
#    print(df)
# directory = "/home/albert/repos/albert/swe-planner/trajs/evaluations/20240906_moatless_claude-3.5-sonnet"
# directory = "/home/albert/repos/albert/sw-planner-2/trajs/evaluations/vf_1_3_U_3-r122/20240920_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_use_testbed_True"
# generate_report(directory)

if __name__ == "__main__":
    import sys
    directory = sys.argv[1]
    generate_report(directory)
