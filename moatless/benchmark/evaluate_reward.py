import os
import json
import logging
import traceback
import collections

from moatless.search.reward import LLM_Value_Function
from moatless.trajectory import Trajectory

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

value_function = LLM_Value_Function(log_to_file=False)


def evaluate(trajectory: Trajectory, ids=None):
    value_results = collections.defaultdict(dict)
    for transition in trajectory.transitions:
        # if transition.state.name in ["PlanToCode"]: # ["Finished"]:
        if ids and transition.state.id not in ids:
            continue
        if hasattr(transition.state, "output"):
            evaluation_result = transition.state.output.get("evaluation_result")
            if evaluation_result:
                resolved = evaluation_result.get("resolved", False)
            else:
                resolved = False

            diff = transition.state.output.get("diff")
        else:
            resolved = False
            diff = ""
        # print(f"\n\nFinished state {transition.state.id}, Resolved: {resolved}")

        trajectory.restore_from_snapshot(transition)
        if transition.state.previous_state is None:
            continue
        reward, explanation = value_function.get_reward(state=transition.state.previous_state)
        # debate_reward, debate_explanation = value_function.get_reward(state=transition.state.previous_state, debate=True)
        value_results[transition.state.id] = {
            "reward": reward,
            "explanation": explanation,
            "resolved": resolved,
        }
        # print(f"Reward: {reward}, Debate Reward: {debate_reward}")
        print(f"Diff:\n{diff}")
        print(f"\n\nExplanation: {explanation}")
            # print(f"\n\nDebate Explanation: {debate_explanation}")
    else:
        print(f"\n\nNo output for transition {transition.state.id}")
    
    return value_results


def build_state_trajectories(nodes: dict, finished_state_ids: list[str]) -> dict[str, list[str]]:
    state_trajectories = {}
    
    def find_node_by_state_id(state_id):
        for node_key, node_data in nodes.items():
            if str(node_data['state_id']) == str(state_id):
                return node_key
        return None

    def find_parent_node(parent_id):
        for node_key, node_data in nodes.items():
            if node_data['id'] == parent_id:
                return node_key
        return None

    for state_id in finished_state_ids:
        trajectory = []
        current_node_key = find_node_by_state_id(state_id)
        
        while current_node_key is not None:
            node = nodes[current_node_key]
            trajectory.append(int(node['state_id']))
            current_node_key = find_parent_node(node['parent_id'])
        
        trajectory.reverse()
        if trajectory:  # Only add to state_trajectories if the trajectory is not empty
            state_trajectories[state_id] = trajectory
    
    return state_trajectories


def find_finished_node_ids(trajectory: Trajectory) -> list[str]:
    finished_node_ids = []
    for transition in trajectory.transitions:
        if transition.state.name == "Finished":
            finished_node_ids.append(transition.state.id)
    return finished_node_ids

if __name__ == "__main__":
    base_path = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/search_and_code/20240912_search_and_code_gpt-4o-mini-2024-07-18_max_exp5_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_eval_name_reasonable_test_dataset_verified/django__django-11179"
    trajectory = Trajectory.load(os.path.join(base_path, "trajectory.json"))
    
    finished_state_ids = find_finished_node_ids(trajectory)
    print(f"Finished state IDs: {finished_state_ids}")
    
    nodes = json.load(open(os.path.join(base_path, "search", "nodes.json")))
    state_trajectories = build_state_trajectories(nodes, finished_state_ids)
    
    print("State trajectories:")
    for state_id, traj in state_trajectories.items():
        print(f"{state_id}: {traj}")

    # iterate over state trajectories and evaluate the reward for each state
    trajectory_values = {}
    for finished_id, traj_ids in state_trajectories.items():
        # create a new trajectory with only the states in traj_ids
        trajectory_values[finished_id] = evaluate(trajectory, ids=traj_ids)
    
    print(f"Trajectory values: {trajectory_values}")