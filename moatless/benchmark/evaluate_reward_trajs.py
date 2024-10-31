import os
import json
import logging
import traceback
import collections
from tqdm import tqdm
import argparse

from moatless.search.reward import LLM_Value_Function
from moatless.trajectory import Trajectory
from moatless.mcts import MCTSNode

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

OMMITED_STATES = ["Pending"]

# New helper functions
def create_id_mappings(nodes):
    node_to_state = {}
    state_to_node = {}
    for node_id, node_data in nodes.items():
        node_id, state_id = int(node_id), int(node_data['state_id'])
        node_to_state[node_id] = state_id
        state_to_node[state_id] = node_id
    return node_to_state, state_to_node

def node_id_to_state_id(node_id, node_to_state):
    return node_to_state.get(node_id)

def state_id_to_node_id(state_id, state_to_node):
    return state_to_node.get(state_id)


def evaluate(trajectory: Trajectory, value_function,
             ids=None, node_to_state=None, state_to_node=None):
    print(f"ids: {ids}")
    value_results = collections.defaultdict(dict)
    for transition in trajectory.transitions:
        if transition.state.name in OMMITED_STATES:
            # print(f"Skipping state: {transition.state.name}")
            continue
        else:
            # print(f"Evaluating state: {transition.state.id} - {transition.state.name}")
            pass
        if ids and transition.state.id not in ids:
            continue
        if transition.state is None:
            continue
        
        resolved = False
        diff = ""
        
        if hasattr(transition.state, "output") and transition.state.output is not None:
            evaluation_result = transition.state.output.get("evaluation_result")
            if evaluation_result:
                resolved = evaluation_result.get("resolved", False)
            diff = transition.state.output.get("diff", "")

        trajectory.restore_from_snapshot(transition)
        if transition.state.previous_state is None:
            continue
        elif transition.state.previous_state.name == "Pending":
            # print(f"Skipping state {transition.state.name} with previous state {transition.state.previous_state.name}")
            continue
        else:
            # print(f"Evaluating state {transition.state.name} with previous state: {transition.state.previous_state.name}")
            pass

        # transition_node_id = transition.state.id
        transition_node_id = state_id_to_node_id(transition.state.id, state_to_node)
        try:
            # Create MCTSNode instances for the current and previous states
            parent_node = MCTSNode(0, state=transition.state.previous_state)
            current_node = MCTSNode(1, parent=parent_node, state=transition.state)
            
            reward, explanation = value_function.get_reward(
                node=current_node,
                debate=DEBATE,
            )
        except:
            reward, explanation = None, None
            traceback.print_exc()
            continue
        value_results[transition_node_id] = {
            "reward": reward,
            "explanation": explanation,
            "resolved": resolved,
            "state_type": transition.state.name
        }
        print(f"Diff:\n{diff}")
        print(f"""\n\nReward: {reward}
                Explanation: {explanation}""")
    
    print(f"Value results: {value_results}")
    
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
    DEBATE = False
    MODEL = "gpt-4o-mini"
    RUNS = 20
    REPLACE = True
    TEMPERATURE = 0.2

    INSTANCES = [
        "django__django-11039",
        "django__django-12983",
        "django__django-12453",
        "django__django-11583",
        "django__django-15851",
    ]

    INSTANCES = None
    
    BASE_PATH = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/search_and_code/20240910_mcts_plan_deepseek"

    value_function = LLM_Value_Function(log_to_file=False,
                                        temperature=TEMPERATURE)

    if INSTANCES is None:
        INSTANCES = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]

    for instance in INSTANCES:
        print(f"\nProcessing instance: {instance}")
        instance_path = os.path.join(BASE_PATH, instance)
        
        trajectory = Trajectory.load(os.path.join(instance_path, "trajectory.json"))
        traj_filename = f"trajectory_values_debate_{DEBATE}_temp_0.2.json"
        trajectory_values_path = os.path.join(instance_path, "trajectories", traj_filename)
        os.makedirs(os.path.dirname(trajectory_values_path), exist_ok=True)
        
        finished_state_ids = find_finished_node_ids(trajectory)
        
        nodes_path = os.path.join(instance_path, "search", "nodes.json")
        if os.path.exists(nodes_path):
            nodes = json.load(open(nodes_path))
            state_trajectories = build_state_trajectories(nodes, finished_state_ids)
            node_to_state, state_to_node = create_id_mappings(nodes)
        else:
            state_trajectories = json.load(open(os.path.join(instance_path, "search", "trajectories-json.json")))

        for state_id, traj in state_trajectories.items():
            print(f"{state_id}: {traj}")

        trajectory_runs = {}

        # Load existing data if file exists
        if os.path.exists(trajectory_values_path):
            with open(trajectory_values_path, "r") as f:
                trajectory_runs = json.load(f)
            print(f"Loaded existing trajectory values from {trajectory_values_path}")

        # Continue with remaining runs
        for i in range(len(trajectory_runs), RUNS):
            trajectory_values = {}
            for finished_id, traj_ids in tqdm(state_trajectories.items(), desc=f"Evaluating Trajectories (Run {i+1}/{RUNS})"):
                finished_node_id = state_id_to_node_id(finished_id, state_to_node)
                # finished_node_id = finished_id
                trajectory_values[finished_node_id] = evaluate(trajectory, value_function,
                                                               ids=traj_ids,
                                                               node_to_state=node_to_state,
                                                               state_to_node=state_to_node)
            trajectory_runs[i] = trajectory_values

            # Save the updated trajectory values to the json file
            with open(trajectory_values_path, "w") as f:
                json.dump(trajectory_runs, f, indent=4)
                print(f"Saved trajectory values to {trajectory_values_path}")

        print(f"Completed {len(trajectory_runs)} out of {RUNS} runs for instance {instance}")

    print("Finished processing all instances.")