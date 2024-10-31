import os
import json
from argparse import ArgumentParser, Action
from moatless.utils_search.misc import deep_get
from moatless.search.reward import LLM_Value_Function
from tqdm import tqdm
from datetime import datetime


class ParseDict(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for i in range(0, len(values), 2):
            if i+1 < len(values):
                d[values[i]] = values[i+1]
        setattr(namespace, self.dest, d)
        
        
def find_json_files(path, task_type):
    json_files = []
    print(f"Searching for JSON files in: {path}")
    print(f"Task type: {task_type}")
    
    if os.path.isfile(path):
        print(f"Input path is a file: {path}")
        return [path] if path.endswith('.json') else []
    elif os.path.isdir(path):
        print(f"Input path is a directory: {path}")
        
        for root, dirs, files in os.walk(path):
            if os.path.basename(root) == task_type:
                for file in files:
                    if file.endswith('.json'):
                        full_path = os.path.join(root, file)
                        print(f"Found JSON file: {full_path}")
                        json_files.append(full_path)
                # Stop descending into subdirectories
                dirs[:] = []
    
    print(f"Total JSON files found: {len(json_files)}")
    return json_files

def load_message(path):
    with open(path, "r") as f:
        return json.load(f)

def set_environment_variables(keys):
    for name, value in keys.items():
        os.environ[name] = value
        
def create_run_id(tag, value_args):
    if tag:
        run_id = tag
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add value_args to the run_id
    if value_args:
        args_str = '_'.join(f"{k}-{v}" for k, v in value_args.items())
        run_id = f"{run_id}_{args_str}"
    
    return run_id

def eval_trajectory(value_function, trajectory, 
                    history=False, val_args={}):
    print(f"trajectory_keys: {trajectory.keys()}")
    init_msg_key = "initial_message" if "initial_message" in trajectory else "input.problem_statement"
    problem_statement = deep_get(trajectory, init_msg_key)
    steps = trajectory["transitions"] if "transitions" in trajectory else trajectory["steps"]
    rewards = []
    state_history = []

    for n, step in enumerate(steps):
        if "actions" not in step:
            continue
        if len(step["actions"]) == 0:
            continue
        actions = step["actions"][-1]
        state_info = step["name"] + "\n\n" + str(step.get("state", ""))
        state_message = str(actions["action"]) if "action" in actions else str(actions["input"])
        state_response = str(actions.get("output", None))

        current_state = {
            "state_info": state_info,
            "state_message": state_message,
            "state_response": state_response,
            "step_count": n,
            "node_id": n
        }
        if val_args.get("state_file_context", None):
            current_state["state_file_context"] = step["file_context"]
        
        merged_args = {**current_state, **val_args}
            
        if history:
            reward = value_function.get_reward(
                problem_statement=problem_statement,
                state_history=state_history,
                **merged_args,
            )
            state_history.append(current_state)
        else:
            reward = value_function.get_reward(
                problem_statement=problem_statement,
                **merged_args,
            )

        rewards.append(reward)

    return rewards

def evaluate_tree(value_function, messages, **kwargs):
    tree_rewards = {}
    for node_data in messages:
        print(f"node_data: {node_data}")
        # Directly pass the messages to get_reward
        node_id = node_data["node_id"]
        reward = value_function.get_reward(
            message=node_data["input"]["messages"], 
            node_id=node_id,
            **kwargs
            )
        tree_rewards[node_id] = reward
    
    return tree_rewards

def process_file(task, model, file_path, run_id=None, **kwargs):
    message = load_message(file_path)
    filename = os.path.basename(file_path)
    val_args = kwargs.get("val_args", {})
    
    # Create a new subdirectory with the run_id
    replace_tag = "trajs" if "trajs" in file_path else "rews"
    if run_id:
        save_dir = os.path.dirname(file_path).replace(replace_tag, os.path.join(task, run_id))
    else:
        save_dir = os.path.dirname(file_path).replace(replace_tag, task)
        
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    value_function = LLM_Value_Function(model=model, filename=save_path)
    print(f"save_path: {save_path}, run_id: {run_id}")
    
    if task in ["rews_traj"]:
        out = eval_trajectory(value_function, message,
                              history=(task == "rews_traj"),
                              **kwargs)
    elif task == "rews":
        out = evaluate_tree(value_function, message, 
                            **val_args)
    
    print(f"Results for {save_path} with model {model}")
    return out

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to input file, directory, or base directory", required=True)
    parser.add_argument("--keys_dir", type=str, help="Path to the keys.json file", default="../../keys.json")
    parser.add_argument("--task", type=str, help="Task to run", required=True, choices=["rews", "tree_eval", "rews_traj"])
    parser.add_argument("--instances", nargs='+', help="Specify one or more instances", default=None)
    parser.add_argument("--models", nargs='+', help="Specify one or more models", required=True)
    parser.add_argument("--tag", type=str, help="Optional tag for the run", default="")
    parser.add_argument('--val_args', nargs='*', action=ParseDict, default={},
                        help='Additional arguments for value_function in the format: key1 value1 key2 value2 ...')
    args = parser.parse_args()
    
    print(f"Input path: {args.input_path}")
    print(f"Task: {args.task}")
    print(f"Models: {args.models}")
    print(f"Instances: {args.instances}")
    
    
    with open(args.keys_dir) as f:
        keys = json.load(f)
    set_environment_variables(keys)

    run_id = create_run_id(args.tag, args.val_args)

    json_files = find_json_files(args.input_path, args.task)

    if not json_files:
        print("No JSON files found. Please check the input path and task type.")
        exit(1)

    for model in args.models:
        pbar = tqdm(json_files, desc=f"Processing files for model {model}")
        for file_path in pbar:
            if args.instances and not any(instance in file_path for instance in args.instances):
                continue
            try:
                results = process_file(args.task, model, file_path, 
                                run_id=run_id, 
                                val_args=args.val_args)
                print(f"Results for {file_path} with model {model}:")
                print(json.dumps(results, indent=2))
            except Exception as e:
                print(f"Error processing {file_path}")
                print(f"error: {e}")
                continue