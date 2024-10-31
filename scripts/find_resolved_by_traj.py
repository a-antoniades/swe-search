import os
import json

path = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/evaluations/vf_1_3_U_3-r122/20240920_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_use_testbed_True/django__django-14999/trajectory.json"
path = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/evaluations/vf_1_3_U_3-r122/20240920_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_use_testbed_True"
instances = os.listdir(path)

for instance in instances:
    trajectory_path = os.path.join(path, instance, "trajectory.json")
    if not os.path.isfile(trajectory_path):
        print(f"Warning: trajectory.json not found for instance {instance}")
        continue

    try:
        with open(trajectory_path) as f:
            trajectory = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in trajectory file for instance {instance}")
        continue

    transitions = trajectory.get("transitions", [])

    for idx, transition in enumerate(transitions):
        properties = transition.get("properties", {})
        output = properties.get("output", {})
        evaluation_results = output.get("evaluation_result", {})
        
        if evaluation_results.get("resolved", False):
            print(f"Resolved at transition {idx}, id: {transition.get('id')}, instance: {instance}")

print("Processing complete.")

