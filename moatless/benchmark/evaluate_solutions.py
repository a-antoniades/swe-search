import os
import json
import logging
import traceback
import collections
from tqdm import tqdm
import argparse
import csv
from moatless.search.reward import LLM_Value_Function
from moatless.trajectory import Trajectory

from concurrent.futures import ThreadPoolExecutor, as_completed

"""
"gpt-4o-2024-08-06"
"gpt-4o-mini-2024-07-18"
"""

def evaluate_solution(instance, base_path, model, debate=False):
    try:
        print(f"------ Instance: {instance} ------")
        instance_path = os.path.join(base_path, instance)
        solutions = json.load(open(os.path.join(instance_path, "search", "solutions.json")))
        nodes = json.load(open(os.path.join(instance_path, "search", "nodes.json")))
        trajectory = json.load(open(os.path.join(instance_path, "trajectory.json")))
        problem_statement = trajectory["initial_message"]

        llm_value_function = LLM_Value_Function(model=model, 
                                                temperature=1.0,
                                                base_dir=base_path)
        id, explanation = llm_value_function.compare_solutions(solutions, problem_statement, debate=debate)

        solution = solutions[str(id)]
        
        print(f"--Preferred solution: {id}--")
        print(f"Explanation:\n {explanation}")
        
        resolved_status = nodes[str(id)]['resolved']
        print(f"SOLUTION RESOLVED_STATUS: {resolved_status}")
        
        return instance, solution, 'resolved' if resolved_status else 'not_resolved'
    except Exception as e:
        print(f"Error evaluating instance {instance}: {e}")
        traceback.print_exc()
        return instance, str(e), 'error'


partially_resolved_instances_4o_mini = [
    "sympy__sympy-24213",
    "django__django-16046",
    "matplotlib__matplotlib-23314",
    "sympy__sympy-12481",
    "django__django-13028",
    "django__django-13448"
]

partiall_resolved_instances_qwen2_5 = [
    "django__django-11964",
    "django__django-12983",
    "sympy__sympy-15678",
    "sympy__sympy-18057",
    "scikit-learn__scikit-learn-12471",
    "psf__requests-3362",
    "pydata__xarray-5131",
    "pylint-dev__pylint-5859",
    "mwaskom__seaborn-3190",
    "scikit-learn__scikit-learn-25500",
    "django__django-11848",
    "scikit-learn__scikit-learn-15535",
    "scikit-learn__scikit-learn-13241",
    "django__django-14382",
    "django__django-11620"
]

partially_resolved_instances_4o = [
    "scikit-learn__scikit-learn-14894",
    "django__django-11583",
    "django__django-12700",
    "scikit-learn__scikit-learn-15535",
    "matplotlib__matplotlib-24970",
    "django__django-15498",
    "sympy__sympy-12481",
    "matplotlib__matplotlib-26020",
    "django__django-12708",
    "sympy__sympy-15609",
    "django__django-15902",
    "matplotlib__matplotlib-23562",
    "sympy__sympy-17022",
    "scikit-learn__scikit-learn-15512",
    "django__django-13925",
    "sympy__sympy-18189",
    "scikit-learn__scikit-learn-14092",
    "django__django-13265"
]

not_resolved_4o_debate = [
    "django__django-15498",
    "matplotlib__matplotlib-26020",
    "django__django-12708",
    "django__django-12497",
    "sympy__sympy-18189"
]

# instance_new = ["django__django-12497"]

if __name__ == "__main__":
    # MODEL = "gpt-4o-2024-08-06"
    MODEL = "gpt-4o-mini-2024-07-18"
    DEBATE = True
    # base_path = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/qwen_mcts"
    BASE_PATH = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/evaluations/vf_1_3_U_3-r122/20240920_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_use_testbed_True"
    # BASE_PATH = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/evaluations/vf_1_3_U_3-r122/20240922_gpt-4o-2024-08-06_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_use_testbed_True"
    
    INSTANCES = None

    if INSTANCES is None:
        INSTANCES = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
    
    INSTANCES = [instance for instance in INSTANCES if instance not in partially_resolved_instances_4o]
    # INSTANCES = INSTANCES[:4]
    input(f"will process {len(INSTANCES)} instances... press enter to continue")

    report_path = os.path.join(BASE_PATH, "result.csv")
    predictions_path = os.path.join(BASE_PATH, "discriminator_predictions.jsonl")
    predictions = {}

    resolved = 0
    # Read the existing CSV file
    with open(report_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        fieldnames = reader.fieldnames + ['discriminator_status']

    # Process instances in parallel
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_instance = {executor.submit(evaluate_solution, instance, BASE_PATH, MODEL, DEBATE): instance for instance in INSTANCES}

        for future in tqdm(as_completed(future_to_instance), total=len(INSTANCES), desc="Processing instances"):
            instance, solution, status = future.result()

            prediction = {
                "model_name_or_path": MODEL,
                "instance_id": instance,
                "model_patch": solution
            }
            predictions[instance] = prediction
            if status == 'resolved':
                resolved += 1

    # Write all predictions back to the file
    with open(predictions_path, "w") as file:
        for prediction in predictions.values():
            json_string = json.dumps(prediction)
            file.write(json_string + "\n")

    print(f"Resolved: {resolved}/{len(INSTANCES)}")