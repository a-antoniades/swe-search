import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import shutil
import traceback
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm
from tabulate import tabulate
from moatless.benchmark.report_v2 import read_results_from_json
from moatless.search.reward import LLM_Value_Function
from moatless.trajectory import Trajectory


"""
"gpt-4o-mini-2024-07-18"
"gpt-4o-2024-08-06"
"""

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

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

partially_resolved_instances_4o_mini = [
    "sympy__sympy-24213",
    "django__django-16046",
    "matplotlib__matplotlib-23314",
    "sympy__sympy-12481",
    "django__django-13028",
    "django__django-13448"
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run discriminator script")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory path")
    return parser.parse_args()

args = parse_arguments()

value_function = LLM_Value_Function(model=args.model, temperature=1.0)

# original_predictions_file = f"{args.base_dir}/all_preds.jsonl"

predictions_file = f"{args.base_dir}/all_preds_discriminator.jsonl"

# Create a new copy of the predictions file for discriminator results
# shutil.copy(original_predictions_file, predictions_file)

results = read_results_from_json(f"{args.base_dir}/report.json")

predictions = {}
if os.path.exists(predictions_file):
    with open(predictions_file, "r") as f:
        for line in f:
            prediction = json.loads(line)
            predictions[prediction["instance_id"]] = prediction

existing_predictions = {}
# if os.path.exists(discriminator_predictions_file):
#     with open(discriminator_predictions_file, "r") as f:
#         for line in f:
#             prediction = json.loads(line)
#             existing_predictions[prediction["instance_id"]] = prediction

def process_instance(instance_id):
    try:
        print(f"------ Instance: {instance_id} ------")
        instance_path = os.path.join(args.base_dir, instance_id)
        trajectory = Trajectory.load(os.path.join(instance_path, "trajectory.json"), skip_workspace=True)

        transition = value_function.compare_solutions2(trajectory, include_history=False, show_reward=False, debate=True)

        if not transition.state.output:
            patch = ""
            resolved_status = False
            status_details = "No transition or output found"

        else:
            try:
                patch = transition.state.output.get("diff", "") + "\n"
                evaluation_result = transition.state.output.get("evaluation_result", {})
                resolved_status = evaluation_result.get("resolved", False)
                tests_status = evaluation_result.get("tests_status", {})
                status_details = f"Status: {tests_status.get('status', 'Unknown')}"
                if tests_status.get("fail_to_pass"):
                    status_details += f", Fail to Pass: {tests_status['fail_to_pass']}"
            except Exception as e:
                patch = ""
                resolved_status = False
                status_details = str(e)
        
        print(f"SOLUTION RESOLVED_STATUS: {resolved_status}")
        print(f"Status Details: {status_details}")
        
        return instance_id, patch, 'resolved' if resolved_status else 'not_resolved', status_details
    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        traceback.print_exc()
        return instance_id, None, 'error', str(e)


def summarize_instance(result):
    return [
        result.instance_id,
        result.resolved_solutions,
        result.solutions,
        result.resolved_by
    ]

resolved_instances = [
    "astropy__astropy-14995",
    "django__django-10914",
    "django__django-11039",
    "django__django-11099",
    "django__django-11133",
    "django__django-11179",
    "django__django-11815",
    "django__django-11999",
    "django__django-12286",
    "django__django-12983",
    "django__django-13590",
    "django__django-13658",
    "django__django-13710",
    "django__django-13757",
    "django__django-14016",
    "django__django-14238",
    "django__django-14608",
    "django__django-14672",
    "django__django-14752",
    "django__django-14787",
    "django__django-14855",
    "django__django-14999",
    "django__django-15347",
    "django__django-15498",
    "django__django-15790",
    "django__django-16041",
    "django__django-16139",
    "django__django-16255",
    "django__django-16379",
    "django__django-16527",
    "django__django-16595",
    "matplotlib__matplotlib-23913",
    "matplotlib__matplotlib-23964",
    "matplotlib__matplotlib-24149",
    "matplotlib__matplotlib-24334",
    "matplotlib__matplotlib-26020",
    "mwaskom__seaborn-3010",
    "psf__requests-2317",
    "psf__requests-2674",
    "psf__requests-3362",
    "psf__requests-863",
    "pylint-dev__pylint-7993",
    "pytest-dev__pytest-11143",
    "pytest-dev__pytest-7373",
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-13241",
    "scikit-learn__scikit-learn-13439",
    "scikit-learn__scikit-learn-13496",
    "scikit-learn__scikit-learn-13779",
    "sphinx-doc__sphinx-8595",
    "sympy__sympy-12481",
    "sympy__sympy-13471",
    "sympy__sympy-14774",
    "sympy__sympy-15011",
    "sympy__sympy-15678",
    "sympy__sympy-18057",
    "sympy__sympy-18189",
    "sympy__sympy-20154",
    "sympy__sympy-21614",
    "sympy__sympy-22005",
    "sympy__sympy-24152",
    "sympy__sympy-24213",
    "sphinx-doc__sphinx-8713",
    "pytest-dev__pytest-5692",
    "django__django-15789",
    "django__django-16873",
]

instance_ids = []
instances_to_process = []
for result in results:
    if result.instance_id.startswith("sphinx"):
        print(f"sphinx instance: {result.instance_id}")
    if result.instance_id not in predictions and (result.resolved_solutions > 0 or result.instance_id in resolved_instances):
        instance_ids.append(result.instance_id)
        instances_to_process.append(summarize_instance(result))


print("Instances to be processed:")
table_headers = ["Instance ID", "Resolved Solutions", "Solutions", "Resolved By"]
print(tabulate(instances_to_process, headers=table_headers, tablefmt="grid"))

print(f"Total instances to process: {len(instance_ids)}, instances already processed: {len(predictions)}")

input("Press Enter to start processing...")
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
if os.path.exists(predictions_file):
    shutil.copy(predictions_file, f"{predictions_file}:{date_str}.backup")

with ThreadPoolExecutor(max_workers=50) as executor:
    future_to_instance = {executor.submit(process_instance, instance_id): instance_id for instance_id in instance_ids}
    
    count = 0
    resolved_count = 0
    no_patch_count = 0
    error_count = 0
    new_predictions = {}
    for future in tqdm(as_completed(future_to_instance), total=len(instance_ids), desc="Processing instances"):
        instance_id, patch, status, _ = future.result()
        if patch:
            predictions[instance_id] = patch

            prediction = {
                "model_name_or_path": args.model,
                "instance_id": instance_id,
                "model_patch": patch
            }

            with open(predictions_file, "a") as file:
                json_string = json.dumps(prediction)
                file.write(json_string + "\n")

    print(f"Patches found for {len(instance_ids) - no_patch_count} instances out of {len(instance_ids)}")
    print(f"Processed {count} instances")
    print(f"No patch found for {no_patch_count} instances")
    print(f"Results saved to: {predictions_file}")
