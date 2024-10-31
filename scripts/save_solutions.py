import os
import json

def extract_predictions(base_path, output_file):
    missing_solutions = []
    with open(output_file, 'w') as outfile:
        for issue_name in os.listdir(base_path):
            issue_path = os.path.join(base_path, issue_name)
            if os.path.isdir(issue_path):
                trajectory_path = os.path.join(issue_path, 'trajectory.json')
                if os.path.exists(trajectory_path):
                    with open(trajectory_path, 'r') as trajfile:
                        try:
                            data = json.load(trajfile)
                            if 'info' in data and 'submission' in data['info']:
                                solution = data['info']['submission']
                            else:
                                for step, transition in enumerate(data['transitions']):
                                    if transition['name'] == 'Finished':
                                        if 'snapshot' in transition and 'repository' in transition['snapshot'] and 'patch' in transition['snapshot']['repository']:
                                            solution = transition['snapshot']['repository']['patch']
                                            break
                                        else:
                                            print(f"Missing 'snapshot' or 'repository' or 'patch' key in {trajectory_path}")
                                            missing_solutions.append(issue_name)
                            prediction = {
                                "model_name_or_path": os.path.basename(base_path),
                                "instance_id": issue_name,
                                "model_patch": solution
                            }
                            
                            json.dump(prediction, outfile)
                            outfile.write('\n')
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in {trajectory_path}")
                            missing_solutions.append(issue_name)
                        except KeyError:
                            print(f"Missing 'info' or 'submission' key in {trajectory_path}")
                            missing_solutions.append(issue_name)
                else:
                    missing_solutions.append(issue_name)
    
    return missing_solutions

if __name__ == "__main__":
    # base_path = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/evaluations/vf_1_3_U_3-r122/20240920_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_use_testbed_True"
    base_path = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/evaluations/standard/20240924_gpt-4o-mini-2024-07-18_max_exp3_mcts_False_debate_False_provide_feedback_False_temp_bias_0.0_use_testbed_True"
    # base_path = "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/deepseek_mcts"
    output_file = os.path.join(base_path, "predictions.jsonl")
    
    
    missing = extract_predictions(base_path, output_file)
    print(f"Predictions saved to {output_file}")
    
    if missing:
        print("Missing solutions for the following issues:")
        for issue in missing:
            print(f"- {issue}")
        print(f"n missing: {len(missing)}")
    else:
        print("All solutions were successfully extracted.")