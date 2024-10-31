INSTANCES_SUCCESS="pytest-dev__pytest-5227 django__django-16139 sympy__sympy-24152 django__django-16379 django__django-16527 django__django-13933"
INSTANCES_FAIL="django__django-11019 django__django-11630 django__django-12856 sympy__sympy-18199 sympy__sympy-19487 sympy__sympy-18835 scikit-learn__scikit-learn-13142 scikit-learn__scikit-learn-13241 matplotlib__matplotlib-24970 pydata__xarray-3364 sphinx-doc__sphinx-8506"
# ALL_INSTANCES="$INSTANCES_SUCCESS $INSTANCES_FAIL"
ALL_INSTANCES="pytest-dev__pytest-5227"

# Check if a model argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a model name as an argument."
    echo "Usage: $0 <model_name>"
    exit 1
fi

# Store the model name from the command-line argument
MODEL="$1"

# python run_llm.py \
#  --message "/share/edc/home/antonis/moatless-tools/evaluations/20240617_moatless_gpt-4o-eval_traj/trajs" \
#  --task rews_traj \
#  --instances $ALL_INSTANCES \
#  --tag prompt_2 \
#  --val_args state_file_context True debate True \
#  --models meta-llama/Meta-Llama-3.1-70B-Instruct


# DEBUG RUN
# /share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/20240617_moatless_gpt-4o_debug/rews/gpt-4o-mini/pytest-dev__pytest-5227.json
# /share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/20240617_moatless_gpt-4o_reward_2_0/rews/gpt-4o-mini \
# run evaluate_tree function
python3 run_llm.py \
    --task rews \
    --input_path /share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/search_and_code/20240810_search_and_code_gpt-4o-2024-08-06 \
    --models $MODEL \
    --val_args debate True \
    --tag eval_rew_fn


## Default run
#  --message "/share/edc/home/antonis/moatless-tools/evaluations/20240617_moatless_gpt-4o-2024-05-13/trajs" \
# "/share/edc/home/antonis/moatless-tools/evaluations/20240617_moatless_gpt-4o-eval_traj/trajs"

#  --instances $INSTANCES_FAIL \
#  Qwen/Qwen2-7B-Instruct 