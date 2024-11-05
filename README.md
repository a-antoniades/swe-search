# SWE-Search and Code Evaluation

This repository contains the development code for the [SWE-Search paper](https://arxiv.org/abs/2403.13657). It is only provided for reproducibility purposes. Use the official [Moatless-Tree-Search repo](https://github.com/aorwall/moatless-tree-search).

<img src="assets/SWE-Search_demo.gif" width="100%">

## Usage

### Basic Command

```bash
python -m moatless.benchmark.run_evaluation \
        --repo_base_dir "/path/to/repos" \
        --eval_dir "/path/to/evaluations" \
        --eval_name "my_evaluation" \
        --model "model-name" \
        --max_transitions 100 \
        --temp 0.2 \
        --no_testbed \
        --num_workers 1
```

In order to use the testbed, please refer to official Moatless-Tree-Search repo.

### Key Parameters

- `repo_base_dir`: Base directory containing repository clones
- `eval_dir`: Directory for storing evaluation results
- `eval_name`: Name of the evaluation run
- `model`: LLM model to use (e.g., "gpt-4", "claude-2")
- `max_transitions`: Maximum number of state transitions allowed
- `temp`: Temperature for model sampling
- `num_workers`: Number of parallel workers
- `resolved_by`: Number of attempts before considering an instance resolved
- `mcts`: Enable Monte Carlo Tree Search (flag)
- `no_testbed`: Disable testbed usage (flag)

### Advanced Settings

The evaluation function (`evaluate_search_and_code`) supports additional configuration through `TreeSearchSettings`:

```python
python
tree_search_settings = TreeSearchSettings(
    max_expansions=3,
    max_iterations=100,
    max_transitions=100,
    max_finished_transitions=5,
    states_to_explore=["SearchCode", "PlanToCode"],
    provide_feedback=False,
    debate=False,
    best_first=True,
    value_function_model="model-name",
    value_function_model_temperature=0.0
)
```


### Environment Variables

Required environment variables:
- `MOATLESS_DIR`: Base directory for evaluations
- `CUSTOM_LLM_API_BASE`: (Optional) Custom LLM API base URL
- `CUSTOM_LLM_API_KEY`: (Optional) Custom LLM API key

## Output

The evaluation generates:
- Detailed logs of each run
- Error logs for debugging
- Evaluation results in the specified evaluation directory
- (Optional) SWE-bench evaluation results when using `--no_testbed`

