python -m moatless.benchmark.run_evaluation \
    --repo_base_dir "/share/edc/home/antonis/_swe-planner/.repos" \
    --eval_dir "/share/edc/home/antonis/_swe-planner/moatless-tools/evaluations/paper_iclr" \
    --eval_name "20241024_normal_100_iter_gpt-4o-mini" \
    --model "gpt-4o-mini-2024-07-18" \
    --max_transitions 100 \
    --temp 0.2 \
    --resolved_by 1 \
    --num_workers 1