


def test_edit_code():
    global_params = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.8,
        "verify": False
    }
    loop = AgenticLoop.from_trajectory_file("/home/albert/repos/albert/sw-planner-2/trajs/evaluations/20240915_django_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_eval_name_None/django__django-11039-2/trajectory.json")
    loop._transition_rules = code_transitions(global_params=global_params)
    loop.revert_to_state(state_id=6)

    state = loop.clone_current_state()
    state.model = "gpt-4o-mini-2024-07-18"
    new_state = loop._execute_state_until_transition_to(["PlanToCode"])
    print(new_state)

    loop.trajectory.persist("/home/albert/repos/albert/sw-planner-2/trajs/evaluations/20240915_django_gpt-4o-mini-2024-07-18_max_exp3_mcts_True_debate_False_provide_feedback_True_temp_bias_0.0_eval_name_None/django__django-11039-2/trajectory.json")

