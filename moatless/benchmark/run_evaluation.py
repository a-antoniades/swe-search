import logging
import os
from typing import Optional
from functools import partial
import random
import argparse
from datetime import datetime

from dotenv import load_dotenv

from moatless.benchmark.evaluation import create_evaluation_name, Evaluation

from moatless.benchmark.evaluation import Evaluation
from moatless.edit import EditCode
from moatless.find import SearchCode, IdentifyCode
from moatless.transition_rules import TransitionRule, TreeSearchSettings, AgenticLoopSettings

from moatless.transitions import search_and_code_transitions_v2


reasonable_test_dataset = [
    "django__django-12286",
    "django__django-14915",
    "django__django-16046",
    "django__django-11179",
    "django__django-13710",
    "django__django-15851",
    "django__django-16041",
    "django__django-10914",
    "django__django-12983",
    "django__django-13590",
    "django__django-14999",
    "django__django-15347",
    "django__django-11049",
    "django__django-13933",
    "django__django-14238",
    "django__django-14672",
    "django__django-14787",
    "django__django-11583",
    "django__django-12700",
    "django__django-13447",
    "django__django-13401",
    "django__django-14016",
    "django__django-12497",
    "django__django-11848",
    "sympy__sympy-13647",
    "sympy__sympy-22714",
    "sympy__sympy-18621",
    "sympy__sympy-21847",
    "sympy__sympy-23117",
    "sympy__sympy-20154",
    "sympy__sympy-17022",
    "pytest-dev__pytest-7373",
    "pytest-dev__pytest-5692",
    "pytest-dev__pytest-7432",
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-13439",
    "scikit-learn__scikit-learn-13241",
    "scikit-learn__scikit-learn-13496",
    "scikit-learn__scikit-learn-13779",
    "scikit-learn__scikit-learn-25570",
    "scikit-learn__scikit-learn-13584",
    "scikit-learn__scikit-learn-15535",
    "scikit-learn__scikit-learn-12471",
    "psf__requests-2674",
    "psf__requests-3362",
    "psf__requests-1963",
    "mwaskom__seaborn-3190",
    "sphinx-doc__sphinx-8721",
    "sphinx-doc__sphinx-8595"
]

reasonable_test_dataset_verified = [
    'django__django-14915',
    'django__django-11179',
    'django__django-15851',
    'django__django-10914',
    'django__django-13590',
    'django__django-14999',
    'django__django-13933',
    'django__django-14238',
    'django__django-14672',
    'django__django-14787',
    'django__django-13401',
    'django__django-11848',
    # 'sympy__sympy-13647',
    # 'sympy__sympy-22714',
    # 'sympy__sympy-21847',
    # 'sympy__sympy-20154',
    'pytest-dev__pytest-7432',
    'scikit-learn__scikit-learn-10297',
    'scikit-learn__scikit-learn-13439',
    'scikit-learn__scikit-learn-13496',
    'scikit-learn__scikit-learn-13779',
    'sphinx-doc__sphinx-8721',
    'sphinx-doc__sphinx-8595'
]


reasonable_test_dataset_verified_failed = [
    "django__django-10914",
    "django__django-13401",
    "django__django-13590",
    "django__django-13933",
    "django__django-14672",
    "django__django-15851",
    "pytest-dev__pytest-7432",
    "sphinx-doc__sphinx-8595",
    "sphinx-doc__sphinx-8721",
    "astropy__astropy-12907"
]

verified_failed_gpt_4o_mini = [
    'sphinx-doc__sphinx-8721',
    'sphinx-doc__sphinx-8595',
    'pytest-dev__pytest-7432',
    'sympy__sympy-21847',
    'sympy__sympy-20154',
    'scikit-learn__scikit-learn-10297',
    'scikit-learn__scikit-learn-13779',
    'django__django-13590',
    'django__django-14999',
    'django__django-14238',
    'django__django-14787',
    'django__django-13401',
    'django__django-11848'
]

django_dataset = [
    "django__django-10914",
    "django__django-11039",
    "django__django-11049",
    "django__django-11179",
    "django__django-11583",
    "django__django-12286",
    "django__django-12453",
    "django__django-12497",
    "django__django-12700",
    "django__django-12983",
    "django__django-13230",
    "django__django-13447",
    "django__django-13590",
    "django__django-13710",
    "django__django-13933",
    "django__django-14238",
    "django__django-14608",
    "django__django-14915",
    "django__django-14999",
    "django__django-15347",
    "django__django-15789",
    "django__django-15851",
    "django__django-16041",
    "django__django-16046",
    "django__django-16595",
]

one_easy_per_repo = [
    'django__django-11099',
    'sympy__sympy-24152',
    'mwaskom__seaborn-3010',
    'pydata__xarray-5131',
    #'pytest-dev__pytest-11143',
    'matplotlib__matplotlib-23964',
    'psf__requests-863',
    'scikit-learn__scikit-learn-14894',
    #'sphinx-doc__sphinx-8713',
    'astropy__astropy-14995',
    'pylint-dev__pylint-5859',
]

error_instances = [
    "pytest-dev__pytest-11148",
    "sympy__sympy-17139",
    "sympy__sympy-18057",
    "sympy__sympy-21055",
    "sympy__sympy-21379",
    "sympy__sympy-22840",
    "sympy__sympy-24909"
]

failed_instances = [
    "astropy__astropy-12907", "astropy__astropy-14365", "astropy__astropy-6938",
    "django__django-10924", "django__django-11001", "django__django-11049",
    "django__django-11283", "django__django-11422", "django__django-11583",
    "django__django-11620", "django__django-11742", "django__django-11815",
    "django__django-11848", "django__django-11964", "django__django-11999",
    "django__django-12113", "django__django-12125", "django__django-12184",
    "django__django-12284", "django__django-12308", "django__django-12453",
    "django__django-12700", "django__django-12747", "django__django-12856",
    "django__django-12915", "django__django-13028", "django__django-13158",
    "django__django-13315", "django__django-13321", "django__django-13401",
    "django__django-13447", "django__django-13551", "django__django-13660",
    "django__django-13710", "django__django-13757", "django__django-13768",
    "django__django-13925", "django__django-13964", "django__django-14016",
    "django__django-14017", "django__django-14155", "django__django-14238",
    "django__django-14411", "django__django-14580", "django__django-14608",
    "django__django-14787", "django__django-15061", "django__django-15202",
    "django__django-15213", "django__django-15388", "django__django-15400",
    "django__django-15781", "django__django-15814", "django__django-15902",
    "django__django-16041", "django__django-16046", "django__django-17051",
    "django__django-17087", "matplotlib__matplotlib-22711", "matplotlib__matplotlib-23299",
    "matplotlib__matplotlib-23314", "matplotlib__matplotlib-23562", "matplotlib__matplotlib-23563",
    "matplotlib__matplotlib-23913", "matplotlib__matplotlib-24149", "matplotlib__matplotlib-24970",
    "matplotlib__matplotlib-25311", "matplotlib__matplotlib-25332", "matplotlib__matplotlib-25442",
    "matplotlib__matplotlib-26011", "matplotlib__matplotlib-26020", "mwaskom__seaborn-2848",
    "mwaskom__seaborn-3190", "mwaskom__seaborn-3407", "pallets__flask-4992",
    "psf__requests-3362", "pydata__xarray-5131", "pylint-dev__pylint-6506",
    "pylint-dev__pylint-7080", "pylint-dev__pylint-7114", "pylint-dev__pylint-7993",
    "pytest-dev__pytest-11143", "pytest-dev__pytest-5495", "pytest-dev__pytest-5692",
    "pytest-dev__pytest-7168", "pytest-dev__pytest-7220", "pytest-dev__pytest-7490",
    "pytest-dev__pytest-8365", "scikit-learn__scikit-learn-11281", "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-13241", "scikit-learn__scikit-learn-13497", "scikit-learn__scikit-learn-13584",
    "scikit-learn__scikit-learn-14087", "scikit-learn__scikit-learn-14092", "scikit-learn__scikit-learn-14894",
    "scikit-learn__scikit-learn-14983", "scikit-learn__scikit-learn-25500", "sphinx-doc__sphinx-10325",
    "sphinx-doc__sphinx-7975", "sphinx-doc__sphinx-8435", "sphinx-doc__sphinx-8506",
    "sphinx-doc__sphinx-8595", "sphinx-doc__sphinx-8627", "sphinx-doc__sphinx-8713",
    "sphinx-doc__sphinx-8721", "sympy__sympy-12419", "sympy__sympy-12481",
    "sympy__sympy-13031", "sympy__sympy-13471", "sympy__sympy-13480",
    "sympy__sympy-13971", "sympy__sympy-14396", "sympy__sympy-14817",
    "sympy__sympy-15011", "sympy__sympy-15345", "sympy__sympy-15609",
    "sympy__sympy-15678", "sympy__sympy-16792", "sympy__sympy-16988",
    "sympy__sympy-17022", "sympy__sympy-17655", "sympy__sympy-18532",
    "sympy__sympy-19007", "sympy__sympy-19487", "sympy__sympy-20154",
    "sympy__sympy-20212", "sympy__sympy-20442", "sympy__sympy-20590",
    "sympy__sympy-21614", "sympy__sympy-21847", "sympy__sympy-22005",
    "sympy__sympy-23117", "sympy__sympy-23262", "sympy__sympy-24066",
    "sympy__sympy-24213"
]

all_error_and_failed = error_instances + failed_instances

# sample 10 instances for testing
# all_error_and_failed_small = random.sample(all_error_and_failed, 10)
all_error_and_failed_small = [
    "django__django-11001",
    "django__django-11583",
    "django__django-11815",
    "django__django-11999",
    "django__django-12113",
    "django__django-12125",
    "django__django-12184",
    "django__django-12453",
    "django__django-13028",
    "django__django-13321",
    "django__django-13757",
    "django__django-13768",
    "django__django-13964",
    "django__django-14016",
    "django__django-14017",
    "django__django-15213",
    "django__django-15388",
    "django__django-15400",
    "django__django-15902",
    "matplotlib__matplotlib-22711",
    "matplotlib__matplotlib-23299",
    "matplotlib__matplotlib-23314",
    "matplotlib__matplotlib-23562",
    "matplotlib__matplotlib-23563",
    "matplotlib__matplotlib-24970",
    "matplotlib__matplotlib-25311",
    "matplotlib__matplotlib-25332",
    "matplotlib__matplotlib-26011",
    "mwaskom__seaborn-3190",
    "psf__requests-3362",
    "pydata__xarray-5131",
    "pytest-dev__pytest-7490",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-13241",
    "scikit-learn__scikit-learn-13497",
    "sphinx-doc__sphinx-7975",
    "sphinx-doc__sphinx-8435",
    "sphinx-doc__sphinx-8506",
    "sphinx-doc__sphinx-8627",
    "sympy__sympy-13480",
    "sympy__sympy-14396",
    "sympy__sympy-15678",
    "sympy__sympy-16988",
    "sympy__sympy-19007",
    "sympy__sympy-20154",
    "sympy__sympy-20442",
    "sympy__sympy-20590",
    "sympy__sympy-21055",
    "sympy__sympy-22840",
    "sympy__sympy-24213"
]

all_error_and_failed_remaining = [instance for instance in all_error_and_failed if instance not in all_error_and_failed_small]
all_error_and_failed_remaining_small = random.sample(all_error_and_failed_remaining, 20)

debug_set = ["pytest-dev__pytest-5227"]


def evaluate_search_and_code(
        model: str,
        temperature: float = 0.2,
        identify_model: Optional[str] = None,
        resolved_by: Optional[int] = 5,
        enable_mcts: bool = True,
        instance_ids: Optional[list] = None,
        repo_base_dir: str | None = None,
        use_testbed: bool = True,
        name: str = "search_and_code",
        num_workers: int = 4,
        max_cost: float = 5.0,
        evaluation_name=None,
        evaluations_dir=None,
        date=None,
        tree_search_settings: TreeSearchSettings = None,
        retry_trajectory: bool = False,
        **kwargs,
):
    temperature = temperature or kwargs.get("temp_bias", 0.2)

    if evaluation_name is None:
        evaluation_name = create_evaluation_name(model=model,
                                                date=date,
                                                max_expansions=tree_search_settings.max_expansions,
                                                mcts=enable_mcts,
                                                debate=tree_search_settings.debate,
                                                provide_feedback=tree_search_settings.provide_feedback,
                                                temp_bias=temperature,
                                                use_testbed=use_testbed)

    if not evaluations_dir:
        evaluations_dir = os.getenv("MOATLESS_DIR")
        evaluation_name = os.path.join(name, evaluation_name)

    global_params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 4000,
        "max_prompt_file_tokens": 16000,
    }

    # Expect models with prefix openai/ to be custom
    if model.startswith("openai/"):
        global_params["model_base_url"] = os.getenv("CUSTOM_LLM_API_BASE")
        global_params["model_api_key"] = os.getenv("CUSTOM_LLM_API_KEY")

    state_params = {
        IdentifyCode: {
            "model": identify_model or model,
            "temperature": 0.0
        },
        EditCode: {
            "model": model,
            "temperature": 0.0
        }
    }

    loop_settings = AgenticLoopSettings(
        max_cost=max_cost,
        max_transitions=tree_search_settings.max_transitions
    )

    tree_search_settings.value_function_model = model

    logger.info("Evaluation Parameters:")
    logger.info(f"Evalation dir: {evaluations_dir}")
    logger.info(f"Evaluation Name: {evaluation_name}")
    logger.info(f"Model: {model}")
    logger.info(f"Model Base URL: {global_params.get('model_base_url')}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max Tokens: {global_params['max_tokens']}")
    logger.info(f"Max Prompt File Tokens: {global_params['max_prompt_file_tokens']}")
    logger.info(f"Identify Model: {identify_model or model}")
    logger.info(f"Tree Search Settings:")
    logger.info(f"  Max Expansions: {tree_search_settings.max_expansions}")
    logger.info(f"  Max Iterations: {tree_search_settings.max_iterations}")
    logger.info(f"  Min Finished Transitions: {tree_search_settings.min_finished_transitions_for_early_stop}")
    logger.info(f"  Max Finished Transitions: {tree_search_settings.max_finished_transitions}")
    logger.info(f"  States to Explore: {tree_search_settings.states_to_explore}")
    logger.info(f"  Provide Feedback: {tree_search_settings.provide_feedback}")
    logger.info(f"  Debate: {tree_search_settings.debate}")
    logger.info(f"  Value Function Model: {tree_search_settings.value_function_model}")
    logger.info(f"  Value Function Model Temperature: {tree_search_settings.value_function_model_temperature}")
    logger.info(f"Loop Settings:")
    logger.info(f"  Max Cost: {loop_settings.max_cost}")
    logger.info(f"  Max Transitions: {loop_settings.max_transitions}")
    logger.info(f"Enable MCTS: {enable_mcts}")
    logger.info(f"Number of Workers: {num_workers}")
    logger.info(f"Use Testbed: {use_testbed}")
    logger.info(f"Resolved By: {resolved_by}")
    logger.info(f"Instance IDs: {instance_ids}")

    evaluation = Evaluation(
        transitions=search_and_code_transitions_v2(
            tree_search_settings=tree_search_settings,
            loop_settings=loop_settings,
            global_params=global_params,
            state_params=state_params
        ),
        retry_trajectory=retry_trajectory,
        evaluations_dir=evaluations_dir,
        evaluation_name=evaluation_name,
        repo_base_dir=repo_base_dir,
        max_file_context_tokens=16000,
        enable_mcts=enable_mcts,
        num_workers=num_workers,
        detailed_report=True,
        model=model,
        use_testbed=use_testbed,
        use_local_git_upstream=True,
        enable_vizualizer=False,
        **kwargs,
    )

    evaluation.run_evaluation(
        resolved_by=resolved_by,
        instance_ids=instance_ids,
    )

    return os.path.join(evaluations_dir, evaluation_name)


def run_swebench_evaluation(predictions_path, run_id):
    # Set default values for other parameters
    dataset_name = "princeton-nlp/SWE-bench_Lite"
    split = "test"
    instance_ids = None
    max_workers = 4
    force_rebuild = False
    cache_level = "env"
    clean = False
    open_file_limit = 4096
    timeout = 1800
    from swebench.harness.run_evaluation import main as run_swebench

    # Call the run_evaluation function
    run_swebench(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=predictions_path,
        max_workers=max_workers,
        force_rebuild=force_rebuild,
        cache_level=cache_level,
        clean=clean,
        open_file_limit=open_file_limit,
        run_id=run_id,
        timeout=timeout
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcts", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--no_testbed", action="store_true", help="Disable testbed usage")
    parser.add_argument("--debate", action="store_true")
    parser.add_argument("--max_expansions", type=int, default=3)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--max_transitions", type=int, default=100)
    parser.add_argument("--max_cost", type=float, default=5.0)
    parser.add_argument("--reward_threshold", type=int, default=None)
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--temp_bias", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--name", type=str, default="search_and_code")
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--eval_name", type=str, default=None)
    parser.add_argument("--repo_base_dir", type=str, default=None)
    parser.add_argument("--instance_ids", type=str, nargs="+", default=None)
    parser.add_argument("--retry_trajectory", action="store_true")
    parser.add_argument("--sample_first", action="store_true")
    parser.add_argument("--resolved_by", type=int, default=None)
    args = parser.parse_args()

    # Update logging configuration
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.eval_name is not None:
        log_filename = f"run_evaluation_{args.eval_name}_{current_time}.log"
        error_log_filename = f"run_evaluation_{args.eval_name}_{current_time}_error.log"
    elif args.name is not None:
        log_filename = f"run_evaluation_{args.name}_{current_time}.log"
        error_log_filename = f"run_evaluation_{args.name}_{current_time}_error.log"
    else:
        log_filename = f"run_evaluation_{current_time}.log"
        error_log_filename = f"run_evaluation_{current_time}_error.log"

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Create error file handler which logs error messages
    error_file_handler = logging.FileHandler(error_log_filename)
    error_file_handler.setLevel(logging.ERROR)
    error_file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    error_file_handler.setFormatter(error_file_formatter)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARN)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)

    # Adjust log levels for specific loggers
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("moatless").setLevel(logging.INFO)
    logging.getLogger("moatless.benchmark.evaluation").setLevel(logging.INFO)
    logging.getLogger("moatless.benchmark.run_evaluation").setLevel(logging.INFO)
    # logging.getLogger("mcts_tree").setLevel(logging.INFO)

    load_dotenv()

    tree_search_settings = TreeSearchSettings(
            max_expansions=args.max_expansions,
            max_iterations=args.max_iterations,
            max_transitions=args.max_transitions,
            min_finished_transitions_for_early_stop=None,
            max_finished_transitions=5,
            reward_threshold=args.reward_threshold,
            states_to_explore=["SearchCode", "PlanToCode"],
            provide_feedback=args.feedback,
            debate=args.debate,
            best_first=True,
            value_function_model=args.model,
            value_function_model_temperature=0.0
    )

    if not args.resolved_by:
        if not args.instance_ids:
            # instance_ids = reasonable_test_dataset_verified_failed
            # instance_ids = reasonable_test_dataset_verified
            # instance_ids = ["sphinx-doc__sphinx-8595"]
            # instance_ids = ["astropy__astropy-12907"]
            instance_ids = ["sphinx-doc__sphinx-8721",
                            "django__django-14915",
                            "django__django-14787",
                            # "sympy__sympy-21847",
                            'django__django-14238',
                            'django__django-14672',
                            'django__django-13401',
                            'django__django-11848',
                            "scikit-learn__scikit-learn-13779"]
            # instance_ids = set(reasonable_test_dataset_verified + django_dataset)
            # instance_ids = all_error_and_failed_remaining_small
            instance_ids = None
        else:
            instance_ids = args.instance_ids
    else:
        instance_ids = None
    
    # Create a partial function with preset arguments
    evaluate_preset = partial(
        evaluate_search_and_code,
        retry_trajectory=args.retry_trajectory,
        evaluation_name=args.eval_name,
        evaluations_dir=args.eval_dir,
        name=args.name,
        repo_base_dir=args.repo_base_dir,
        enable_mcts=args.mcts,
        tree_search_settings=tree_search_settings,
        instance_ids=instance_ids,
        date=args.date,
        model=args.model,
        temperature=args.temp,
        max_cost=args.max_cost,
        use_testbed=not args.no_testbed,
        num_workers=args.num_workers,
        best_first=not args.sample_first,
        resolved_by=args.resolved_by
    )

    # First run
    eval_dir = evaluate_preset()

    if args.no_testbed:
        # Run SWEBench evaluation
        run_swebench_evaluation(os.path.join(eval_dir, "all_preds.jsonl"), "result")

        # Rerun evaluate_search_and_code
        eval_dir_rerun = evaluate_preset(retry_trajectory=False)

    try:
        from moatless.benchmark.test_solutions import main as test_solutions

        # run eval for all solutions
        test_solutions(eval_dir, instance_ids=instance_ids)
    except Exception as e:
        logger.info(f"Error running test_solutions: {e}")
