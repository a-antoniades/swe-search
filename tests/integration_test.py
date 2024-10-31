import logging
import os
import tempfile
from datetime import datetime

import pytest
from dotenv import load_dotenv

from moatless import AgenticLoop
from moatless.benchmark.swebench import load_instance, create_workspace
from moatless.benchmark.utils import get_moatless_instance
from moatless.codeblocks import CodeBlockType
from moatless.edit import PlanToCode, plan_v2
from moatless.find import SearchCode
from moatless.repository import FileRepository
from moatless.state import Finished, Pending, Rejected
from moatless.trajectory import Trajectory
from moatless.transition_rules import TransitionRule, TransitionRules
from moatless.transitions import (
    search_and_code_transitions, search_and_code_transitions_v2, code_transitions, )
from moatless.workspace import Workspace

load_dotenv()
moatless_dir = os.getenv("MOATLESS_DIR", "/tmp/moatless")

global_params = {
    "model": "gpt-4o-mini-2024-07-18",  # "azure/gpt-4o",
    "temperature": 0.5,
    "max_tokens": 2000,
    "max_prompt_file_tokens": 8000,
}

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs",
)


@pytest.mark.parametrize("model", [
    "claude-3-5-sonnet-20240620",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "deepseek/deepseek-chat"
])
@pytest.mark.llm_integration
def test_simple_search(model):
    global_params = {
        "model": model,
        "temperature": 0.0
    }

    transitions = TransitionRules(
        global_params=global_params,
        transition_rules=[
            TransitionRule(source=Pending, dest=SearchCode, trigger="init"),
            TransitionRule(source=SearchCode, dest=Finished, trigger="did_search"),
            TransitionRule(source=SearchCode, dest=Finished, trigger="finish"),
        ],
    )

    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)
    loop = AgenticLoop(
        transitions,
        initial_message=instance["problem_statement"],
        workspace=workspace,
        prompt_log_dir=dir,
    )

    response = loop.run()
    assert response.status == "finished"

def test_plan_to_code_deepseek():
    test_plan_to_code("deepseek/deepseek-chat")


@pytest.mark.parametrize("model", [
    #"claude-3-5-sonnet-20240620",
    #"gpt-4o-mini-2024-07-18",
    #"gpt-4o",
    "deepseek/deepseek-chat"
])
@pytest.mark.llm_integration
def test_plan_to_code(model):
    global_params = {
        "model": model,
        "temperature": 0.0
    }

    transitions = TransitionRules(
        global_params=global_params,
        transition_rules=[
            TransitionRule(source=Pending, dest=PlanToCode, trigger="init"),
            TransitionRule(source=PlanToCode, dest=Finished, trigger="edit_code"),
            TransitionRule(source=PlanToCode, dest=Rejected, trigger="finish"),
            TransitionRule(source=PlanToCode, dest=Rejected, trigger="reject"),
        ],
    )

    instance = get_moatless_instance("django__django-16379")
    workspace = create_workspace(instance, use_perfect_file_context=True)
    loop = AgenticLoop(
        transitions,
        initial_message=instance["problem_statement"],
        workspace=workspace,
        prompt_log_dir=dir,
    )

    response = loop.run()
    assert response.status == "finished"


@pytest.mark.parametrize("model", [
    "claude-3-5-sonnet-20240620",
    "gpt-4o-mini-2024-07-18",
    "deepseek/deepseek-chat"
])
def test_create_new_file(model):
    global_params = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.0,
        "verify": False
    }
    with tempfile.TemporaryDirectory() as dir:
        file_repo = FileRepository(repo_path=dir)
        workspace = Workspace(file_repo=file_repo)
        loop = AgenticLoop(
            code_transitions(global_params=global_params),
            initial_message="Create a new file with a function that prints 'hello world'",
            workspace=workspace,
        )
        response = loop.run()
        assert response.status == "finished"

        print(workspace.file_repo.file_paths)
        for file_path in workspace.file_repo.file_paths:
            print(file_path)
            file = workspace.file_repo.get_file(file_path)
            print(file.content)

        assert len(workspace.file_repo.file_paths) == 1
        file = workspace.file_repo.get_file(workspace.file_repo.file_paths[0])
        assert file.content
        assert file.module
        assert file.module.find_block_by_type(CodeBlockType.FUNCTION)


@pytest.mark.llm_integration
def test_plan_to_code_tool_model():
    global_params = {
        "model": "deepseek/deepseek-chat",
        "tool_model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.2
    }

    import moatless.edit.plan_v2 as plan_v2
    transitions = TransitionRules(
        global_params=global_params,
        transition_rules=[
            TransitionRule(source=Pending, dest=plan_v2.PlanToCode, trigger="init"),
            TransitionRule(source=plan_v2.PlanToCode, dest=Finished, trigger="edit_code"),
            TransitionRule(source=plan_v2.PlanToCode, dest=Rejected, trigger="finish"),
            TransitionRule(source=plan_v2.PlanToCode, dest=Rejected, trigger="reject"),
        ],
    )

    instance = get_moatless_instance("django__django-16379")
    workspace = create_workspace(instance, use_perfect_file_context=True)
    loop = AgenticLoop(
        transitions,
        initial_message=instance["problem_statement"],
        workspace=workspace,
        prompt_log_dir=dir,
    )

    response = loop.run()
    assert response.status == "finished"



@pytest.mark.llm_integration
def test_run_search_and_code_transitions_v2():
    global_params = {
        "model": "deepseek/deepseek-chat",
        "temperature": 0.0
    }

    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/{datestr}_test_django_16379_search_for_small_change"
    trajectory_path = f"{dir}/trajectory.json"

    loop = AgenticLoop(
        search_and_code_transitions_v2(global_params=global_params),
        workspace=workspace,
        trajectory_path=trajectory_path,
        prompt_log_dir=dir,
    )

    response = loop.run(message=instance["problem_statement"])

    diff = loop.workspace.file_repo.diff()

    assert workspace.file_context.has_span(
        "django/core/cache/backends/filebased.py", "FileBasedCache.has_key"
    )

    saved_loop = AgenticLoop.from_trajectory_file(trajectory_path=trajectory_path)

    saved_response = saved_loop.run(message=instance["problem_statement"])

    assert saved_response.status == response.status
    assert saved_response.message == response.message

    assert saved_loop.workspace.file_context.has_span(
        "django/core/cache/backends/filebased.py", "FileBasedCache.has_key"
    )

    assert saved_loop.workspace.file_repo.diff() == diff


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_run_and_reload_django_16379()
    model = "gpt-4o-mini-2024-07-18"
    test_simple_search(model)
    test_plan_to_code(model)
