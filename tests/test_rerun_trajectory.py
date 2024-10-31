import json
import os
import tempfile

import litellm
import pytest
from unittest.mock import MagicMock, patch

from testbed.client.sdk import TestbedSDK

from moatless.benchmark.utils import trace_metadata, get_moatless_instance
from moatless.edit import EditCode, PlanToCode, ClarifyCodeChange
from moatless.loop import AgenticLoop
from moatless.benchmark.swebench import create_workspace, load_instance
from moatless.repository import GitRepository
from moatless.settings import Settings
from moatless.state import AgenticState
from moatless.trajectory import Trajectory

# django__django-9296.json check if the LLM corrects on retry

@pytest.mark.skip(reason="Test is not ready")
def test_expect_failed_edit():
    instance_id = "django__django-11095"
    trajectory = Trajectory.load(f"tests/trajectories/{instance_id}/trajectory.json")
    Settings.cheap_model = None

    instance = load_instance(instance_id, dataset_name="princeton-nlp/SWE-bench_Verified")
    workspace = create_workspace(instance)
    assert isinstance(workspace.file_repo, GitRepository)
    mocked_actions = trajectory.get_mocked_actions()
    expected_states = trajectory.get_expected_states()

    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    def verify_state_func(state: AgenticState):
        if isinstance(state, EditCode):
            for message in state.messages():
                print(f"Role: {message.role}")
                print(message.content)

    trajectory_path = f"tests/trajectories/{instance_id}/trajectory-correction.json"
    loop = AgenticLoop(
        trajectory.transition_rules, workspace=workspace, mocked_actions=mocked_actions, verify_state_func=verify_state_func, continue_after_mocks=False, trajectory_path=trajectory_path, metadata=trace_metadata(instance_id=instance["instance_id"], session_id="test_rerun", trace_name="test_rerun")
    )
    response = loop.run(message=trajectory.initial_message)



#@pytest.mark.skip(reason="Test is not ready")
def test_updated_file_context():
    """
    Verify that the file context is updated after code as been edited.
    """
    instance_id = "django__django-13768"
    trajectory = Trajectory.load(f"tests/trajectories/{instance_id}/trajectory.json")
    Settings.cheap_model = None

    instance = get_moatless_instance(instance_id)
    workspace = create_workspace(instance, use_perfect_file_context=True)
    assert isinstance(workspace.file_repo, GitRepository)
    mocked_actions = trajectory.get_mocked_actions()
    expected_states = trajectory.get_expected_states()

    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    result_list = []

    def verify_state_func(state: AgenticState):
        if isinstance(state, PlanToCode):
            print(f"===========================")
            print(f"{state.id} {state.name}")
            message = state.messages()[-1]
            print(message.content)

            print(json.dumps(state.workspace.file_context.model_dump(), indent=2))

            print("File:")
            print(state.file_repo.get_file("django/dispatch/dispatcher.py").content)

            print("File context:")

            file_context_str = state.workspace.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
            print(file_context_str)

    trajectory_path = f"tests/trajectories/{instance_id}/trajectory-correction.json"
    loop = AgenticLoop(
        trajectory.transition_rules, workspace=workspace, mocked_actions=mocked_actions, verify_state_func=verify_state_func, continue_after_mocks=False, trajectory_path=trajectory_path, metadata=trace_metadata(instance_id=instance["instance_id"], session_id="test_rerun", trace_name="test_rerun")
    )

    try:
        response = loop.run(message=trajectory.initial_message)
    except Exception as e:
        print(e)
    finally:
        for c, status in result_list:
            assert status






# @pytest.mark.skip(reason="Test is not ready")
def test_testbed_verification():
    instance_id = "django__django-15731"
    trajectory = Trajectory.load(f"tests/trajectories/{instance_id}/trajectory.json")
    Settings.cheap_model = None

    instance = load_instance(instance_id, dataset_name="princeton-nlp/SWE-bench_Verified")

    with TestbedSDK(instance_id, os.getenv("KUBE_NAMESPACE", "testbed-dev"), dataset_name="princeton-nlp/SWE-bench_Verified") as sdk:
        workspace = create_workspace(instance, testbed=sdk.testbed)
        assert isinstance(workspace.file_repo, GitRepository)
        mocked_actions = trajectory.get_mocked_actions()
        expected_states = trajectory.get_expected_states()

        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]

        def verify_state_func(state: AgenticState):
            print(f"verify_state_func {state.name}")

        trajectory_path = f"tests/trajectories/{instance_id}/trajectory-correction.json"
        loop = AgenticLoop(
            trajectory.transition_rules, workspace=workspace, mocked_actions=mocked_actions, verify_state_func=verify_state_func, continue_after_mocks=True, trajectory_path=trajectory_path, metadata=trace_metadata(instance_id=instance["instance_id"], session_id="test_rerun", trace_name="test_rerun")
        )
        response = loop.run(message=trajectory.initial_message)
