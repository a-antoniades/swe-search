import pytest
from unittest.mock import Mock, patch

from moatless.benchmark.swebench import create_workspace, load_instance
from moatless.benchmark.utils import get_moatless_instance
from moatless.edit.plan_v2 import PlanToCode, TakeAction, RequestCodeChange, PlanRequest, Finish, ChangeType

@pytest.fixture
def scikit_learn_workspace():
    instance = get_moatless_instance("scikit-learn__scikit-learn-25570", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("sklearn/compose/tests/test_column_transformer.py", ["test_column_transformer_sparse_stacking"])
    return workspace

@pytest.fixture
def django_workspace():
    instance = get_moatless_instance("django__django-14016", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("django/db/models/query_utils.py", ["_combine", 'Q.__init__', 'Q._combine', 'Q.__or__'])

    return workspace

@pytest.fixture
def sympy_workspace():
    instance = load_instance("sympy__sympy-16988")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("sympy/sets/sets.py", ["imports", "imageset"])
    return workspace

def test_request_for_modifing_two_functions(scikit_learn_workspace):
    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=scikit_learn_workspace, initial_message="Test initial message")

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.addition,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=475,
        end_line=500,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'sklearn/compose/tests/test_column_transformer.py',
        'change_type': 'addition',
        'span_ids': ['test_column_transformer_mixed_cols_sparse', 'test_column_transformer_sparse_threshold'],
        'start_line': 477,
        'end_line': 548
    }

def test_request_for_delete_one_import(scikit_learn_workspace):
    """
    Include more than one line when deleting a single row to avoid deleting the wrong line
    """
    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=scikit_learn_workspace, initial_message="Test initial message")
    print(plan_to_code.file_context.create_prompt(show_line_numbers=True, show_outcommented_code=True, outcomment_code_comment="... rest of the code"))

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.deletion,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=12,
        end_line=12,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'sklearn/compose/tests/test_column_transformer.py',
        'change_type': 'modification',
        'span_ids': ['imports'],
        'start_line': 6,
        'end_line': 18
    }

def test_request_for_delete_one_function(scikit_learn_workspace):
    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=scikit_learn_workspace, initial_message="Test initial message")

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.deletion,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=452,
        end_line=474,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'sklearn/compose/tests/test_column_transformer.py',
        'change_type': 'deletion',
        'span_ids': ['test_column_transformer_sparse_stacking'],
        'start_line': 452,
        'end_line': 474
    }

def test_request_for_adding_function_on_outcommented_line(scikit_learn_workspace):
    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=scikit_learn_workspace, initial_message="Test initial message")
    print(plan_to_code.file_context.create_prompt(show_line_numbers=True, show_outcommented_code=True, outcomment_code_comment="... rest of the code"))

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.addition,
        file_path="sklearn/compose/tests/test_column_transformer.py",
        start_line=475,
        end_line=475,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    print("outcome", outcome)

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'sklearn/compose/tests/test_column_transformer.py',
        'change_type': 'addition',
        'span_ids': ['test_column_transformer_sparse_stacking'],
        'start_line': 452,
        'end_line': 476
    }

def test_request_for_modification_one_small_function(django_workspace):
    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=django_workspace, initial_message="Test initial message")
    print(plan_to_code.file_context.create_prompt(show_line_numbers=True, show_outcommented_code=True, outcomment_code_comment="... rest of the code"))

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.modification,
        file_path="django/db/models/query_utils.py",
        start_line=43,
        end_line=58,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'django/db/models/query_utils.py',
        'change_type': 'modification',
        'span_ids': ['Q._combine'],
        'start_line': 43,
        'end_line': 58
    }

    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=django_workspace, initial_message="Test initial message", min_tokens_in_edit_prompt=200)
    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'django/db/models/query_utils.py',
        'change_type': 'modification',
        'span_ids': ['Q.__init__', 'Q._combine', 'Q.__or__'],
        'start_line': 40,
        'end_line': 61
    }


def test_request_for_modification_class_init_span(django_workspace):
    """
    Verify that the class init span is included if it covers the start and end line
    """

    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=django_workspace, initial_message="Test initial message")
    print(plan_to_code.file_context.create_prompt(show_line_numbers=True, show_outcommented_code=True, outcomment_code_comment="... rest of the code"))

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.modification,
        file_path="django/db/models/query_utils.py",
        start_line=29,
        end_line=29,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'django/db/models/query_utils.py',
        'change_type': 'modification',
        'span_ids': ['Q', 'Q.__init__'],
        'start_line': 29,
        'end_line': 41
    }


def test_request_for_modification_class_with_decorator():
    """
    Verify that the class init span is included if it covers the start and end line
    """
    instance = get_moatless_instance("django__django-10914", split="lite")
    workspace = create_workspace(instance)

    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=workspace, initial_message="Test initial message")
    plan_to_code.file_context.add_spans_to_context("tests/file_uploads/tests.py",  {"FileUploadTests", "FileUploadTests.setUpClass", "FileUploadTests.tearDownClass"})
    print(plan_to_code.file_context.create_prompt(show_line_numbers=True, show_outcommented_code=True, outcomment_code_comment="... rest of the code"))

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.modification,
        file_path="tests/file_uploads/tests.py",
        start_line=26,
        end_line=26,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'django/db/models/query_utils.py',
        'change_type': 'modification',
        'span_ids': ['Q', 'Q.__init__'],
        'start_line': 29,
        'end_line': 41
    }

def test_unexpected_end_line(sympy_workspace):
    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=sympy_workspace, initial_message="Test initial message")

    print(plan_to_code.file_context.create_prompt(show_line_numbers=True))
    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.modification,
        file_path="sympy/sets/sets.py",
        start_line=1774,
        end_line=1894,
    )

    outcome = plan_to_code.execute(PlanRequest(action=request))
    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'sympy/sets/sets.py',
        'change_type': 'modification',
        'span_ids': ['imageset'],
        'start_line': 1774,
        'end_line': 1896
    }

    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=sympy_workspace, max_tokens_in_edit_prompt=500, initial_message="Test initial message")
    outcome = plan_to_code.execute(PlanRequest(action=request))
    print(outcome)
    assert outcome.trigger == "retry"

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.modification,
        file_path="sympy/sets/sets.py",
        start_line=1864,
        end_line=1894,
    )
    outcome = plan_to_code.execute(PlanRequest(action=request))

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'sympy/sets/sets.py',
        'change_type': 'modification',
        'span_ids': ['imageset'],
        'start_line': 1859,
        'end_line': 1896
    }


def test_include_full_blocks_in_prompt():
    instance = get_moatless_instance("django__django-13768", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("django/dispatch/dispatcher.py", ["Signal", "Signal.__init__"])
    print(workspace.file_context.create_prompt(show_line_numbers=True))

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.addition,
        file_path="django/dispatch/dispatcher.py",
        start_line=22,
        end_line=23,
    )

    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=workspace, max_tokens_in_edit_prompt=500, initial_message="Test initial message")
    outcome = plan_to_code.execute(PlanRequest(action=request))

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'file_path': 'django/dispatch/dispatcher.py',
        'change_type': 'modification',
        'span_ids': ['Signal'],
        'start_line': 21,
        'end_line': 29
    }


def test_verify_get_start_and_end_lines():
    instance = get_moatless_instance("pytest-dev__pytest-11143", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("src/_pytest/assertion/rewrite.py", ["AssertionRewriter", "AssertionRewriter.is_rewrite_disabled"])

    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.modification,
        file_path="src/_pytest/assertion/rewrite.py",
        start_line=745,
        end_line=746,
    )

    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=workspace, max_tokens_in_edit_prompt=500, initial_message="Test initial message")
    outcome = plan_to_code.execute(PlanRequest(action=request))

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'end_line': 753,
        'change_type': 'modification',
        'file_path': 'src/_pytest/assertion/rewrite.py',
        'span_ids': ['AssertionRewriter.is_rewrite_disabled', 'AssertionRewriter.variable'],
        'start_line': 744
    }


def test_modify_two_functions():
    instance_id = "matplotlib__matplotlib-26011"
    instance = get_moatless_instance(instance_id, split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("lib/matplotlib/axes/_base.py", ["twinx", "twiny"])
    print(workspace.file_context.create_prompt(show_line_numbers=True))
    request = RequestCodeChange(
        scratch_pad="Applying change",
        planned_steps=[],
        instructions="Fix",
        pseudo_code="# foo",
        change_type=ChangeType.modification,
        file_path="lib/matplotlib/axes/_base.py",
        start_line=2800,
        end_line=2850,
    )

    plan_to_code = PlanToCode(id=0, model="gpt", _workspace=workspace, max_tokens_in_edit_prompt=1000, initial_message="Test initial message")
    outcome = plan_to_code.execute(PlanRequest(action=request))

    assert outcome.trigger == "edit_code"
    assert outcome.output == {
        'instructions': 'Fix',
        'pseudo_code': '# foo',
        'end_line': 753,
        'change_type': 'modification',
        'file_path': 'src/_pytest/assertion/rewrite.py',
        'span_ids': ['AssertionRewriter.is_rewrite_disabled', 'AssertionRewriter.variable'],
        'start_line': 744
    }

def test_action_dump():
    take_action = PlanRequest(action=Finish(scratch_pad="", reason="Task completed successfully"))
    dump = take_action.model_dump()
    assert dump == {'action': {'scratch_pad': '', 'finish_reason': 'Task completed successfully'}, 'action_name': 'Finish'}

    action = PlanRequest.model_validate(dump)
    assert isinstance(action.action, Finish)
    assert action == take_action


def test_parse_plan_request():

    print(PlanRequest.model_json_schema())

    request = PlanRequest.model_validate_json("{\n  \"action\": {\n    \"scratch_pad\": \"The issue is a race condition in the `has_key` method where the file might be deleted between the `os.path.exists` check and the `open` call. To fix this, we should use a try-except block to handle the `FileNotFoundError` that might occur when trying to open the file.\",\n    \"change_type\": \"modification\",\n    \"instructions\": \"Modify the `has_key` method to catch `FileNotFoundError` and return `False` if the file does not exist.\",\n    \"start_line\": 91,\n    \"end_line\": 96,\n    \"pseudo_code\": \"def has_key(self, key, version=None):\\n    fname = self._key_to_file(key, version)\\n    try:\\n        with open(fname, \\\"rb\\\") as f:\\n            return not self._is_expired(f)\\n    except FileNotFoundError:\\n        return False\",\n    \"file_path\": \"django/core/cache/backends/filebased.py\",\n    \"planned_steps\": [\n      \"Add a test case to verify that the `has_key` method correctly handles the case where the file is deleted between the `exists` check and the `open` call.\"\n    ]\n  }\n}")
    assert isinstance(request.action, RequestCodeChange)
    print(request.model_dump_json(indent=2))


