import pytest
from unittest.mock import Mock, patch

from moatless.benchmark.swebench import create_workspace
from moatless.benchmark.utils import get_moatless_instance
from moatless.edit.clarify import ClarifyCodeChange, LineNumberClarification
from moatless.state import StateOutcome, FileWithSpans
from moatless.workspace import Workspace
from moatless.file_context import FileContext
from moatless.repository import CodeFile
from moatless.codeblocks.codeblocks import BlockSpan, CodeBlock
from moatless.codeblocks import CodeBlockType

class TestClarifyCodeChange:
    @pytest.fixture
    def clarify_code_change(self):
        mock_file_repo = Mock()
        mock_workspace = Workspace(file_repo=mock_file_repo)
        mock_file_context = Mock(spec=FileContext)
        
        return ClarifyCodeChange(
            id=2,
            instructions="Update function",
            file_path="test.py",
            span_id="span1",
            _workspace=mock_workspace,
            file_context=mock_file_context
        )

    def test_action_type(self, clarify_code_change: ClarifyCodeChange):
        assert clarify_code_change.action_type() == LineNumberClarification

    @patch('moatless.edit.clarify.ClarifyCodeChange._verify_line_numbers')
    def test_execute_action_reject(self, mock_verify, clarify_code_change: ClarifyCodeChange):
        action = LineNumberClarification(
            scratch_pad="Cannot complete the task",
            start_line=1,
            end_line=5,
            reject=True
        )

        response = clarify_code_change._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "reject"
        assert response.output["message"] == "Cannot complete the task"

    @patch('moatless.edit.clarify.ClarifyCodeChange._verify_line_numbers')
    @patch('moatless.edit.clarify.ClarifyCodeChange.get_line_span')
    def test_execute_action_edit_code(self, mock_get_line_span, mock_verify, clarify_code_change: ClarifyCodeChange):
        action = LineNumberClarification(
            scratch_pad="Updating lines",
            start_line=2,
            end_line=4
        )

        mock_verify.return_value = None
        mock_get_line_span.return_value = (2, 4)

        response = clarify_code_change._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "edit_code"
        assert response.output["instructions"] == "Update function\n\nUpdating lines"
        assert response.output["file_path"] == "test.py"
        assert response.output["span_id"] == "span1"
        assert response.output["start_line"] == 2
        assert response.output["end_line"] == 4

    @patch('moatless.edit.clarify.ClarifyCodeChange._verify_line_numbers')
    def test_execute_action_retry(self, mock_verify, clarify_code_change: ClarifyCodeChange):
        action = LineNumberClarification(
            scratch_pad="Retry needed",
            start_line=1,
            end_line=10
        )

        mock_verify.return_value = "Invalid line numbers"

        response = clarify_code_change._execute_action(action)

        assert isinstance(response, StateOutcome)
        assert response.trigger == "retry"
        assert response.retry_message == "Invalid line numbers"

    def test_required_fields(self, clarify_code_change: ClarifyCodeChange):
        assert clarify_code_change.required_fields() == {"instructions", "file_path", "span_id"}

    def test_messages(self, clarify_code_change: ClarifyCodeChange):
        # TODO: Test init() properly
        clarify_code_change._file_context_str = "Mock file context"
        messages = clarify_code_change.messages()

        assert len(messages) == 1
        assert "<instructions>" in messages[0].content
        assert "Update function" in messages[0].content
        assert "<code>" in messages[0].content
        assert "Mock file context" in messages[0].content
        

    @patch('moatless.repository.CodeFile')
    @patch('moatless.codeblocks.codeblocks.BlockSpan')
    def test_verify_line_numbers_valid(self, mock_span, mock_file, clarify_code_change: ClarifyCodeChange):
        mock_file.content = "line1\nline2\nline3\nline4\nline5"
        mock_span.start_line = 1
        mock_span.end_line = 5
        clarify_code_change._file = mock_file
        clarify_code_change._span = mock_span

        action = LineNumberClarification(
            scratch_pad="Valid lines",
            start_line=2,
            end_line=4
        )

        result = clarify_code_change._verify_line_numbers(action)

        assert result is None

    @patch('moatless.repository.CodeFile')
    @patch('moatless.codeblocks.codeblocks.BlockSpan')
    def test_verify_line_numbers_invalid(self, mock_span, mock_file, clarify_code_change: ClarifyCodeChange):
        mock_file.content = "line1\nline2\nline3\nline4\nline5"
        mock_span.start_line = 1
        mock_span.end_line = 5
        clarify_code_change._file = mock_file
        clarify_code_change._span = mock_span

        action = LineNumberClarification(
            scratch_pad="Invalid lines",
            start_line=1,
            end_line=5
        )

        result = clarify_code_change._verify_line_numbers(action)

        assert result is not None
        assert "covers the whole code span" in result

def test_clarify_class():
    instance = get_moatless_instance("django__django-11095", split="verified")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("django/contrib/admin/checks.py", ["ModelAdminChecks", "ModelAdminChecks._check_inlines_item"])

    clarify_code = ClarifyCodeChange(id=0, _workspace=workspace, initial_message="Test initial message", instructions="update", span_id="ModelAdminChecks", file_path="django/contrib/admin/checks.py")
    prompt_message = clarify_code.messages()[-1].content

    assert "class ModelAdminChecks(BaseModelAdminChecks):" in prompt_message
    assert "def _check_inlines_item(self, obj, inline, label):" in prompt_message
    assert "return inline(obj.model, obj.admin_site).check()" in prompt_message

def test_clarify_line_after_span():
    instance = get_moatless_instance("mwaskom__seaborn-3190", split="lite")
    workspace = create_workspace(instance)
    workspace.file_context.add_spans_to_context("tests/_core/test_scales.py", ["TestContinuous.test_label_type_checks"])

    clarify_code = ClarifyCodeChange(id=0, _workspace=workspace, initial_message="Test initial message", instructions="update", span_id="TestContinuous", file_path="tests/_core/test_scales.py")
    clarify_code.init()
    prompt_message = clarify_code.messages()[-1].content

    assert "class TestContinuous:" in prompt_message
    assert "def test_label_type_checks" in prompt_message

    action = LineNumberClarification(
        scratch_pad="Add the function",
        start_line=314,
        end_line=314
    )
    outcome = clarify_code.execute(action)
    assert outcome.output["file_path"] == "tests/_core/test_scales.py"
    assert outcome.output["span_id"] == "TestContinuous"
    assert outcome.output["start_line"] == 298
    assert outcome.output["end_line"] == 313
