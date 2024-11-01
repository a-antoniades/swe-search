import hashlib
import json
import logging
import re
from datetime import datetime
from typing import List

from moatless.file_context import RankedFileSpan
from moatless.verify.verify import Verifier

from moatless.repository import GitRepository, CodeFile
from moatless.schema import TestResult, TestStatus, VerificationIssueType, FileWithSpans
from testbed.schema import EvaluationResult, TraceItem
from testbed.sdk import TestbedClient, TestbedSDK

logger = logging.getLogger(__name__)


class TestbedVerifier(Verifier):

    def __init__(self, testbed_sdk: TestbedSDK, repository: GitRepository, instance: dict = None, max_context_tokens: int = 2000):
        self.testbed_sdk = testbed_sdk
        self.repository = repository
        self.instance = instance
        self.max_context_tokens = max_context_tokens
        self.tests_to_ignore = []
        self.log_dir = None

    @classmethod
    def from_instance(cls, instance: dict, repository: GitRepository, **kwargs):
        return cls(testbed_sdk=TestbedSDK(), repository=repository, instance=instance, **kwargs)

    def run_tests(self, test_files: list[str]) -> List[TestResult]:
        patch = self.repository.diff()
        if patch and not patch.endswith("\n"):
            patch += "\n"

        log_content = "# Test Run\n\n"
        log_content += f"Files: {test_files}"
        log_content += f"\n\n# Patch:\n```diff\n{patch}\n```"

        testbed = None
        try:
            testbed = self.testbed_sdk.create_client(instance_id=self.instance["instance_id"])
            response = testbed.run_tests(test_files=test_files, patch=patch)

            if response.output:
                log_content += f"\n\n## Log:\n{response.output}\n"

            if response.test_results:
                log_content += f"\n\n## Testbed test results:"
                test_results_json = response.model_dump_json(exclude={"output"}, indent=2)
                log_content += f"```json\n{test_results_json}\n```"

            # Ignore tests that fails before any changes were made
            if not patch:
                self.tests_to_ignore = [test.name for test in response.test_results if test.status in ["ERROR", "FAILED"]]
                if self.tests_to_ignore and self.log_dir:
                    log_content += f"\n\n## Ignored tests:\n{self.tests_to_ignore}"
                    with open(f"{self.log_dir}/ignored_tests.json", "w") as f:
                        json.dump(self.tests_to_ignore, f)

            test_results = [test for test in response.test_results if test.name not in self.tests_to_ignore]

            mapped_results = self._map_test_results_to_issues(test_results)

            if mapped_results:
                dicts = [result.model_dump() for result in mapped_results]
                for dict in dicts:
                    dict["status"] = dict["status"].value
                log_content += f"\n\n## Mapped test results:\n```json\n{json.dumps(dicts, indent=2)}\n```"

            return mapped_results
        except Exception as e:
            logger.exception(f"Error running tests {test_files}")
            log_content += f"\n\n## Error:\n{e}"
            import traceback
            traceback = traceback.format_exc()
            log_content += f"\n\n# Traceback:\n{traceback}"
            return []
        finally:
            if testbed:
                try:
                    testbed.destroy()
                except Exception as e:
                    logger.error(f"Error destroying testbed client: {e}")
            if self.log_dir:
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                with open(f"{self.log_dir}/{datetime_str}_test_run.md", "w") as f:
                    f.write(log_content)

    def evaluate(self) -> EvaluationResult | None:
        if not self.instance:
            logger.warning("No instance provided for evaluation")
            return None

        logger.info(f"Running evaluation for instance {self.instance['instance_id']}")

        test_patch_files = self.instance.get("test_file_spans", {}).keys()

        log_content = "# Evaluation\n\n"
        log_content += f"Test files: {test_patch_files}"

        testbed = None
        try:
            testbed = self.testbed_sdk.create_client(instance_id=self.instance["instance_id"])

            diff = self.repository.diff(ignore_paths=list(test_patch_files))
            if diff:
                if diff.endswith("\n"):
                    diff += "\n"

                log_content += f"\n\n# Patch:\n```diff\n{diff}\n```"

                evaluation_result = testbed.run_evaluation(patch=diff)

                if evaluation_result.output:
                    log_content += f"\n\n## Log:\n```\n{evaluation_result.output}\n```\n"

                log_content += f"\n\n## Evaluation result:\n```json\n{evaluation_result.model_dump_json(indent=2)}\n```"
                return evaluation_result
            else:
                log_content += "\n\n# Patch: No changes to evaluate"
                logger.info("No changes to evaluate")
        except Exception as e:
            logger.exception("Error running evaluation")
            log_content += f"\n\n## Error:\n{e}"
            import traceback
            traceback = traceback.format_exc()
            log_content += f"\n\n# Traceback:\n{traceback}"
        finally:
            if self.log_dir:
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                with open(f"{self.log_dir}/{datetime_str}_evaluation.md", "w") as f:
                    f.write(log_content)
            if testbed:
                try:
                    testbed.destroy()
                except Exception as e:
                    logger.error(f"Error destroying testbed client: {e}")
        return None

    def _get_code_block(self, file_path: str, line_number: int):
        file = self.repository.get_file(file_path)
        if not file or not file.module:
            return None

        block = file.module.find_first_by_start_line(line_number)
        if not block or not block.belongs_to_span:
            return None

        return block

    def _relevant_files_from_trace(self, trace_items: List[TraceItem]) -> List[RankedFileSpan]:
        ranked_file_spans = []

        for i, trace_item in enumerate(trace_items):
            block = self._get_code_block(trace_item.file_path, trace_item.line_number)

            if not block:
                continue

            ranked_file_spans.append(RankedFileSpan(
                file_path=trace_item.file_path,
                span_id=block.belongs_to_span.span_id,
                rank=i,
            ))

        return ranked_file_spans

    def _hash_output(self, output: str):
        """
        Hash only lines with > or E and the last line if it matches the format path:line_number: <Error>
        """
        lines = output.split("\n")
        
        # Regular expression to match the format path:line_number: <Error>
        error_regex = re.compile(r'.+:\d+:.+')
        
        # Check if the last line matches the regex
        if error_regex.match(lines[-1]):
            return hashlib.sha256(lines[-1].encode()).hexdigest()
        
        filtered_out_lines = [line for line in lines if line.startswith("E ") or line.startswith("> ")]
        return hashlib.sha256("\n".join(filtered_out_lines).encode()).hexdigest()

    def _map_test_results_to_issues(self, test_results: List) -> List[TestResult]:
        logger.debug(f"Map {len(test_results)} test results.")

        root_causes = set()
        ignored_tests = 0

        mapped_results = []
        for test_result in test_results:
            trace_items = test_result.stacktrace

            test_status = TestStatus(test_result.status)

            # reverse to start from root cause method on ERROR
            if test_status == TestStatus.ERROR:
                trace_items.reverse()

            hashed_section = None

            ignored_errors = ["PermissionError not raised", "DeprecationWarning"]

            # DeprecationWarnings are probably false negatives because of incorrect dependencies in the testbed environment
            if test_result.failure_output:
                last_line = [line for line in test_result.failure_output.split("\n") if line.strip()][-1]
                if any(error in last_line for error in ignored_errors):
                    logger.info(f"Skipping test {test_result.method} in {test_result.file_path} with ignored error on last line: '{last_line}'")
                    continue

            if not test_result.failure_output:
                logger.debug("Skipping test {test_result.method} in {test_result.file_path} with no failure output")
                test_output = None
            else:
                failure_sections = test_result.failure_output.split("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
                if len(failure_sections) > 1:
                    # skip tests with the same root cause
                    hashed_section = self._hash_output(failure_sections[-1])
                elif trace_items and trace_items[0].output:
                    hashed_section = hashlib.sha256((str(trace_items[0])).encode()).hexdigest()
                else:
                    hashed_section = self._hash_output(test_result.failure_output)

                if hashed_section in root_causes:
                    ignored_tests += 1
                    test_output = None
                else:
                    # If the test has more than 50 lines justp ick the last 50
                    if len(test_result.failure_output.split("\n")) > 50:
                        test_output = "\n".join(test_result.failure_output.split("\n")[-50:])
                    else:
                        test_output = test_result.failure_output

            relevant_files = self._relevant_files_from_trace(trace_items)
            if not test_result.file_path or not test_result.method:

                # Use the file with the root cause for the error if no file path or method is set
                if test_status in [TestStatus.ERROR, TestStatus.FAILED] and relevant_files:
                    test_result.file_path = relevant_files[0].file_path
                    test_result.method = relevant_files[0].span_id
                    logger.info(f"No filepath found on test \"{test_result.name}\". Using file path {test_result.file_path} and {test_result.method} from trace for test {test_result.name}")
                elif test_status in [TestStatus.ERROR, TestStatus.FAILED] and not test_result.file_path :
                    logger.warning(f"Could not find file path and/or method for test {test_result}")
                elif test_status in [TestStatus.ERROR, TestStatus.FAILED] and not test_result.method:
                    logger.info(f"Could not find method for test {test_result}")

            if test_result.method:
                method = test_result.method
                if "[" in method:
                    method = method.split("[")[0]
            else:
                method = None

            if test_result.file_path:
                file = self.repository.get_file(test_result.file_path)
                if not file and trace_items:
                    for item in trace_items:
                        file = self.repository.get_file(item.file_path)
                        if file:
                            method = item.method
                            break
            else:
                file = None

            if file and file.module and method:
                block = None
                if "." in method:
                    path = method.split(".")
                    logger.debug(f"Looking for path {path} in file {file.file_path}")
                    block = file.module.find_by_path(path)
                    if not block:
                        method = path[-1]

                if method == "<module>" and trace_items:
                    block = file.module.children[0]
                    for item in trace_items:
                        if item.file_path == file.file_path and item.line_number:
                            block = file.module.find_first_by_start_line(item.line_number)
                            break

                if not block:
                    block = file.module.find_by_identifier(method, recursive=True)

                if block:
                    span_id = block.belongs_to_span.span_id
                    existing_issue = next((issue for issue in mapped_results if issue.span_id == span_id and issue.file_path == file.file_path), None)
                    if existing_issue:
                        logger.debug(f"Skipping content on duplicate span id {span_id} for failure in {file.file_path} and method {method}.")
                        test_output = None
                else:
                    span_id = None

                mapped_results.append(TestResult(
                    status=test_status,
                    message=test_output,
                    file_path=file.file_path,
                    span_id=span_id,
                    relevant_files=relevant_files,
                ))

                if hashed_section:
                    root_causes.add(hashed_section)
            elif test_output:
                logger.info(f"Could not find file {test_result.file_path} or method in test \"{test_result.name}\"")
                mapped_results.append(TestResult(
                    status=test_status,
                    message=test_output
                ))
            elif test_status in [TestStatus.ERROR, TestStatus.FAILED]:
                logger.warning(f"Could not find file {test_result.file_path} or method in test \"{test_result.name}\" and no output exists, will ignore")
            else:
                logger.info(f"Skipping test {test_result.name} with status {test_status}")

        if ignored_tests:
            logger.info(f"Ignored {ignored_tests} tests with redundant root cause")

        return mapped_results
