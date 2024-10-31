from moatless.search.reward import parse_value, parse_explanation, parse_alternative_suggestion

def test_parse_value():
    # Test various formats of reward
    assert parse_value("<Reward>9</Reward>") == 9
    assert parse_value("<Reward>9") == 9
    assert parse_value("Reward: 9") == 9
    assert parse_value("**Reward**: 9") == 9
    assert parse_value("<Reward>: 9") == 9
    assert parse_value("**<Reward>**: 9") == 9
    assert parse_value("**Reward:** 9") == 9
    
    # Test negative values
    assert parse_value("<Reward>-5</Reward>") == -5
    assert parse_value("<Reward>-5") == -5
    
    # Test string values (should return None and log a warning)
    assert parse_value("<Reward>'test'</Reward>") is None
    
    assert parse_value("<Reward>'test'") is None
    
    # Test with allowed_values
    assert parse_value("<Reward>5</Reward>", allowed_values=range(1, 11)) == 5
    assert parse_value("<Reward>15</Reward>", allowed_values=range(1, 11)) is None
    
    # Test with different keyword
    assert parse_value("<Score>7</Score>", keyword='score') == 7
    assert parse_value("<Score>7", keyword='score') == 7
    
    # Test invalid formats
    assert parse_value("No reward here") is None
    
    assert parse_value("<Reward>invalid</Reward>") is None
    
    assert parse_value("<Reward>invalid") is None

    complex_response = """<Explanation> Some explanation text here.
    <Reward> 50"""
    assert parse_value(complex_response) == 50

    # Test new format with spaces inside tags
    assert parse_value("< Reward >: 65") == 65
    assert parse_value("< reward >: -30") == -30
    assert parse_value("< ReWaRd >: 100") == 100

    # Test new format with spaces and newlines
    assert parse_value("< Reward >\n: 75") == 75

def test_parse_explanation():
    # Test with closing tag
    content = "<Explanation>This is an explanation.\nIt spans multiple lines.</Explanation>\n<Reward>10</Reward>"
    assert parse_explanation(content) == "This is an explanation.\nIt spans multiple lines."
    
    # Test without closing tag
    content_no_closing_tag = "<Explanation>This is another explanation\n<Reward>5</Reward>"
    assert parse_explanation(content_no_closing_tag) == "This is another explanation"
    
    # Test with no explanation
    content_no_explanation = "There's no explanation here.\n<Reward>0</Reward>"
    assert parse_explanation(content_no_explanation) == content_no_explanation
    
    # Test with explanation after feedback
    content_explanation_after_feedback = """<Feedback_to_Alternative_Branch>This is feedback</Feedback_to_Alternative_Branch>
    <Explanation>This is the explanation
    <Reward>7</Reward>"""
    assert parse_explanation(content_explanation_after_feedback) == "This is the explanation"

def test_parse_alternative_suggestion():
    # Test with closing tag
    content = """<Explanation>This is the main explanation.</Explanation>
    <Feedback_to_Alternative_Branch>This is the alternative suggestion.</Feedback_to_Alternative_Branch>
    <Reward>8</Reward>"""
    assert parse_alternative_suggestion(content) == "This is the alternative suggestion."
    
    # Test without closing tag
    content_no_closing_tag = "<Feedback_to_Alternative_Branch>Another suggestion\n<Reward>5</Reward>"
    assert parse_alternative_suggestion(content_no_closing_tag) == "Another suggestion"
    
    # Test with no suggestion
    content_no_suggestion = "<Explanation>Only an explanation.</Explanation>\n<Reward>0</Reward>"
    assert parse_alternative_suggestion(content_no_suggestion) is None
    
    # Test with feedback before explanation
    content_feedback_before_explanation = """<Feedback_to_Alternative_Branch>This is feedback
    <Explanation>This is the explanation</Explanation>
    <Reward>6</Reward>"""
    assert parse_alternative_suggestion(content_feedback_before_explanation) == "This is feedback"

def test_parse_complex_response_with_line_breaks():
    complex_response = """<Explanation> 
    The search action was generally well-constructed, particularly in its inclusion of relevant class names and function names that align closely with the problem statement regarding `TransactionTestCase` and the deserialization functions. The resulting file context retrieved from `base/creation.py` is relevant as it contains the methods `serialize_db_to_string` and `deserialize_db_from_string`, which are directly related to the issue discussed in the problem statement. However, the search could have been improved by including a broader file pattern or also searching for related test cases that may demonstrate the bug or solution in action. The identified code correctly highlights the methods involved but does not include the proposed fix to wrap the deserialization process in a transaction, which is critical for addressing the problem effectively. Overall, while the existing search parameters were effective, enhancements could provide a more rounded understanding of the issue.
    </Explanation>
    <Feedback_to_Alternative_Branch> 
    An alternative search strategy could include fetching results from broader files that might define additional context or directly related test cases that could give insights into proper usage of the `TransactionTestCase` and its interaction with serialization/deserialization methods. This could involve searching across the entire `**/db/backends/**` directory or using terms that capture potential use cases and tests demonstrating the behavior connected to `serialized_rollback`.
    </Feedback_to_Alternative_Branch>
    <Reward> 
    70 
    </Reward>"""
    
    expected_explanation = """The search action was generally well-constructed, particularly in its inclusion of relevant class names and function names that align closely with the problem statement regarding `TransactionTestCase` and the deserialization functions. The resulting file context retrieved from `base/creation.py` is relevant as it contains the methods `serialize_db_to_string` and `deserialize_db_from_string`, which are directly related to the issue discussed in the problem statement. However, the search could have been improved by including a broader file pattern or also searching for related test cases that may demonstrate the bug or solution in action. The identified code correctly highlights the methods involved but does not include the proposed fix to wrap the deserialization process in a transaction, which is critical for addressing the problem effectively. Overall, while the existing search parameters were effective, enhancements could provide a more rounded understanding of the issue."""

    expected_feedback = """An alternative search strategy could include fetching results from broader files that might define additional context or directly related test cases that could give insights into proper usage of the `TransactionTestCase` and its interaction with serialization/deserialization methods. This could involve searching across the entire `**/db/backends/**` directory or using terms that capture potential use cases and tests demonstrating the behavior connected to `serialized_rollback`."""

    assert parse_explanation(complex_response) == expected_explanation
    assert parse_alternative_suggestion(complex_response) == expected_feedback
    assert parse_value(complex_response) == 70

def test_parse_mixed_tags_without_closing():
    mixed_response = """<Explanation>This is the main explanation.
    <Feedback_to_Alternative_Branch>This is the alternative suggestion.
    <Reward>60</Reward>"""
    
    assert parse_explanation(mixed_response) == "This is the main explanation."
    assert parse_alternative_suggestion(mixed_response) == "This is the alternative suggestion."
    assert parse_value(mixed_response) == 60

def test_parse_reversed_order_without_closing():
    reversed_response = """<Feedback_to_Alternative_Branch>This is the alternative suggestion.
    <Explanation>This is the main explanation.
    <Reward>40</Reward>"""
    
    assert parse_explanation(reversed_response) == "This is the main explanation."
    assert parse_alternative_suggestion(reversed_response) == "This is the alternative suggestion."
    assert parse_value(reversed_response) == 40
