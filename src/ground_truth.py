"""
Ground truth evaluation functions for checking answer correctness.
"""


def _rate_correctness_equal(ground_truth_answer: str, answer: str) -> dict:
    """Check if the answer exactly matches the ground truth.

    Args:
        ground_truth_answer: The correct answer
        answer: The model's answer

    Returns:
        Dict with 'is_equal' key indicating exact match
    """
    # Fix: Remove the incorrect '== str' comparison
    if isinstance(answer, str) and isinstance(ground_truth_answer, str):
        is_equal = answer.strip().lower() == ground_truth_answer.strip().lower()
    else:
        # Try numeric comparison if possible
        try:
            # Convert both to float for numeric comparison
            ground_truth_num = float(str(ground_truth_answer).strip().replace(",", ""))
            answer_num = float(str(answer).strip().replace(",", ""))
            # Use small tolerance for floating point comparison
            is_equal = abs(ground_truth_num - answer_num) < 1e-6
        except (ValueError, TypeError):
            # Fall back to string comparison
            is_equal = str(answer).strip() == str(ground_truth_answer).strip()

    return {"is_equal": is_equal}


def _rate_correctness_subset(ground_truth_answer: str, answer: str) -> dict:
    """Check if the answer contains the ground truth as a substring.

    Args:
        ground_truth_answer: The correct answer
        answer: The model's answer

    Returns:
        Dict with 'contains_answer' key indicating substring match
    """
    # Convert to strings and check containment (case-insensitive)
    ground_truth_str = str(ground_truth_answer).strip().lower()
    answer_str = str(answer).strip().lower()

    contains = ground_truth_str in answer_str if ground_truth_str else False

    return {"contains_answer": contains}


def rate_correctness(ground_truth_answer: str, answer: str) -> dict:
    """Rate the correctness of an answer against ground truth.

    Checks both exact match and substring containment.

    Args:
        ground_truth_answer: The correct answer
        answer: The model's answer

    Returns:
        Dict with 'is_equal' and 'contains_answer' keys
    """
    results = {}
    results.update(_rate_correctness_equal(ground_truth_answer, answer))
    results.update(_rate_correctness_subset(ground_truth_answer, answer))
    return results