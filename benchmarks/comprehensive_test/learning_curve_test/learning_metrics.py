"""
Learning Metrics - Measure system knowledge and learning progress

Provides accuracy measurement for learning curve tests.
"""

import re
from typing import List, Dict, Any, Tuple


def calculate_accuracy(
    system_results: List[Dict[str, Any]],
    expected_answers: List[str],
    threshold: float = 0.5
) -> float:
    """
    Calculate accuracy of system's memory retrieval.

    Args:
        system_results: Search results from memory system
        expected_answers: List of expected answer keywords
        threshold: Similarity threshold for match (0.0-1.0)

    Returns:
        Accuracy score (0.0-1.0)
    """
    if not expected_answers:
        return 0.0

    matches = 0
    for expected in expected_answers:
        if any(keyword_match(expected, result.get('content', ''), threshold)
               for result in system_results):
            matches += 1

    return matches / len(expected_answers)


def keyword_match(expected: str, text: str, threshold: float = 0.5) -> bool:
    """
    Check if expected keywords appear in text.

    Uses simple keyword matching + stemming for deterministic results.

    Args:
        expected: Expected answer/keyword
        text: Retrieved memory text
        threshold: Minimum word overlap ratio (0.0-1.0)

    Returns:
        True if match found
    """
    # Normalize text
    expected_lower = expected.lower().strip()
    text_lower = text.lower()

    # Exact substring match (easy case)
    if expected_lower in text_lower:
        return True

    # Word overlap match
    expected_words = set(tokenize(expected_lower))
    text_words = set(tokenize(text_lower))

    if not expected_words:
        return False

    overlap = len(expected_words & text_words)
    overlap_ratio = overlap / len(expected_words)

    return overlap_ratio >= threshold


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization: split on non-alphanumeric, filter stopwords.
    """
    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b', text.lower())

    # Filter stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'can', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'them', 'their', 'this', 'that', 'these', 'those'
    }

    return [w for w in words if w not in stopwords and len(w) > 2]


def evaluate_retrieval_quality(
    query: str,
    results: List[Dict[str, Any]],
    ground_truth_elements: List[str]
) -> Dict[str, float]:
    """
    Evaluate quality of memory retrieval for a query.

    Args:
        query: Search query
        results: Retrieved memories
        ground_truth_elements: Expected elements that should be retrieved

    Returns:
        Dict with precision, recall, f1_score
    """
    if not results or not ground_truth_elements:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    # Check which ground truth elements were retrieved
    retrieved_elements = set()
    for element in ground_truth_elements:
        for result in results:
            if keyword_match(element, result.get('content', '')):
                retrieved_elements.add(element)
                break

    # Calculate metrics
    true_positives = len(retrieved_elements)
    false_positives = len(results) - true_positives
    false_negatives = len(ground_truth_elements) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'retrieved_count': len(results),
        'relevant_count': true_positives
    }


def check_memory_relevance(
    memory_content: str,
    ideal_elements: List[str],
    match_threshold: float = 0.5
) -> bool:
    """
    Check if a memory contains relevant information.

    Args:
        memory_content: Text content of memory
        ideal_elements: List of expected relevant elements
        match_threshold: Minimum ratio of elements that must match

    Returns:
        True if memory is relevant
    """
    if not ideal_elements:
        return False

    matches = sum(
        1 for element in ideal_elements
        if keyword_match(element, memory_content)
    )

    match_ratio = matches / len(ideal_elements)
    return match_ratio >= match_threshold


def score_response_quality(
    retrieved_memories: List[Dict[str, Any]],
    sensei_preferences: Dict[str, List[str]]
) -> float:
    """
    Score whether retrieved memories align with sensei preferences.

    Args:
        retrieved_memories: Memories retrieved by system
        sensei_preferences: Dict with 'loves' and 'hates' lists

    Returns:
        Quality score (0.0-1.0)
    """
    if not retrieved_memories:
        return 0.0

    loves = sensei_preferences.get('loves', [])
    hates = sensei_preferences.get('hates', [])

    score = 0.0
    max_score = len(loves) + len(hates)

    if max_score == 0:
        return 1.0  # No preferences to check

    # Check if memories contain preferred elements
    all_content = ' '.join(m.get('content', '') for m in retrieved_memories)

    for love in loves:
        if keyword_match(love, all_content):
            score += 1.0

    # Penalize if memories contain disliked elements
    for hate in hates:
        if keyword_match(hate, all_content):
            score -= 0.5  # Half penalty

    return max(0.0, min(1.0, score / max_score))


class LearningCurveTracker:
    """Track learning progress across multiple checkpoints"""

    def __init__(self):
        self.checkpoints = []  # List of (visit_num, accuracy) tuples
        self.metrics_history = []  # Detailed metrics at each checkpoint

    def record_checkpoint(
        self,
        visit_num: int,
        accuracy: float,
        metrics: Dict[str, Any] = None
    ):
        """Record accuracy at a checkpoint"""
        self.checkpoints.append((visit_num, accuracy))
        if metrics:
            self.metrics_history.append({
                'visit_num': visit_num,
                'accuracy': accuracy,
                **metrics
            })

    def get_learning_rate(self) -> float:
        """Calculate average learning rate (improvement per visit)"""
        if len(self.checkpoints) < 2:
            return 0.0

        first_visit, first_acc = self.checkpoints[0]
        last_visit, last_acc = self.checkpoints[-1]

        visit_diff = last_visit - first_visit
        if visit_diff == 0:
            return 0.0

        return (last_acc - first_acc) / visit_diff

    def get_total_improvement(self) -> float:
        """Get total improvement from baseline to final"""
        if len(self.checkpoints) < 2:
            return 0.0

        baseline_acc = self.checkpoints[0][1]
        final_acc = self.checkpoints[-1][1]

        return final_acc - baseline_acc

    def is_learning(self, min_improvement: float = 0.30) -> bool:
        """Check if system shows learning (>30% improvement)"""
        return self.get_total_improvement() >= min_improvement

    def has_regression(self) -> bool:
        """Check if accuracy ever decreased"""
        for i in range(len(self.checkpoints) - 1):
            if self.checkpoints[i+1][1] < self.checkpoints[i][1]:
                return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of learning curve"""
        if not self.checkpoints:
            return {}

        return {
            'baseline_accuracy': self.checkpoints[0][1],
            'final_accuracy': self.checkpoints[-1][1],
            'total_improvement': self.get_total_improvement(),
            'learning_rate': self.get_learning_rate(),
            'is_learning': self.is_learning(),
            'has_regression': self.has_regression(),
            'num_checkpoints': len(self.checkpoints),
            'checkpoints': self.checkpoints
        }
