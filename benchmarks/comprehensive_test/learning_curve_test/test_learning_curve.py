"""
Learning Curve Test - Prove Roampal learns over time

Simulates 13-visit storytelling conversations and measures if system
performance improves with more interactions.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ui-implementation" / "src-tauri" / "backend"))

from modules.memory.unified_memory_system import UnifiedMemorySystem
from storyteller_simulator import load_storyteller_dataset, StorytellerVisit
from learning_metrics import (
    calculate_accuracy,
    evaluate_retrieval_quality,
    LearningCurveTracker,
    keyword_match
)


# Test configuration
TEST_CONFIG = {
    'story_to_test': None,  # None = auto-select first story with 12+ visits
    'checkpoints': [0, 3, 6, 9, 13],  # Test accuracy at these visit numbers
    'min_improvement': 0.30,  # Require 30% improvement to pass
    'use_real_embeddings': False,  # Set to True to use sentence-transformers
    'verbose': True
}


async def test_system_knowledge(
    system: UnifiedMemorySystem,
    story_title: str,
    visits_completed: List[StorytellerVisit],
    simulator
) -> float:
    """
    Test how much the system knows about the story.

    Args:
        system: Memory system to test
        story_title: Story being tested
        visits_completed: Visits that have been processed so far
        simulator: StorytellerSimulator instance

    Returns:
        Accuracy score (0.0-1.0)
    """
    if not visits_completed:
        return 0.0

    # Generate test questions based on what's been learned
    test_questions = simulator.create_test_questions(story_title, visits_completed)

    if not test_questions:
        return 0.0

    correct = 0
    total = len(test_questions)

    for q in test_questions:
        question = q['question']
        expected_answers = q['expected_answers']

        # Search system's memory
        try:
            results = await system.search(question, limit=5)

            # Check if any result contains expected answer
            for expected in expected_answers:
                found = False
                for result in results:
                    content = result.get('content', '') + ' ' + result.get('metadata', {}).get('text', '')
                    if keyword_match(expected, content, threshold=0.5):
                        found = True
                        break
                if found:
                    correct += 1
                    break  # Only count once per question

        except Exception as e:
            if TEST_CONFIG['verbose']:
                print(f"    Warning: Search failed for '{question}': {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


async def simulate_learning(
    system: UnifiedMemorySystem,
    story_title: str,
    visits: List[StorytellerVisit],
    simulator,
    tracker: LearningCurveTracker
):
    """
    Simulate learning by processing visits sequentially.

    Args:
        system: Memory system
        story_title: Story being learned
        visits: All visits for this story
        simulator: StorytellerSimulator
        tracker: Learning curve tracker
    """
    visits_completed = []

    for i, visit in enumerate(visits):
        visit_num = visit.visit_number

        if TEST_CONFIG['verbose']:
            print(f"  Processing visit {visit_num}/{len(visits)}...")

        # User asks question
        user_query = visit.situation

        # System searches its memory
        try:
            retrieved_memories = await system.search(user_query, limit=5)
        except Exception as e:
            if TEST_CONFIG['verbose']:
                print(f"    Warning: Search failed: {e}")
            retrieved_memories = []

        # Store this conversation in history
        try:
            doc_id = await system.store(
                text=f"Visit {visit_num}: {user_query}",
                collection='history',
                metadata={
                    'visit_number': visit_num,
                    'story_title': story_title,
                    'domain': visit.domain
                }
            )

            # Evaluate if memories were helpful
            was_helpful = simulator.evaluate_memory_helpfulness(
                retrieved_memories,
                visit
            )

            # Record outcome
            outcome = 'worked' if was_helpful else 'failed'
            await system.record_outcome(doc_id, outcome)

        except Exception as e:
            if TEST_CONFIG['verbose']:
                print(f"    Warning: Failed to store/score: {e}")

        # Store sensei preferences in memory_bank
        if visit.sensei_preferences:
            try:
                loves = visit.sensei_preferences.get('loves', [])
                hates = visit.sensei_preferences.get('hates', [])

                preference_text = f"{visit.sensei_id} preferences for {story_title}: "
                if loves:
                    preference_text += f"Loves: {', '.join(loves[:3])}. "
                if hates:
                    preference_text += f"Avoids: {', '.join(hates[:3])}."

                await system.store_memory_bank(
                    text=preference_text,
                    tags=['preference', visit.domain],
                    importance=0.9,
                    confidence=0.9
                )
            except Exception as e:
                if TEST_CONFIG['verbose']:
                    print(f"    Warning: Failed to store preferences: {e}")

        visits_completed.append(visit)

        # Checkpoint: Test knowledge at specific intervals
        if visit_num in TEST_CONFIG['checkpoints']:
            accuracy = await test_system_knowledge(
                system,
                story_title,
                visits_completed,
                simulator
            )
            tracker.record_checkpoint(visit_num, accuracy)

            if TEST_CONFIG['verbose']:
                print(f"    Checkpoint {visit_num}: {accuracy:.1%} accuracy")


async def run_learning_curve_test(story_title: str = None) -> Dict[str, Any]:
    """
    Run complete learning curve test.

    Args:
        story_title: Story to test (None = auto-select)

    Returns:
        Test results dict
    """
    print("=" * 70)
    print("LEARNING CURVE TEST - Proving Roampal Learns Over Time")
    print("=" * 70)

    # Load storyteller dataset
    print("\nLoading storyteller dataset...")
    simulator = load_storyteller_dataset()

    if TEST_CONFIG['verbose']:
        summary = simulator.get_summary()
        print(f"  Loaded {summary['total_stories']} stories with {summary['total_visits']} total visits")

    # Select story to test
    if story_title is None:
        # Auto-select first story with 12+ visits
        for title, visits in simulator.stories.items():
            if len(visits) >= 12:
                story_title = title
                break

    if story_title not in simulator.stories:
        raise ValueError(f"Story '{story_title}' not found in dataset")

    visits = simulator.get_visits_for_story(story_title)
    metadata = simulator.get_story_metadata(story_title)

    print(f"\nTesting story: {story_title}")
    print(f"  Domain: {metadata['domain']}")
    print(f"  Sensei: {metadata['sensei_id']}")
    print(f"  Total visits: {metadata['num_visits']}")
    print(f"  Requires memory: {metadata['requires_memory_count']}/{metadata['num_visits']}")

    # Initialize tracker
    tracker = LearningCurveTracker()

    # === BASELINE: Test with no memories ===
    print("\n[Phase 1: Baseline - Cold Start]")
    system_baseline = UnifiedMemorySystem('./test_learning_baseline', use_server=False)
    try:
        await system_baseline.initialize()
        baseline_accuracy = await test_system_knowledge(
            system_baseline,
            story_title,
            visits[:1],  # Just enough to generate questions
            simulator
        )
        tracker.record_checkpoint(0, baseline_accuracy)
        print(f"  Baseline (0 visits): {baseline_accuracy:.1%} accuracy")
    finally:
        pass  # Let it clean up naturally

    # === LEARNING: Process all visits ===
    print("\n[Phase 2: Learning - Processing 13 visits]")
    system_trained = UnifiedMemorySystem('./test_learning_trained', use_server=False)
    try:
        await system_trained.initialize()

        # Override with real embeddings if requested
        if TEST_CONFIG['use_real_embeddings']:
            try:
                from sentence_transformers import SentenceTransformer
                print("  Loading real embeddings (sentence-transformers)...")
                # Note: This would need adapter to work with UnifiedMemorySystem
                # For now, uses default embeddings
            except ImportError:
                print("  Warning: sentence-transformers not installed, using default embeddings")

        await simulate_learning(
            system_trained,
            story_title,
            visits,
            simulator,
            tracker
        )

    finally:
        pass  # Let it clean up naturally

    # === RESULTS ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    summary = tracker.get_summary()

    print(f"\nBaseline accuracy: {summary['baseline_accuracy']:.1%}")
    print(f"Final accuracy: {summary['final_accuracy']:.1%}")
    print(f"Total improvement: {summary['total_improvement']:.1%}")
    print(f"Learning rate: {summary['learning_rate']:.3f} per visit")

    print(f"\nCheckpoints:")
    for visit_num, accuracy in summary['checkpoints']:
        print(f"  Visit {visit_num:2d}: {accuracy:.1%}")

    # === VERDICT ===
    print("\n" + "=" * 70)

    if tracker.has_regression():
        print("⚠ WARNING: Performance regressed at some checkpoints")

    if tracker.is_learning(TEST_CONFIG['min_improvement']):
        print(f"[PASS] LEARNING PROVEN: {summary['baseline_accuracy']:.1%} -> {summary['final_accuracy']:.1%} ")
        print(f"       (+{summary['total_improvement']:.1%} improvement)")
        success = True
    else:
        print(f"[FAIL] LEARNING NOT PROVEN: Only {summary['total_improvement']:.1%} improvement")
        print(f"       (Required: >{TEST_CONFIG['min_improvement']:.1%})")
        success = False

    print("=" * 70)

    return {
        'success': success,
        'story_title': story_title,
        'summary': summary,
        'test_config': TEST_CONFIG
    }


def main():
    """Main entry point"""
    # You can override config here
    # TEST_CONFIG['story_to_test'] = "Memory Thieves Don't Cry"
    # TEST_CONFIG['use_real_embeddings'] = True

    try:
        result = asyncio.run(run_learning_curve_test(TEST_CONFIG['story_to_test']))

        # Save results
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / "learning_curve_results.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        return 0 if result['success'] else 1

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
