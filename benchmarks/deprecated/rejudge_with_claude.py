"""
Re-judge LOCOMO predictions using Claude API
This script reads saved predictions and evaluates them
"""
import json
import sys

def load_predictions(filename):
    """Load saved predictions"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python rejudge_with_claude.py <predictions_file.json>")
        sys.exit(1)

    predictions_file = sys.argv[1]

    print("="*80)
    print("LOCOMO RE-JUDGING WITH CLAUDE")
    print("="*80)
    print(f"Loading predictions from: {predictions_file}")

    predictions = load_predictions(predictions_file)

    print(f"Total predictions to judge: {len(predictions)}")
    print()
    print("="*80)
    print("READY TO JUDGE")
    print("="*80)
    print()
    print("Instructions:")
    print("1. I will show you batches of predictions")
    print("2. For each, respond with CORRECT or INCORRECT")
    print("3. I'll track the scores and give you the final accuracy")
    print()

    # Show first 10 as example
    print("First 10 predictions:")
    print("-"*80)
    for i, pred in enumerate(predictions[:10], 1):
        print(f"\n{i}. Question: {pred['question']}")
        print(f"   Ground Truth: {pred['ground_truth']}")
        print(f"   Predicted: {pred['predicted']}")
        print()

    print("="*80)
    print(f"Ready to judge all {len(predictions)} predictions in batches")
    print("="*80)

if __name__ == "__main__":
    main()
