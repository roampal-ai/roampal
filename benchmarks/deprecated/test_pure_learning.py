"""
PURE LEARNING TEST - Zero Knowledge Bootstrap

Tests if Roampal can learn from scratch with ZERO prior knowledge.

Flow:
1. NO document upload - completely empty memory
2. Ask questions (system will get most wrong initially)
3. Provide corrections: "No, actually it's X"
4. System stores corrections and learns over 5 rounds
5. Track improvement: Should go from ~0% to higher accuracy

This tests: Can Roampal bootstrap knowledge purely from conversational feedback?
"""
import asyncio
import json
import httpx
import sys
import random
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'ui-implementation/src-tauri/backend'))
from modules.memory.unified_memory_system import UnifiedMemorySystem


async def load_longmemeval_dataset() -> List[Dict]:
    """Load LongMemEval oracle dataset"""
    with open("longmemeval_data/longmemeval_oracle.json", 'r', encoding='utf-8') as f:
        return json.load(f)


async def ask_question_cold(
    memory: UnifiedMemorySystem,
    question: str,
    ground_truth: str
) -> Tuple[str, bool, List[str]]:
    """
    Ask question with no prior knowledge, then provide correction
    Returns: (answer, is_correct, retrieved_ids)
    """
    # Search memory (will be mostly empty initially)
    results = await memory.search(question, collections=None, limit=5)

    context_items = []
    retrieved_ids = []
    for r in results:
        if isinstance(r, dict):
            content = r.get('content') or r.get('text', '')
            context_items.append(content)
            if r.get('id'):
                retrieved_ids.append(r.get('id'))

    context = "\n".join(context_items) if context_items else "No relevant information found."

    # LLM answers (will guess initially)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""You are Roampal. Answer the question based on your memory.

Memory:
{context}

Question: {question}

Answer briefly and directly:""",
                "stream": False
            }
        )

        if response.status_code == 200:
            answer = response.json().get('response', '').strip()
        else:
            answer = "I don't know."

    # Judge correctness
    judge_response = await client.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": f"""Compare these answers:

Question: {question}
Ground truth: {ground_truth}
System answer: {answer}

Are they semantically equivalent? Answer ONLY 'yes' or 'no':""",
            "stream": False
        }
    )

    is_correct = False
    if judge_response.status_code == 200:
        judge_result = judge_response.json().get('response', '').strip().lower()
        is_correct = 'yes' in judge_result

    return answer, is_correct, retrieved_ids


async def provide_correction(
    memory: UnifiedMemorySystem,
    question: str,
    ground_truth: str,
    retrieved_ids: List[str],
    is_correct: bool
):
    """
    Provide natural feedback and store correction
    """
    if is_correct:
        # Reinforce correct memories
        positive_phrases = [
            "Perfect, thanks!",
            "That's right, thank you!",
            "Awesome, exactly what I needed!"
        ]
        feedback = random.choice(positive_phrases)

        # Boost memories that helped
        for mem_id in retrieved_ids[:3]:
            await memory.record_outcome(mem_id, "worked")

    else:
        # Provide correction
        negative_phrases = [
            f"No, actually {ground_truth}",
            f"That's not right, it's {ground_truth}",
            f"Wrong, the answer is {ground_truth}"
        ]
        feedback = random.choice(negative_phrases)

        # Store correction as new knowledge
        correction_text = f"Question: {question}\nAnswer: {ground_truth}"
        await memory.store(
            text=correction_text,
            collection="patterns",
            metadata={
                "score": 0.9,
                "source": "user_correction",
                "question": question,
                "answer": ground_truth
            }
        )

        # Punish memories that misled (if any)
        for mem_id in retrieved_ids[:3]:
            await memory.record_outcome(mem_id, "failed")


async def run_pure_learning_test(num_questions: int = 100, num_rounds: int = 5):
    """
    Run pure learning test with zero prior knowledge
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"pure_learning_{timestamp}.log"

    print("="*80)
    print("PURE LEARNING TEST - Zero Knowledge Bootstrap")
    print("="*80)
    print(f"Questions: {num_questions}")
    print(f"Rounds: {num_rounds}")
    print("No documents uploaded - learning purely from corrections")
    print("="*80)

    # Load dataset
    print("\nLoading questions...")
    questions_data = await load_longmemeval_dataset()
    questions_data = questions_data[:num_questions]
    print(f"[OK] Loaded {len(questions_data)} questions")

    # Initialize memory (EMPTY - no uploads!)
    print("\nInitializing empty memory system...")
    memory = UnifiedMemorySystem(data_dir="./test_data_pure_learning", use_server=False)
    await memory.initialize()
    print("[OK] Memory ready (empty)")

    # Run multiple rounds
    all_results = []

    with open(log_filename, 'w', encoding='utf-8') as log:
        log.write("PURE LEARNING TEST - Zero Knowledge Bootstrap\n")
        log.write(f"Questions: {num_questions}, Rounds: {num_rounds}\n")
        log.write("="*80 + "\n\n")

        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*80}")
            print(f"ROUND {round_num}/{num_rounds}")
            print(f"{'='*80}")

            log.write(f"\n{'='*80}\n")
            log.write(f"ROUND {round_num}/{num_rounds}\n")
            log.write(f"{'='*80}\n\n")

            round_correct = 0

            for q_idx, question_data in enumerate(questions_data, 1):
                question_text = question_data.get('question', '')
                ground_truth = question_data.get('answer', '')

                # Ask question
                answer, is_correct, retrieved_ids = await ask_question_cold(
                    memory, question_text, ground_truth
                )

                if is_correct:
                    round_correct += 1

                # Log interaction
                log.write(f"Q{q_idx}: {question_text}\n")
                log.write(f"Expected: {ground_truth}\n")
                log.write(f"Got: {answer}\n")
                log.write(f"Status: {'[OK] CORRECT' if is_correct else '[X] WRONG'}\n")

                # Provide correction/reinforcement
                await provide_correction(
                    memory, question_text, ground_truth, retrieved_ids, is_correct
                )

                # Progress updates
                if q_idx % 10 == 0:
                    current_accuracy = (round_correct / q_idx) * 100
                    print(f"  Q{q_idx}: {current_accuracy:.1f}% correct so far")
                    log.write(f"  Progress: {current_accuracy:.1f}% ({round_correct}/{q_idx})\n")

                log.write("\n")

            # Round summary
            round_accuracy = (round_correct / len(questions_data)) * 100
            all_results.append({
                'round': round_num,
                'correct': round_correct,
                'total': len(questions_data),
                'accuracy': round_accuracy
            })

            print(f"\nRound {round_num} Results: {round_accuracy:.1f}% ({round_correct}/{len(questions_data)})")
            log.write(f"\nRound {round_num} FINAL: {round_accuracy:.1f}% ({round_correct}/{len(questions_data)})\n")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS - Learning Curve")
    print(f"{'='*80}")

    for result in all_results:
        print(f"Round {result['round']}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")

    # Calculate improvement
    if len(all_results) >= 2:
        improvement = all_results[-1]['accuracy'] - all_results[0]['accuracy']
        print(f"\nImprovement: {improvement:+.1f}% (Round 1 -> Round {num_rounds})")

        if improvement > 10:
            print("[SUCCESS] System learned from pure conversational feedback!")
        elif improvement > 0:
            print("[PARTIAL] Some learning occurred")
        else:
            print("[FAILED] No learning improvement")

    print(f"\nFull log: {log_filename}")


if __name__ == "__main__":
    asyncio.run(run_pure_learning_test(num_questions=500, num_rounds=5))
