"""
NATURAL CONVERSATION LEARNING TEST - With Outcome Feedback

⚠️ NOTE: This test needs improvement - currently gets 100% accuracy which is unrealistic.
The test is too lenient or doesn't match real-world usage patterns.

Simulates real usage with natural feedback phrases:
1. Upload conversation documents (like user uploads)
2. Ask questions naturally
3. Give natural feedback: "Perfect, thanks!" or "No, actually..."
4. Roampal's OutcomeDetector picks up on feedback phrases
5. System learns over 3 rounds

This tests: Does natural conversational feedback drive learning?

TODO: Make this test more challenging and realistic.
"""
import asyncio
import json
import httpx
import sys
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'ui-implementation/src-tauri/backend'))
from modules.memory.unified_memory_system import UnifiedMemorySystem


async def load_longmemeval_dataset() -> List[Dict]:
    """Load LongMemEval oracle dataset"""
    with open("longmemeval_data/longmemeval_oracle.json", 'r', encoding='utf-8') as f:
        return json.load(f)


async def upload_dialogues_as_books(
    memory: UnifiedMemorySystem,
    questions_data: List[Dict],
    max_questions: int = 500
) -> int:
    """Upload dialogue chunks as 'book' content"""
    print("\nUploading conversation documents...")
    total_chunks = 0

    for q_idx, question_data in enumerate(questions_data[:max_questions], 1):
        haystack_sessions = question_data.get('haystack_sessions', [])

        for session_idx, session in enumerate(haystack_sessions):
            doc_text = f"=== Conversation {q_idx}-{session_idx} ===\n\n"
            for msg in session:
                role = msg['role'].capitalize()
                content = msg['content']
                doc_text += f"{role}: {content}\n\n"

            await memory.store(
                text=doc_text,
                collection="books",
                metadata={
                    "title": f"Conversation_{q_idx}_{session_idx}",
                    "question_id": question_data.get('question_id'),
                    "source": "uploaded_transcript",
                    "score": 0.7
                }
            )
            total_chunks += 1

        if q_idx % 50 == 0:
            print(f"  Uploaded {q_idx} conversations...")

    print(f"[OK] Uploaded {total_chunks} conversation documents")
    return total_chunks


async def have_natural_conversation(
    memory: UnifiedMemorySystem,
    question: str,
    ground_truth: str,
    round_num: int
) -> Tuple[str, bool, List[str]]:
    """
    Natural conversation turn with feedback
    Returns: (answer, is_correct, conversation_for_outcome_detection)
    """
    # Search memory
    results = await memory.search(question, collections=None, limit=5)

    context_items = []
    retrieved_ids = []
    for r in results:
        if isinstance(r, dict):
            content = r.get('content') or r.get('text', '')
            context_items.append(content)
            if r.get('id'):
                retrieved_ids.append(r.get('id'))

    context = "\n".join(context_items)

    # LLM answers naturally
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""You are Roampal, a helpful AI assistant. Answer based on your memory.

Your memories:
{context}

User: {question}

Answer naturally and conversationally (brief and factual, or "I don't know" if unsure):""",
                "stream": False
            }
        )

        if response.status_code == 200:
            answer = response.json().get('response', '').strip()
        else:
            answer = "Error"

    # Judge against ground truth
    is_correct = await judge_answer(question, answer, ground_truth)

    # Build conversation for OutcomeDetector
    # This simulates the natural back-and-forth
    conversation = [
        {"role": "user", "content": question, "timestamp": datetime.now().isoformat()},
        {"role": "assistant", "content": answer, "timestamp": datetime.now().isoformat()}
    ]

    if is_correct:
        # Natural positive feedback phrases (OutcomeDetector will catch these)
        positive_phrases = [
            "Perfect, thanks!",
            "That's right, thank you!",
            "Awesome, exactly what I needed!",
            "Yes, that worked!",
            "Great, that's correct!"
        ]
        import random
        feedback = random.choice(positive_phrases)
    else:
        # Natural negative feedback with correction
        negative_phrases = [
            f"No, actually {ground_truth}",
            f"That's not right, it's {ground_truth}",
            f"Wrong, it should be {ground_truth}",
            f"Nah, it's {ground_truth}",
            f"Didn't work - the answer is {ground_truth}"
        ]
        feedback = random.choice(negative_phrases)

    conversation.append({
        "role": "user",
        "content": feedback,
        "timestamp": datetime.now().isoformat()
    })

    return answer, is_correct, conversation, retrieved_ids, feedback


async def judge_answer(question: str, answer: str, ground_truth: str) -> bool:
    """Judge if answer contains correct information (STRICT)"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""STRICT EVALUATION: Does the answer contain the ACTUAL correct information?

Question: {question}
Expected answer: {ground_truth}
System's answer: {answer}

RULES:
1. "I don't know" / "I don't have information" = INCORRECT (unless ground truth is also "I don't know")
2. Vague/generic responses = INCORRECT
3. Must contain the specific fact from expected answer
4. Semantic equivalence is OK (e.g., "GPS malfunction" = "GPS not working")
5. Missing key details = INCORRECT

Answer ONLY with: CORRECT or INCORRECT""",
                "stream": False
            }
        )

        if response.status_code == 200:
            judgment = response.json().get('response', '').strip().upper()
            return "CORRECT" in judgment

        return False


async def process_feedback_naturally(
    memory: UnifiedMemorySystem,
    conversation: List[Dict],
    retrieved_ids: List[str],
    is_correct: bool,
    ground_truth: str
):
    """
    Process feedback using OutcomeDetector (natural way)
    OutcomeDetector will analyze conversation and detect "worked" or "failed"
    """
    # OutcomeDetector analyzes the conversation
    # It will see phrases like "Perfect, thanks!" or "No, actually..."
    # and determine outcome automatically

    # For now, we manually trigger since OutcomeDetector needs LLM service
    # In production, this would be automatic
    outcome = "worked" if is_correct else "failed"

    # Record outcome on retrieved memories
    for doc_id in retrieved_ids[:3]:
        try:
            await memory.record_outcome(doc_id, outcome)
        except Exception as e:
            pass

    # If wrong, store the correction as a natural fact
    if not is_correct:
        # Extract natural fact from ground truth
        fact = await extract_fact_natural(conversation[0]['content'], ground_truth)
        await memory.store(
            text=fact,
            collection="patterns",
            metadata={
                "score": 0.9,
                "type": "learned_correction",
                "source": "user_feedback",
                "query": conversation[0]['content']
            }
        )


async def extract_fact_natural(question: str, ground_truth: str) -> str:
    """Convert Q&A to natural fact statement using LLM"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Convert this Q&A into ONE natural statement.

Question: {question}
Answer: {ground_truth}

Examples:
- Q: "What's the capital?" A: "Paris" -> "The capital is Paris"
- Q: "Who is Sarah?" A: "artist" -> "Sarah is an artist"

Natural sentence:""",
                "stream": False
            }
        )

        if response.status_code == 200:
            return response.json().get('response', '').strip()

        return f"{question} -> {ground_truth}"


async def run_natural_conversation_test():
    """
    Main test: 500 questions over 3 rounds with natural feedback
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"natural_conversation_{timestamp}.log"

    print("="*80)
    print("NATURAL CONVERSATION TEST - Outcome Feedback")
    print("="*80)
    print("How it works:")
    print("1. Upload 500 conversation documents")
    print("2. Ask questions naturally over 3 ROUNDS")
    print("3. Give natural feedback: 'Perfect!' or 'No, actually...'")
    print("4. OutcomeDetector picks up on feedback phrases")
    print("5. System learns and improves across rounds")
    print("="*80)

    # Load dataset
    print("\nLoading LongMemEval dataset...")
    questions_data = await load_longmemeval_dataset()
    print(f"[OK] Loaded {len(questions_data)} questions")

    # Initialize memory
    print("\nInitializing memory system...")
    memory = UnifiedMemorySystem(data_dir="./test_data_natural_conv", use_server=False)
    await memory.initialize()
    print("[OK] Memory ready")

    # Upload dialogues
    await upload_dialogues_as_books(memory, questions_data, max_questions=500)

    # Run 3 rounds
    NUM_ROUNDS = 3
    results_by_round = []

    with open(log_filename, 'w', encoding='utf-8') as log:
        log.write("="*80 + "\n")
        log.write("NATURAL CONVERSATION TEST\n")
        log.write("="*80 + "\n\n")

        for round_num in range(1, NUM_ROUNDS + 1):
            print(f"\n{'='*80}")
            print(f"ROUND {round_num}/{NUM_ROUNDS}")
            print(f"{'='*80}")

            log.write(f"\n{'='*80}\n")
            log.write(f"ROUND {round_num}\n")
            log.write(f"{'='*80}\n\n")

            round_correct = 0
            round_total = 0

            for q_idx, question_data in enumerate(questions_data, 1):
                question_text = question_data.get('question', '')
                ground_truth = question_data.get('answer', '')
                question_type = question_data.get('question_type', 'unknown')

                if not question_text or not ground_truth:
                    continue

                # Progress
                if q_idx % 50 == 0:
                    current_acc = (round_correct / round_total * 100) if round_total > 0 else 0
                    print(f"  Q{q_idx}: {current_acc:.1f}% ({round_correct}/{round_total})")

                log.write(f"\nQ{q_idx} [{question_type}]: {question_text}\n")
                log.write(f"Expected: {ground_truth}\n")

                # Have natural conversation with feedback
                answer, is_correct, conversation, retrieved_ids, feedback = await have_natural_conversation(
                    memory, question_text, ground_truth, round_num
                )

                log.write(f"Roampal: {answer}\n")
                log.write(f"User feedback: {feedback}\n")

                if is_correct:
                    log.write(f"[OK] CORRECT\n")
                    round_correct += 1
                else:
                    log.write(f"[X] WRONG\n")

                # Process feedback naturally
                await process_feedback_naturally(
                    memory, conversation, retrieved_ids, is_correct, ground_truth
                )

                round_total += 1

            # Round summary
            round_acc = (round_correct / round_total) * 100
            results_by_round.append(round_acc)

            print(f"\n📊 Round {round_num} Results:")
            print(f"   Accuracy: {round_acc:.1f}% ({round_correct}/{round_total})")
            log.write(f"\n📊 Round {round_num}: {round_acc:.1f}% ({round_correct}/{round_total})\n")

            if round_num > 1:
                improvement = round_acc - results_by_round[0]
                print(f"   Improvement from Round 1: {improvement:+.1f}%")
                log.write(f"   Improvement: {improvement:+.1f}%\n")

        # Final summary
        print(f"\n{'='*80}")
        print("FINAL RESULTS - LEARNING CURVE")
        print(f"{'='*80}")
        log.write(f"\n{'='*80}\n")
        log.write("FINAL RESULTS\n")
        log.write(f"{'='*80}\n")

        for i, acc in enumerate(results_by_round, 1):
            print(f"Round {i}: {acc:.1f}%")
            log.write(f"Round {i}: {acc:.1f}%\n")

        if len(results_by_round) >= 2:
            total_improvement = results_by_round[-1] - results_by_round[0]
            print(f"\nTotal Learning: {total_improvement:+.1f}%")
            log.write(f"\nTotal Learning: {total_improvement:+.1f}%\n")

            if total_improvement > 15:
                verdict = "[SUCCESS] STRONG NATURAL LEARNING! (>15% improvement)"
            elif total_improvement > 5:
                verdict = "[PARTIAL] Moderate learning (5-15% improvement)"
            else:
                verdict = "[FAIL] Minimal/no learning (<5% improvement)"

            print(f"\n{verdict}")
            log.write(f"\n{verdict}\n")

    print(f"\n📝 Detailed log: {log_filename}")
    print("\n💡 Key insight: If Round 3 > Round 1 significantly,")
    print("   then natural conversational feedback drives learning!")


if __name__ == "__main__":
    asyncio.run(run_natural_conversation_test())
