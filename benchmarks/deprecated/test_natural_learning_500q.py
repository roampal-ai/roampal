"""
NATURAL CONVERSATION LEARNING TEST - 500 Questions

Uses LongMemEval dataset but simulates natural usage:
1. Upload dialogue chunks as "documents" (like user uploaded convos)
2. Ask questions naturally over 3 rounds
3. System learns from corrections naturally (not flashcards)
4. Measure if accuracy improves across rounds

This tests: Can the system learn like a human would in real usage?
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
    questions_data: List[Dict]
) -> int:
    """
    Upload all dialogue chunks as 'book' content
    Simulates: User uploaded conversation transcripts as documents
    """
    print("\n📚 Uploading conversation documents to memory...")
    total_chunks = 0

    for q_idx, question_data in enumerate(questions_data, 1):
        haystack_sessions = question_data.get('haystack_sessions', [])

        for session_idx, session in enumerate(haystack_sessions):
            # Format session as a readable "document"
            doc_text = f"=== Conversation {q_idx}-{session_idx} ===\n\n"

            for msg in session:
                role = msg['role'].capitalize()
                content = msg['content']
                doc_text += f"{role}: {content}\n\n"

            # Store as book content
            await memory.store(
                text=doc_text,
                collection="books",
                metadata={
                    "title": f"Conversation_{q_idx}_{session_idx}",
                    "question_id": question_data.get('question_id'),
                    "source": "uploaded_transcript",
                    "score": 0.7  # Default book score
                }
            )
            total_chunks += 1

        # Progress indicator
        if q_idx % 50 == 0:
            print(f"  Uploaded {q_idx} conversations...")

    print(f"✓ Uploaded {total_chunks} conversation documents")
    return total_chunks


async def ask_question_naturally(
    memory: UnifiedMemorySystem,
    question: str,
    ground_truth: str,
    round_num: int
) -> Tuple[str, bool, List[Dict]]:
    """
    Have a natural conversation turn
    """
    # Search memory naturally
    results = await memory.search(question, collections=None, limit=5)

    # Build context
    context_items = []
    for r in results:
        if isinstance(r, dict):
            content = r.get('content') or r.get('text', '')
            context_items.append({
                'content': content,
                'id': r.get('id'),
                'collection': r.get('collection', 'unknown'),
                'score': r.get('metadata', {}).get('score', 0.5)
            })

    context = "\n".join([c['content'] for c in context_items])

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

    # Judge answer
    is_correct = await judge_answer_flexible(question, answer, ground_truth)

    return answer, is_correct, context_items


async def judge_answer_flexible(question: str, answer: str, ground_truth: str) -> bool:
    """Flexible judging - does answer contain the key facts?"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Does the answer contain the correct information?

Question: {question}
Expected: {ground_truth}
Actual: {answer}

The answer doesn't need to match exactly - it just needs to convey the same facts.
Answer ONLY: CORRECT or INCORRECT""",
                "stream": False
            }
        )

        if response.status_code == 200:
            judgment = response.json().get('response', '').strip().upper()
            return "CORRECT" in judgment

        return False


async def learn_from_mistake_natural(
    memory: UnifiedMemorySystem,
    question: str,
    ground_truth: str,
    context_items: List[Dict],
    round_num: int
):
    """
    Natural learning: Extract fact and store conversationally
    """
    # Extract natural fact using LLM
    fact = await extract_fact_natural(question, ground_truth)

    # Store as learned fact
    await memory.store(
        text=fact,
        collection="patterns",
        metadata={
            "score": 0.9,
            "type": "learned_fact",
            "source": "conversation_correction",
            "round": round_num,
            "query": question
        }
    )

    # Punish wrong memories
    for item in context_items[:3]:
        if item.get('id'):
            try:
                await memory.record_outcome(item['id'], "failed")
            except:
                pass


async def extract_fact_natural(question: str, ground_truth: str) -> str:
    """Convert Q&A to natural fact statement"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Convert this Q&A into a natural statement.

Question: {question}
Answer: {ground_truth}

Write ONE natural sentence that states this fact clearly.
Examples:
- Q: "What's the capital?" A: "Paris" → "The capital is Paris"
- Q: "Who wrote X?" A: "Shakespeare" → "Shakespeare wrote X"

Natural sentence:""",
                "stream": False
            }
        )

        if response.status_code == 200:
            return response.json().get('response', '').strip()

        return f"{question} → {ground_truth}"


async def reinforce_success(memory: UnifiedMemorySystem, context_items: List[Dict]):
    """Boost memories that led to correct answer"""
    for item in context_items[:3]:
        if item.get('id'):
            try:
                await memory.record_outcome(item['id'], "worked")
            except:
                pass


async def run_natural_learning_500q():
    """
    Main test: 500 questions over 3 rounds
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"natural_learning_500q_{timestamp}.log"

    print("="*80)
    print("🎭 NATURAL CONVERSATION LEARNING TEST - 500 Questions")
    print("="*80)
    print("Simulating real-world usage:")
    print("1. Upload 500 conversation documents (like user uploads)")
    print("2. Ask 500 questions naturally over 3 ROUNDS")
    print("3. System learns corrections as facts (not flashcards)")
    print("4. Measure improvement: Round 1 → Round 2 → Round 3")
    print("="*80)

    # Load dataset
    print("\n📂 Loading LongMemEval dataset...")
    questions_data = await load_longmemeval_dataset()
    print(f"✓ Loaded {len(questions_data)} questions")

    # Initialize memory
    print("\n🧠 Initializing memory system...")
    memory = UnifiedMemorySystem(
        chroma_host="localhost",
        chroma_port=8000,
        collection_prefix="natural_500q_test"
    )
    await memory.initialize()
    print("✓ Memory system ready")

    # Upload all dialogues as "books"
    await upload_dialogues_as_books(memory, questions_data)

    # Run 3 rounds
    NUM_ROUNDS = 3
    results_by_round = []

    with open(log_filename, 'w', encoding='utf-8') as log:
        log.write("="*80 + "\n")
        log.write("NATURAL CONVERSATION LEARNING TEST - 500 Questions\n")
        log.write("="*80 + "\n\n")

        for round_num in range(1, NUM_ROUNDS + 1):
            print(f"\n{'='*80}")
            print(f"🔄 ROUND {round_num}/{NUM_ROUNDS}")
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

                # Progress indicator
                if q_idx % 50 == 0:
                    current_acc = (round_correct / round_total * 100) if round_total > 0 else 0
                    print(f"  Q{q_idx}: {current_acc:.1f}% ({round_correct}/{round_total})")

                log.write(f"\nQ{q_idx} [{question_type}]: {question_text}\n")
                log.write(f"Expected: {ground_truth}\n")

                # Ask question naturally
                answer, is_correct, context_items = await ask_question_naturally(
                    memory, question_text, ground_truth, round_num
                )

                log.write(f"Roampal: {answer}\n")

                if is_correct:
                    log.write(f"✓ CORRECT\n")
                    round_correct += 1
                    await reinforce_success(memory, context_items)
                else:
                    log.write(f"✗ WRONG\n")
                    await learn_from_mistake_natural(
                        memory, question_text, ground_truth,
                        context_items, round_num
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
        print("📊 FINAL RESULTS - LEARNING CURVE")
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
                verdict = "✅ STRONG LEARNING DETECTED! (>15% improvement)"
            elif total_improvement > 5:
                verdict = "⚠️  Moderate learning (5-15% improvement)"
            else:
                verdict = "❌ Minimal/no learning (<5% improvement)"

            print(f"\n{verdict}")
            log.write(f"\n{verdict}\n")

    print(f"\n📝 Detailed log: {log_filename}")


if __name__ == "__main__":
    asyncio.run(run_natural_learning_500q())
