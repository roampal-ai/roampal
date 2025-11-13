"""
LongMemEval - OFFICIAL Test (Per-Question Isolation)
Each question gets its OWN dialogue history (like the official benchmark)
"""
import asyncio
import json
import httpx
import sys
from datetime import datetime
from typing import List, Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'ui-implementation/src-tauri/backend'))
from modules.memory.unified_memory_system import UnifiedMemorySystem


async def load_longmemeval_dataset() -> List[Dict]:
    """Load LongMemEval oracle dataset"""
    with open("longmemeval_data/longmemeval_oracle.json", 'r', encoding='utf-8') as f:
        return json.load(f)


async def ingest_question_dialogues(memory: UnifiedMemorySystem, question_data: Dict) -> int:
    """Ingest dialogue history for ONE question only"""
    haystack_sessions = question_data.get('haystack_sessions', [])
    total_turns = 0

    for session_idx, session in enumerate(haystack_sessions):
        session_id = f"session_{session_idx}"

        for msg in session:
            text = f"{msg['role']}: {msg['content']}"

            await memory.store(
                text=text,
                collection="history",
                metadata={
                    "session_id": session_id,
                    "role": msg['role'],
                    "score": 0.7,
                    "uses": 0
                }
            )
            total_turns += 1

    return total_turns


async def answer_question(memory: UnifiedMemorySystem, question: str) -> tuple[str, list, list]:
    """Answer question using memory"""
    results = await memory.search(question, collections=None, limit=5)

    context_items = []
    for r in results:
        if isinstance(r, dict):
            context_items.append(r.get('content') or r.get('text', ''))

    context = "\n".join(context_items)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Based on conversation history, answer the question concisely.

Conversation history:
{context}

Question: {question}

Answer (brief and factual, or "I don't know" if unsure):""",
                "stream": False
            }
        )

        if response.status_code == 200:
            answer = response.json().get('response', '').strip()
            return answer, context_items, results

        return "Error", context_items, []


async def judge_answer(question: str, predicted: str, ground_truth: str) -> tuple[float, str]:
    """LLM-as-a-Judge"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Evaluate if predicted matches ground truth.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Does predicted convey same facts as ground truth?
Answer ONLY: CORRECT or INCORRECT""",
                "stream": False
            }
        )

        if response.status_code == 200:
            judgment = response.json().get('response', '').strip().upper()
            score = 1.0 if judgment == "CORRECT" else 0.0
            return score, judgment

        return 0.0, "ERROR"


async def run_longmemeval_official(num_questions: int = 500):
    """Run LongMemEval the OFFICIAL way (per-question isolation)"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"longmemeval_OFFICIAL_{timestamp}.log"

    print("="*80)
    print("LONGMEMEVAL - OFFICIAL TEST (PER-QUESTION ISOLATION)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing: {num_questions} questions")
    print(f"Mode: Each question gets its OWN isolated dialogue history")
    print(f"System: UnifiedMemorySystem (5-tier + KG + ChromaDB)")
    print(f"Baseline: GPT-4o (64%), EmergenceMem (82.40%)")
    print("="*80)

    # Load dataset
    dataset = await load_longmemeval_dataset()
    test_data = dataset[:num_questions]

    total_correct = 0
    total_questions = 0

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("="*80 + "\n")
        log_file.write("LONGMEMEVAL - OFFICIAL TEST (PER-QUESTION ISOLATION)\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Testing: {num_questions} questions\n")
        log_file.write(f"Mode: Each question isolated with its own dialogue\n")
        log_file.write("="*80 + "\n\n")

        for q_idx, question_data in enumerate(test_data, 1):
            question_text = question_data.get('question', '')
            ground_truth = question_data.get('answer', '')
            question_type = question_data.get('question_type', 'unknown')

            if not question_text or not ground_truth:
                continue

            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"Q{q_idx}/{num_questions} - {question_type}\n")
            log_file.write(f"{'='*80}\n")
            log_file.write(f"Question: {question_text}\n")
            log_file.write(f"Ground Truth: {ground_truth}\n\n")

            # Create FRESH memory for THIS question only
            test_dir = Path(f"./longmemeval_official_data/question_{q_idx}")
            test_dir.mkdir(parents=True, exist_ok=True)

            memory = UnifiedMemorySystem(
                data_dir=str(test_dir),
                use_server=False,
                llm_service=None
            )
            await memory.initialize()

            # Ingest THIS question's dialogue only
            total_turns = await ingest_question_dialogues(memory, question_data)
            log_file.write(f"Ingested: {total_turns} dialogue turns\n\n")

            # Answer from this question's memory
            predicted, context_items, results = await answer_question(memory, question_text)

            log_file.write(f"Retrieved Context ({len(context_items)} items):\n")
            for i, ctx in enumerate(context_items[:3], 1):
                preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
                log_file.write(f"  {i}. {preview}\n")
            log_file.write(f"\nRoampal Answer: {predicted}\n")

            # Judge
            score, judgment = await judge_answer(question_text, predicted, ground_truth)
            log_file.write(f"Judge: {judgment} (score: {score})\n")

            if score >= 1.0:
                total_correct += 1
            total_questions += 1

            if q_idx % 50 == 0:
                current_acc = (total_correct / total_questions) * 100
                print(f"  Progress: {q_idx}/{num_questions} | Accuracy: {current_acc:.1f}%")

        # Final results
        overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        log_file.write("\n" + "="*80 + "\n")
        log_file.write("FINAL RESULTS\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})\n\n")
        log_file.write("Comparison:\n")
        log_file.write(f"  Roampal (Official):    {overall_accuracy:.2f}%\n")
        log_file.write(f"  EmergenceMem:          82.40%\n")
        log_file.write(f"  GPT-4o:                64.00%\n")
        log_file.write(f"  ChatGPT:               57.73%\n")

        if overall_accuracy > 64.0:
            log_file.write(f"\n[PASS] ROAMPAL BEATS GPT-4o by {overall_accuracy - 64.0:.2f} points!\n")

        log_file.write(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"\nComparison:")
    print(f"  Roampal (Official):    {overall_accuracy:.2f}%")
    print(f"  EmergenceMem:          82.40%")
    print(f"  GPT-4o:                64.00%")

    if overall_accuracy > 64.0:
        print(f"\n[PASS] ROAMPAL BEATS GPT-4o by {overall_accuracy - 64.0:.2f} points!")

    print(f"\nLog: {log_filename}")
    print("="*80)

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions
    }


if __name__ == "__main__":
    asyncio.run(run_longmemeval_official(num_questions=500))
