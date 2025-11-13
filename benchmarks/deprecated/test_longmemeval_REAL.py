"""
LongMemEval Benchmark - Testing REAL UnifiedMemorySystem
115K tokens per conversation - TRUE long-term memory test
"""
import asyncio
import json
import httpx
import sys
from datetime import datetime
from typing import List, Dict
from pathlib import Path

# Import REAL Roampal system
sys.path.insert(0, str(Path(__file__).parent.parent / 'ui-implementation/src-tauri/backend'))
from modules.memory.unified_memory_system import UnifiedMemorySystem


async def load_longmemeval_dataset(file_path: str = "longmemeval_data/longmemeval_s_cleaned.json") -> List[Dict]:
    """Load LongMemEval dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def ingest_dialogues(memory: UnifiedMemorySystem, question_data: Dict, log_file) -> int:
    """Ingest dialogue history into memory"""
    dialogues = question_data.get('dialogue', [])

    log_file.write(f"\nIngesting {len(dialogues)} dialogue turns from {len(set(d.get('session_id', 0) for d in dialogues))} sessions\n")

    total_turns = 0
    for turn in dialogues:
        text = f"{turn['role']}: {turn['content']}"
        session_id = turn.get('session_id', 'session_0')
        timestamp = turn.get('timestamp', '')

        # Store in history collection with metadata
        await memory.store(
            text=text,
            collection="history",
            metadata={
                "session_id": session_id,
                "timestamp": timestamp,
                "role": turn['role'],
                "score": 0.7,
                "uses": 0
            }
        )
        total_turns += 1

    return total_turns


async def answer_question(memory: UnifiedMemorySystem, question: str, question_date: str) -> tuple[str, list, list]:
    """Answer question using KG routing"""
    # Search with KG routing
    results = await memory.search(question, collections=None, limit=5)

    context_items = []
    for r in results:
        if isinstance(r, dict):
            context_items.append(r.get('content') or r.get('text', ''))
        else:
            context_items.append(str(r))

    context = "\n".join(context_items)

    # Generate answer
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Based on the conversation history, answer the question concisely and accurately.

Current date: {question_date}

Conversation history:
{context}

Question: {question}

Answer (brief and factual, or say "I don't know" if unsure):""",
                "stream": False
            }
        )

        if response.status_code == 200:
            answer = response.json().get('response', '').strip()
            return answer, context_items, results
        return "Error: Generation failed", context_items, []


async def judge_answer(question: str, predicted: str, ground_truth: str) -> tuple[float, str]:
    """LLM-as-a-Judge"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Evaluate if the predicted answer matches the ground truth answer.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Does the predicted answer convey the same factual information as the ground truth?
Be precise - only mark CORRECT if the core facts match.

Respond ONLY with "CORRECT" or "INCORRECT":""",
                "stream": False
            }
        )

        if response.status_code == 200:
            judgment = response.json().get('response', '').strip().upper()
            score = 1.0 if judgment == "CORRECT" else 0.0
            return score, judgment
        return 0.0, "ERROR"


async def run_longmemeval_benchmark(num_questions: int = 50):
    """Run LongMemEval with UnifiedMemorySystem"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"longmemeval_REAL_{timestamp}.log"

    # Setup test data directory
    test_dir = Path("./longmemeval_test_data")
    test_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LONGMEMEVAL BENCHMARK - REAL UNIFIED MEMORY SYSTEM")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log File: {log_filename}")
    print(f"Testing: {num_questions} questions")
    print(f"Baseline: GPT-4o (64%), EmergenceMem (82.40%)")
    print(f"System: UnifiedMemorySystem (5-tier + KG routing + ChromaDB embedded)")
    print("="*80)

    # Load dataset
    dataset = await load_longmemeval_dataset()
    test_data = dataset[:num_questions]

    total_correct = 0
    total_questions = 0

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("="*80 + "\n")
        log_file.write("LONGMEMEVAL BENCHMARK - UNIFIED MEMORY SYSTEM\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model: qwen2.5:14b\n")
        log_file.write(f"Embeddings: nomic-embed-text (Ollama)\n")
        log_file.write(f"Memory System: UnifiedMemorySystem (5-tier + KG + ChromaDB embedded)\n")
        log_file.write(f"Dataset: LongMemEval (115K tokens avg per conversation)\n")
        log_file.write("="*80 + "\n\n")

        # Process each question
        for q_idx, question_data in enumerate(test_data, 1):
            question_id = question_data.get('id', f'q{q_idx}')
            question_text = question_data.get('question', '')
            ground_truth = question_data.get('answer', '')
            question_type = question_data.get('type', 'unknown')
            question_date = question_data.get('date', '2023/01/01')

            if not question_text or not ground_truth:
                continue

            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"Q{q_idx}/{num_questions} - {question_type}\n")
            log_file.write(f"{'='*80}\n")
            log_file.write(f"Question ID: {question_id}\n")
            log_file.write(f"Date: {question_date}\n")
            log_file.write(f"Question: {question_text}\n")
            log_file.write(f"Ground Truth: {ground_truth}\n\n")

            # Create fresh memory for this question
            question_dir = test_dir / f"question_{q_idx}"
            question_dir.mkdir(parents=True, exist_ok=True)

            memory = UnifiedMemorySystem(
                data_dir=str(question_dir),
                use_server=False,
                llm_service=None
            )
            await memory.initialize()

            # Ingest dialogues
            turns_ingested = await ingest_dialogues(memory, question_data, log_file)

            # Answer question
            predicted, context_items, results = await answer_question(memory, question_text, question_date)

            log_file.write(f"\nRetrieved Context:\n")
            for i, ctx in enumerate(context_items[:3], 1):
                preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
                log_file.write(f"  {i}. {preview}\n")

            log_file.write(f"\nRoampal Answer: {predicted}\n")

            # Judge
            score, judgment = await judge_answer(question_text, predicted, ground_truth)

            result = "CORRECT" if score >= 1.0 else "INCORRECT"
            log_file.write(f"Judge: {judgment}\n")
            log_file.write(f"Score: {score}\n")

            if score >= 1.0:
                total_correct += 1

            total_questions += 1

            # Progress
            if q_idx % 10 == 0:
                current_acc = (total_correct / total_questions) * 100
                print(f"Progress: {q_idx}/{num_questions} | Accuracy: {current_acc:.1f}%")

        # Final results
        overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        log_file.write("\n" + "="*80 + "\n")
        log_file.write("FINAL RESULTS\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})\n\n")
        log_file.write("Comparison:\n")
        log_file.write(f"  Roampal (Unified):  {overall_accuracy:.2f}%\n")
        log_file.write(f"  EmergenceMem:       82.40%\n")
        log_file.write(f"  GPT-4o:             64.00%\n")
        log_file.write(f"  ChatGPT:            57.73%\n")

        if overall_accuracy > 64.0:
            log_file.write(f"\nROAMPAL BEATS GPT-4o by {overall_accuracy - 64.0:.2f} points!\n")

        log_file.write(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*80 + "\n")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"\nComparison:")
    print(f"  Roampal (Unified):  {overall_accuracy:.2f}%")
    print(f"  EmergenceMem:       82.40%")
    print(f"  GPT-4o:             64.00%")
    print(f"  ChatGPT:            57.73%")

    if overall_accuracy > 64.0:
        print(f"\nROAMPAL BEATS GPT-4o by {overall_accuracy - 64.0:.2f} points!")

    print(f"\nDetailed log: {log_filename}")
    print("="*80)

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'log_file': log_filename
    }


if __name__ == "__main__":
    # Full test - all 500 questions
    asyncio.run(run_longmemeval_benchmark(num_questions=500))
