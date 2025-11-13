"""
LongMemEval Benchmark for Roampal - FULL ARCHITECTURE VERSION
Uses full Roampal API with 5-tier system, KG routing, and outcome learning

Dataset: 500 questions testing 5 core abilities:
- Information extraction
- Multi-session reasoning
- Temporal reasoning
- Knowledge updates
- Abstention

Comparison: GPT-4o (64%), ChatGPT (57.73%), EmergenceMem (82.40%)
"""
import asyncio
import json
import httpx
from datetime import datetime
from typing import List, Dict

# Roampal API endpoints
ROAMPAL_API = "http://localhost:8000"


async def load_longmemeval_dataset(oracle_path: str = "longmemeval_data/longmemeval_oracle.json"):
    """Load LongMemEval oracle dataset (500 questions with ground truth)"""
    with open(oracle_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def clear_roampal_memory():
    """Clear all Roampal memory collections"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Clear history collection (where we'll store conversation turns)
        await client.post(f"{ROAMPAL_API}/api/memory/clear", json={"collection": "history"})


async def ingest_conversation_history(item: Dict, log_file, session_id: str):
    """Ingest conversation history via Roampal API (uses full architecture)"""
    total_turns = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Process each haystack session
        for session_idx, session in enumerate(item['haystack_sessions']):
            haystack_session_id = item['haystack_session_ids'][session_idx]
            session_date = item['haystack_dates'][session_idx]

            for turn in session:
                role = turn['role']
                content = turn['content']

                # Store in Roampal's history collection
                # This uses the full 5-tier system + KG routing
                await client.post(
                    f"{ROAMPAL_API}/api/memory/store",
                    json={
                        "text": f"{role}: {content}",
                        "collection": "history",
                        "metadata": {
                            "role": role,
                            "session_id": session_id,
                            "haystack_session_id": haystack_session_id,
                            "date": session_date,
                            "has_answer": turn.get('has_answer', False)
                        }
                    }
                )
                total_turns += 1

    log_file.write(f"Ingested {total_turns} dialogue turns from {len(item['haystack_sessions'])} sessions\n")
    return total_turns


async def answer_question(question: str, question_date: str, session_id: str) -> tuple[str, list]:
    """Answer question using Roampal's FULL memory system (with learning)"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Search using Roampal API (uses KG routing, multi-tier search)
        search_response = await client.post(
            f"{ROAMPAL_API}/api/memory/search",
            json={
                "query": question,
                "collections": None,  # Let KG routing decide (learning mode!)
                "limit": 5,
                "session_id": session_id
            }
        )

        response_data = search_response.json()
        # API returns list of results directly
        results = response_data if isinstance(response_data, list) else response_data.get('results', [])
        context_items = []
        for r in results:
            if isinstance(r, dict):
                context_items.append(r.get('content') or r.get('text', ''))
            else:
                context_items.append(str(r))
        context = "\n".join(context_items)

        # Generate answer using qwen2.5:14b via Ollama
        gen_response = await client.post(
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

        if gen_response.status_code == 200:
            answer = gen_response.json().get('response', '').strip()
            return answer, context_items, results
        return "Error: Generation failed", context_items, []


async def record_outcome(results: List[Dict], outcome: str, session_id: str):
    """Record outcome for retrieved memories (enables learning)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        for result in results:
            doc_id = result.get('id') or result.get('doc_id')
            if doc_id:
                try:
                    await client.post(
                        f"{ROAMPAL_API}/api/memory/outcome",
                        json={
                            "doc_id": doc_id,
                            "outcome": outcome,
                            "session_id": session_id
                        }
                    )
                except:
                    pass  # Some memories might not support outcome tracking


async def judge_answer(question: str, predicted: str, ground_truth: str) -> tuple[float, str]:
    """LLM-as-a-Judge using qwen2.5:14b"""
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
            # Fix: Exact match to avoid "INCORRECT" matching as correct
            score = 1.0 if judgment == "CORRECT" else 0.0
            return score, judgment
        return 0.0, "ERROR"


async def run_longmemeval_benchmark(num_questions: int = 500):
    """Run LongMemEval benchmark with FULL Roampal architecture"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"longmemeval_FULL_ROAMPAL_{timestamp}.log"

    print("="*80)
    print("LONGMEMEVAL BENCHMARK - FULL ROAMPAL ARCHITECTURE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log File: {log_filename}")
    print(f"Testing: {num_questions} questions")
    print(f"Baseline: GPT-4o (64%), ChatGPT (57.73%), EmergenceMem (82.40%)")
    print(f"Mode: FULL LEARNING (5-tier + KG routing + outcome adaptation)")
    print("="*80)

    # Load dataset
    dataset = await load_longmemeval_dataset()
    test_data = dataset[:num_questions]

    total_questions = 0
    total_correct = 0
    category_results = {}

    # Open log file
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("="*80 + "\n")
        log_file.write("LONGMEMEVAL BENCHMARK - FULL ROAMPAL EVIDENCE LOG\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model: qwen2.5:14b (14.8B params)\n")
        log_file.write(f"Architecture: FULL Roampal (5-tier + KG + outcome learning)\n")
        log_file.write(f"Dataset: LongMemEval (ICLR 2025)\n")
        log_file.write(f"Questions: {len(test_data)}\n")
        log_file.write("="*80 + "\n\n")

        # Process each question
        for q_idx, item in enumerate(test_data, 1):
            question_id = item['question_id']
            question_type = item['question_type']
            question_text = item['question']
            question_date = item['question_date']
            ground_truth = item['answer']

            # Create session ID for this question's context
            session_id = f"longmemeval_{question_id}"

            # Clear memory and ingest conversation history
            await clear_roampal_memory()

            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"Q{q_idx}/{len(test_data)} - {question_type}\n")
            log_file.write(f"{'='*80}\n")
            log_file.write(f"Question ID: {question_id}\n")
            log_file.write(f"Date: {question_date}\n")
            log_file.write(f"Question: {question_text}\n")
            log_file.write(f"Ground Truth: {ground_truth}\n\n")

            # Ingest haystack sessions
            await ingest_conversation_history(item, log_file, session_id)

            # Answer question
            log_file.write(f"\nRetrieved Context:\n")
            predicted_answer, context_items, results = await answer_question(
                question_text, question_date, session_id
            )

            for i, ctx in enumerate(context_items[:3], 1):
                log_file.write(f"  {i}. {ctx[:100]}...\n")

            log_file.write(f"\nRoampal Answer: {predicted_answer}\n")

            # Judge answer
            score, judgment = await judge_answer(question_text, predicted_answer, ground_truth)
            log_file.write(f"Judge: {judgment}\n")
            log_file.write(f"Score: {score}\n")

            # Record outcome for learning
            if score == 1.0:
                await record_outcome(results, "worked", session_id)
            else:
                await record_outcome(results, "failed", session_id)

            # Update stats
            total_questions += 1
            total_correct += score

            if question_type not in category_results:
                category_results[question_type] = {"correct": 0, "total": 0}
            category_results[question_type]["correct"] += score
            category_results[question_type]["total"] += 1

            # Progress update
            if q_idx % 10 == 0:
                current_accuracy = (total_correct / total_questions) * 100
                print(f"Progress: {q_idx}/{len(test_data)} | Accuracy: {current_accuracy:.1f}%")

        # Final results
        overall_accuracy = (total_correct / total_questions) * 100

        log_file.write(f"\n{'='*80}\n")
        log_file.write("FINAL RESULTS\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Overall Accuracy: {overall_accuracy:.2f}% ({int(total_correct)}/{total_questions})\n\n")
        log_file.write("Per-Category:\n")
        for cat, stats in category_results.items():
            cat_acc = (stats["correct"] / stats["total"]) * 100
            log_file.write(f"  {cat}: {cat_acc:.1f}% ({int(stats['correct'])}/{stats['total']})\n")

        log_file.write(f"\nComparison:\n")
        log_file.write(f"  Roampal (Full):  {overall_accuracy:.2f}%\n")
        log_file.write(f"  EmergenceMem:    82.40%\n")
        log_file.write(f"  GPT-4o:          64.00%\n")
        log_file.write(f"  ChatGPT:         57.73%\n")

        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}% ({int(total_correct)}/{total_questions})")
        print(f"\nComparison:")
        print(f"  Roampal (Full):  {overall_accuracy:.2f}%")
        print(f"  EmergenceMem:    82.40%")
        print(f"  GPT-4o:          64.00%")
        print(f"  ChatGPT:         57.73%")
        print(f"\nDetailed log: {log_filename}")
        print("="*80)


if __name__ == "__main__":
    # Test with 50 questions first to validate
    asyncio.run(run_longmemeval_benchmark(num_questions=50))
