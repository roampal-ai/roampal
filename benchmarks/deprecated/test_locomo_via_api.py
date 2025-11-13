"""
LOCOMO Benchmark via Roampal API - Uses RUNNING backend (no model loading!)
"""
import asyncio
import json
import httpx
from datetime import datetime
from typing import List, Dict

async def load_locomo_dataset(file_path: str = "locomo/data/locomo10.json") -> List[Dict]:
    """Load LOCOMO dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def ingest_conversation_via_api(conversation: Dict, conv_id: int, session_id: str):
    """Ingest conversation via Roampal API"""
    print(f"\n[Conversation {conv_id}] Ingesting conversation...")

    conv_data = conversation['conversation']
    total_turns = 0

    # Find all sessions
    sessions = [key for key in conv_data.keys()
                if key.startswith('session_') and not key.endswith('_date_time')]
    sessions = sorted(sessions, key=lambda x: int(x.split('_')[1]))

    async with httpx.AsyncClient(timeout=30.0) as client:
        for session_key in sessions:
            session_data = conv_data[session_key]
            for turn in session_data:
                text = f"{turn['speaker']}: {turn['text']}"

                # Store via API
                await client.post(
                    "http://localhost:8000/api/memory/store",
                    json={
                        "text": text,
                        "collection": "history",
                        "metadata": {
                            "conv_id": conv_id,
                            "dia_id": turn['dia_id'],
                            "session_id": session_id
                        }
                    }
                )
                total_turns += 1

    print(f"[Conversation {conv_id}] Ingested {total_turns} turns")
    return total_turns

async def answer_question_via_api(question: str, session_id: str) -> str:
    """Answer question via Roampal API"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Search for context
        search_response = await client.post(
            "http://localhost:8000/api/memory/search",
            json={
                "query": question,
                "collections": ["history"],
                "limit": 10,
                "session_id": session_id
            }
        )

        if search_response.status_code != 200:
            return "Error: Search failed"

        results = search_response.json()
        context = "\n".join([r.get("content", r.get("text", "")) for r in results[:5]])

        # Generate answer with qwen2.5:14b
        answer_response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Based on the conversation context, answer the question concisely.

Context:
{context}

Question: {question}

Answer (brief and factual):""",
                "stream": False
            }
        )

        if answer_response.status_code == 200:
            return answer_response.json().get('response', '').strip()
        return "Error: Generation failed"

async def judge_answer(question: str, predicted: str, ground_truth: str) -> float:
    """LLM-as-a-Judge"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Evaluate if the predicted answer matches the ground truth.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Does the predicted answer convey the same meaning?
Respond ONLY with "CORRECT" or "INCORRECT":""",
                "stream": False
            }
        )

        if response.status_code == 200:
            judgment = response.json().get('response', '').strip().upper()
            return 1.0 if "CORRECT" in judgment else 0.0
        return 0.0

async def run_locomo_benchmark():
    """Run LOCOMO benchmark via Roampal API"""

    print("="*80)
    print("LOCOMO BENCHMARK - VIA ROAMPAL API (INSTANT START)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Baseline: Mem0 v1.0.0 (66.9%), OpenAI Memory (52.9%)")
    print("="*80)

    # Load dataset
    print("\nLoading LOCOMO dataset...")
    dataset = await load_locomo_dataset()
    print(f"Loaded {len(dataset)} conversations")

    # Create session for benchmark
    session_id = "locomo_benchmark_session"

    total_questions = 0
    total_correct = 0
    conversation_results = []

    # Process each conversation
    for conv_idx, conversation in enumerate(dataset):
        conv_id = conv_idx + 1
        print(f"\n{'='*80}")
        print(f"PROCESSING CONVERSATION {conv_id}/{len(dataset)}")
        print(f"{'='*80}")

        # Ingest conversation
        await ingest_conversation_via_api(conversation, conv_id, session_id)

        # Get questions
        questions = conversation['qa']
        print(f"[Conversation {conv_id}] Testing {len(questions)} questions...")

        conv_correct = 0

        for q_idx, qa in enumerate(questions, 1):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')

            if not question or not ground_truth:
                continue

            # Get answer
            predicted = await answer_question_via_api(question, session_id)

            # Judge answer
            score = await judge_answer(question, predicted, str(ground_truth))

            if score >= 1.0:
                conv_correct += 1
                total_correct += 1

            total_questions += 1

            # Progress every 10 questions
            if q_idx % 10 == 0:
                current_acc = (conv_correct / q_idx) * 100
                print(f"  Progress: {q_idx}/{len(questions)} | Accuracy: {current_acc:.1f}%")

        conv_accuracy = (conv_correct / len(questions)) * 100
        conversation_results.append({
            'conv_id': conv_id,
            'accuracy': conv_accuracy,
            'correct': conv_correct,
            'total': len(questions)
        })

        print(f"[Conversation {conv_id}] Accuracy: {conv_accuracy:.1f}% ({conv_correct}/{len(questions)})")

    # Final results
    overall_accuracy = (total_correct / total_questions) * 100

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"\nComparison:")
    print(f"  Roampal:         {overall_accuracy:.2f}%")
    print(f"  Mem0 v1.0.0:     66.90%")
    print(f"  OpenAI Memory:   52.90%")

    if overall_accuracy > 66.9:
        print(f"\n✅ ROAMPAL BEATS MEM0 by {overall_accuracy - 66.9:.2f} points!")
    elif overall_accuracy > 52.9:
        print(f"\n✅ Roampal beats OpenAI Memory")

    print(f"\nPer-Conversation:")
    for result in conversation_results:
        print(f"  Conv {result['conv_id']}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")

    print("\n" + "="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_locomo_benchmark())
