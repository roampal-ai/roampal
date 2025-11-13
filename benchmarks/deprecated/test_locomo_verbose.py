"""
LOCOMO Benchmark - VERBOSE MODE
Shows questions, predicted answers, ground truth, and judgment
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

async def answer_question_via_api(question: str, session_id: str) -> tuple[str, list]:
    """Answer question via Roampal API - returns (answer, context)"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Search for context
        search_response = await client.post(
            "http://localhost:8000/api/memory/search",
            json={
                "query": question,
                "collections": ["history"],
                "limit": 5,
                "session_id": session_id
            }
        )

        if search_response.status_code != 200:
            return "Error: Search failed", []

        results = search_response.json()
        context_items = []
        context_text = []

        for r in results[:5]:
            content = r.get("content", r.get("text", ""))
            context_items.append(content)
            context_text.append(content)

        context = "\n".join(context_text)

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
            answer = answer_response.json().get('response', '').strip()
            return answer, context_items
        return "Error: Generation failed", context_items

async def judge_answer(question: str, predicted: str, ground_truth: str) -> tuple[float, str]:
    """LLM-as-a-Judge - returns (score, judgment)"""
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
            score = 1.0 if "CORRECT" in judgment else 0.0
            return score, judgment
        return 0.0, "ERROR"

async def run_verbose_test(num_conversations: int = 1, questions_per_conv: int = 10):
    """Run LOCOMO test with verbose output"""

    print("="*80)
    print("LOCOMO BENCHMARK - VERBOSE MODE")
    print("="*80)
    print(f"Testing {num_conversations} conversation(s), {questions_per_conv} questions each")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load dataset
    dataset = await load_locomo_dataset()

    # Create session
    session_id = "locomo_verbose_test"

    total_questions = 0
    total_correct = 0

    # Process conversations
    for conv_idx in range(min(num_conversations, len(dataset))):
        conversation = dataset[conv_idx]
        conv_id = conv_idx + 1

        print(f"\n{'='*80}")
        print(f"CONVERSATION {conv_id}")
        print(f"{'='*80}")

        # Ingest
        await ingest_conversation_via_api(conversation, conv_id, session_id)

        # Test questions
        questions = conversation['qa'][:questions_per_conv]

        for q_idx, qa in enumerate(questions, 1):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')
            category = qa.get('category', 'unknown')

            if not question or not ground_truth:
                continue

            print(f"\n{'-'*80}")
            print(f"Question {q_idx}/{questions_per_conv} (Category: {category})")
            print(f"{'-'*80}")
            print(f"Q: {question}")
            print(f"\nGround Truth: {ground_truth}")

            # Get answer with context
            predicted, context_items = await answer_question_via_api(question, session_id)

            print(f"\nRetrieved Context ({len(context_items)} items):")
            for i, ctx in enumerate(context_items[:3], 1):
                preview = ctx[:150] + "..." if len(ctx) > 150 else ctx
                print(f"  {i}. {preview}")

            print(f"\nPredicted Answer: {predicted}")

            # Judge
            score, judgment = await judge_answer(question, predicted, str(ground_truth))

            status = "✓ CORRECT" if score >= 1.0 else "✗ INCORRECT"
            print(f"\nJudgment: {judgment} → {status}")

            if score >= 1.0:
                total_correct += 1

            total_questions += 1

        # Conversation summary
        conv_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        print(f"\n{'='*80}")
        print(f"Conversation {conv_id} Summary: {conv_accuracy:.1f}% ({total_correct}/{total_questions})")
        print(f"{'='*80}")

    # Final results
    overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    # Test first conversation, first 10 questions
    asyncio.run(run_verbose_test(num_conversations=1, questions_per_conv=10))
