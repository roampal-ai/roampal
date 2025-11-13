"""
LOCOMO Benchmark - SAVES ALL PREDICTIONS for re-judging
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

async def run_locomo_with_save():
    """Run LOCOMO benchmark and SAVE all predictions"""

    print("="*80)
    print("LOCOMO BENCHMARK - SAVING PREDICTIONS FOR RE-JUDGING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load dataset
    print("\nLoading LOCOMO dataset...")
    dataset = await load_locomo_dataset()
    print(f"Loaded {len(dataset)} conversations")

    # Create session for benchmark
    session_id = "locomo_save_predictions"

    # Store all predictions
    all_predictions = []

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

        for q_idx, qa in enumerate(questions, 1):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')

            if not question or not ground_truth:
                continue

            # Get answer
            predicted = await answer_question_via_api(question, session_id)

            # Save prediction (no judging yet!)
            all_predictions.append({
                'conv_id': conv_id,
                'question_idx': q_idx,
                'question': question,
                'ground_truth': str(ground_truth),
                'predicted': predicted
            })

            # Progress every 10 questions
            if q_idx % 10 == 0:
                print(f"  Progress: {q_idx}/{len(questions)} questions answered")

        print(f"[Conversation {conv_id}] Answered {len([p for p in all_predictions if p['conv_id'] == conv_id])} questions")

    # Save all predictions to JSON
    output_file = f"locomo_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("PREDICTIONS SAVED")
    print("="*80)
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Saved to: {output_file}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    print("Next step: Run judging script to evaluate these predictions")

if __name__ == "__main__":
    asyncio.run(run_locomo_with_save())
