"""
LOCOMO Benchmark via Roampal Chat API - Uses running backend
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

async def ingest_conversation_via_chat(conversation: Dict, conv_id: int, session_id: str):
    """Ingest conversation by chatting with Roampal (it auto-stores in history)"""
    print(f"\n[Conversation {conv_id}] Ingesting conversation...")

    conv_data = conversation['conversation']
    total_turns = 0

    sessions = [key for key in conv_data.keys()
                if key.startswith('session_') and not key.endswith('_date_time')]
    sessions = sorted(sessions, key=lambda x: int(x.split('_')[1]))

    async with httpx.AsyncClient(timeout=60.0) as client:
        for session_key in sessions:
            session_data = conv_data[session_key]
            for turn in session_data:
                # Send each turn as a chat message - Roampal will store it
                text = f"{turn['speaker']}: {turn['text']}"

                await client.post(
                    "http://localhost:8000/api/agent/chat",
                    json={
                        "message": text,
                        "session_id": session_id,
                        "model": "qwen2.5:0.5b",  # Fast model for ingestion
                        "max_tokens": 10,  # Don't need long responses
                        "temperature": 0.0
                    }
                )
                total_turns += 1

                # Progress indicator
                if total_turns % 50 == 0:
                    print(f"  Ingested {total_turns} turns...")

    print(f"[Conversation {conv_id}] Ingested {total_turns} turns")
    return total_turns

async def answer_question_via_chat(question: str, session_id: str) -> str:
    """Answer question via Roampal chat"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/api/agent/chat",
            json={
                "message": question,
                "session_id": session_id,
                "model": "qwen2.5:14b",
                "max_tokens": 100,
                "temperature": 0.0
            }
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('response', '').strip()
        return f"Error: {response.status_code}"

async def run_locomo_save():
    """Run LOCOMO and save predictions"""

    print("="*80)
    print("LOCOMO BENCHMARK - VIA CHAT API (SAVES PREDICTIONS)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    dataset = await load_locomo_dataset()
    print(f"\nLoaded {len(dataset)} conversations")

    session_id = "locomo_chat_test"
    all_predictions = []

    for conv_idx, conversation in enumerate(dataset):
        conv_id = conv_idx + 1
        print(f"\n{'='*80}")
        print(f"PROCESSING CONVERSATION {conv_id}/{len(dataset)}")
        print(f"{'='*80}")

        # Ingest conversation
        await ingest_conversation_via_chat(conversation, conv_id, session_id)

        # Test questions
        questions = conversation['qa']
        print(f"[Conversation {conv_id}] Testing {len(questions)} questions...")

        answered = 0
        for q_idx, qa in enumerate(questions, 1):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')

            if not question or not ground_truth:
                continue

            predicted = await answer_question_via_chat(question, session_id)

            all_predictions.append({
                'conv_id': conv_id,
                'question_idx': q_idx,
                'question': question,
                'ground_truth': str(ground_truth),
                'predicted': predicted
            })

            answered += 1
            if answered % 10 == 0:
                print(f"  Progress: {answered} questions answered")

        print(f"[Conversation {conv_id}] Answered {answered} questions")

    # Save
    output_file = f"locomo_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("PREDICTIONS SAVED")
    print("="*80)
    print(f"Total: {len(all_predictions)}")
    print(f"File: {output_file}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_locomo_save())
