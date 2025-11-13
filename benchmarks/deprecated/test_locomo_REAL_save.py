"""
LOCOMO Benchmark - Save Predictions for Claude Re-Judging

Uses Roampal's UnifiedMemorySystem directly (no API) to generate predictions.
Saves all question/answer pairs to JSON for Claude Sonnet 4.5 to judge.
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Dict

# Roampal imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ui-implementation', 'src-tauri', 'backend'))
from modules.memory.unified_memory_system import UnifiedMemorySystem

# Will use Ollama for answer generation
import httpx


async def load_locomo_dataset(file_path: str = "locomo/data/locomo10.json") -> List[Dict]:
    """Load LOCOMO dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


async def ingest_conversation_to_roampal(memory: UnifiedMemorySystem, conversation: Dict, conv_id: int):
    """Ingest a LOCOMO conversation into Roampal memory system"""
    print(f"\n[Conversation {conv_id}] Ingesting conversation...")

    conv_data = conversation['conversation']
    total_turns = 0

    # Find all sessions
    sessions = []
    for key in conv_data.keys():
        if key.startswith('session_') and not key.endswith('_date_time'):
            sessions.append(key)

    sessions = sorted(sessions, key=lambda x: int(x.split('_')[1]))

    for session_key in sessions:
        session_num = session_key.split('_')[1]
        session_data = conv_data[session_key]
        session_date = conv_data.get(f'{session_key}_date_time', 'unknown')

        # Ingest each turn in the session
        for turn in session_data:
            speaker = turn['speaker']
            text = turn['text']
            dia_id = turn['dia_id']

            # Store in history with rich metadata
            await memory.store(
                text=f"{speaker}: {text}",
                collection="history",
                metadata={
                    "role": "assistant",
                    "speaker": speaker,
                    "dia_id": dia_id,
                    "session": session_num,
                    "date": session_date,
                    "conversation_id": conv_id,
                    "score": 0.5
                }
            )
            total_turns += 1

    print(f"[Conversation {conv_id}] Ingested {total_turns} turns")
    return total_turns


async def answer_question_with_roampal(memory: UnifiedMemorySystem, question: str, conv_id: int) -> str:
    """Answer a LOCOMO question using Roampal's memory system"""
    # Search for relevant memories
    search_results = await memory.search(
        query=question,
        collections=["history"],
        limit=10
    )

    # Build context from search results
    context_parts = []
    for result in search_results:
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        dia_id = metadata.get('dia_id', '')
        context_parts.append(f"[{dia_id}] {content}")

    context = "\n".join(context_parts) if context_parts else "No relevant context found."

    # Generate answer using Ollama
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b",
                    "prompt": f"""Based on the following conversation excerpts, answer the question concisely and accurately.

Conversation Context:
{context}

Question: {question}

Answer (be brief and factual):""",
                    "stream": False
                }
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer
            else:
                return "Error generating answer"

    except Exception as e:
        return f"Error: {str(e)}"


async def run_locomo_save_predictions():
    """Run LOCOMO benchmark and SAVE all predictions (no judging)"""

    print("="*80)
    print("LOCOMO BENCHMARK - SAVING PREDICTIONS FOR CLAUDE RE-JUDGING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load dataset
    print("\nLoading LOCOMO dataset...")
    dataset = await load_locomo_dataset()
    print(f"Loaded {len(dataset)} conversations")

    # Initialize Roampal memory (embedded mode - no server needed)
    memory = UnifiedMemorySystem(data_dir="./.benchmarks/locomo_test", use_server=False)

    # Store all predictions
    all_predictions = []

    # Process each conversation
    for conv_idx, conversation in enumerate(dataset):
        conv_id = conv_idx + 1
        print(f"\n{'='*80}")
        print(f"PROCESSING CONVERSATION {conv_id}/{len(dataset)}")
        print(f"{'='*80}")

        # Ingest conversation into memory
        await ingest_conversation_to_roampal(memory, conversation, conv_id)

        # Get questions for this conversation
        questions = conversation['qa']
        print(f"[Conversation {conv_id}] Testing {len(questions)} questions...")

        answered_count = 0

        for q_idx, qa in enumerate(questions, 1):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')

            # Skip if missing fields (Category 5 - empty/adversarial)
            if not question or not ground_truth:
                continue

            # Get Roampal's answer
            predicted = await answer_question_with_roampal(memory, question, conv_id)

            # Save prediction (NO JUDGING!)
            all_predictions.append({
                'conv_id': conv_id,
                'question_idx': q_idx,
                'question': question,
                'ground_truth': str(ground_truth),
                'predicted': predicted
            })

            answered_count += 1

            # Print progress every 10 questions
            if answered_count % 10 == 0:
                print(f"  Progress: {answered_count} questions answered")

        print(f"[Conversation {conv_id}] Answered {answered_count} questions")

    # Save predictions to JSON
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
    print("\nNext step: Claude will judge these predictions")

    return output_file


if __name__ == "__main__":
    # Check Ollama is running
    import httpx
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code != 200:
            print("ERROR: Ollama not running. Start it with: ollama serve")
            sys.exit(1)
    except:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        sys.exit(1)

    # Run
    asyncio.run(run_locomo_save_predictions())
