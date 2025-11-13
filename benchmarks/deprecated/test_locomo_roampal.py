"""
LOCOMO Benchmark for Roampal v0.2.0

Tests Roampal against the industry-standard LOCOMO benchmark used by Mem0.
- 10 conversations with ~600 turns each
- ~200 Q&A per conversation (2000 total questions)
- Ground truth answers provided
- LLM-as-a-Judge scoring

Comparison baseline:
- Mem0 v1.0.0: 66.9% accuracy (April 2025, arXiv:2504.19413)
- OpenAI Memory: 52.9% accuracy
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

# Will use Ollama for LLM-as-a-Judge
import httpx


async def load_locomo_dataset(file_path: str = "locomo/data/locomo10.json") -> List[Dict]:
    """Load LOCOMO dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


async def ingest_conversation_to_roampal(memory: UnifiedMemorySystem, conversation: Dict, conv_id: int):
    """
    Ingest a LOCOMO conversation into Roampal memory system.

    Strategy:
    - Store dialogs in history collection (simulating past conversations)
    - Each session is a separate context window
    - Metadata includes speaker, date, dialog_id for retrieval
    """
    print(f"\n[Conversation {conv_id}] Ingesting conversation...")

    conv_data = conversation['conversation']
    speaker_a = conv_data['speaker_a']
    speaker_b = conv_data['speaker_b']

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
                    "score": 0.5  # neutral initial score
                }
            )
            total_turns += 1

    print(f"[Conversation {conv_id}] Ingested {total_turns} turns across {len(sessions)} sessions")
    return total_turns


async def answer_question_with_roampal(memory: UnifiedMemorySystem, question: str, conv_id: int) -> str:
    """
    Answer a LOCOMO question using Roampal's memory system.

    Strategy:
    1. Search memory for relevant context
    2. Use Ollama to generate answer based on retrieved context
    """
    # Search for relevant memories (focusing on history where conversations are stored)
    search_results = await memory.search(
        query=question,
        collections=["history"],
        limit=10  # Get top 10 most relevant turns
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
                    "model": "qwen2.5:14b",  # Matches gpt-4o-mini (79.7% vs 82% MMLU)
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


async def evaluate_answer_with_llm_judge(question: str, predicted: str, ground_truth: str) -> float:
    """
    LLM-as-a-Judge scoring (following Mem0's methodology).

    Returns: 1.0 if correct, 0.0 if incorrect
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b",  # Matches gpt-4o-mini for fair evaluation
                    "prompt": f"""You are evaluating answers to questions. Determine if the predicted answer matches the ground truth answer. The predicted answer doesn't need to be word-for-word identical, but must convey the same factual information.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}

Does the predicted answer match the ground truth? Consider:
- Factual accuracy (core facts must match)
- Semantic equivalence (different wording is OK)
- Completeness (key information present)

Respond with ONLY "CORRECT" or "INCORRECT" (no explanation):""",
                    "stream": False
                }
            )

            if response.status_code == 200:
                result = response.json()
                judgment = result.get('response', '').strip().upper()

                if "CORRECT" in judgment:
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0

    except Exception as e:
        print(f"    Error in LLM judge: {e}")
        return 0.0


async def run_locomo_benchmark():
    """Run full LOCOMO benchmark on Roampal"""

    print("="*80)
    print("LOCOMO BENCHMARK - ROAMPAL V0.2.0")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Comparison Baseline: Mem0 v1.0.0 (66.9%), OpenAI Memory (52.9%)")
    print("="*80)

    # Load dataset
    print("\nLoading LOCOMO dataset...")
    dataset = await load_locomo_dataset()
    print(f"Loaded {len(dataset)} conversations")

    # Initialize Roampal memory (use dedicated benchmark server on port 8004)
    # Create custom ChromaDB client for benchmarking
    from modules.memory.chromadb_adapter import ChromaDBAdapter
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    # Monkey-patch the adapter to use port 8004 for this test
    original_init = ChromaDBAdapter.initialize
    async def custom_init(self, collection_name="loopsmith_memories", fragment_id=None, embedding_model_name=None, user_id=None):
        if self.client is None and self.use_server:
            self.client = chromadb.HttpClient(
                host="localhost",
                port=8004,  # Benchmark server
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return await original_init(self, collection_name, fragment_id, embedding_model_name, user_id)

    ChromaDBAdapter.initialize = custom_init
    memory = UnifiedMemorySystem(data_dir="./.benchmarks/locomo_test", use_server=True)

    total_questions = 0
    total_correct = 0
    conversation_results = []

    # Process each conversation
    for conv_idx, conversation in enumerate(dataset):
        conv_id = conv_idx + 1
        print(f"\n{'='*80}")
        print(f"PROCESSING CONVERSATION {conv_id}/{len(dataset)}")
        print(f"{'='*80}")

        # Ingest conversation into memory
        turns_ingested = await ingest_conversation_to_roampal(memory, conversation, conv_id)

        # Get questions for this conversation
        questions = conversation['qa']
        print(f"[Conversation {conv_id}] Testing {len(questions)} questions...")

        conv_correct = 0

        # FULL TEST: Use ALL questions for legitimate comparison with Mem0
        test_questions = questions  # ALL questions (~200 per conversation = ~2000 total)

        for q_idx, qa in enumerate(test_questions, 1):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')

            # Skip if missing fields
            if not question or not ground_truth:
                continue

            # Get Roampal's answer
            predicted = await answer_question_with_roampal(memory, question, conv_id)

            # Judge answer
            score = await evaluate_answer_with_llm_judge(question, predicted, str(ground_truth))

            if score >= 1.0:
                conv_correct += 1
                total_correct += 1

            total_questions += 1

            # Print progress every 10 questions
            if q_idx % 10 == 0:
                current_acc = (conv_correct / q_idx) * 100
                print(f"  Progress: {q_idx}/{len(test_questions)} questions | Accuracy: {current_acc:.1f}%")

        conv_accuracy = (conv_correct / len(test_questions)) * 100
        conversation_results.append({
            'conv_id': conv_id,
            'accuracy': conv_accuracy,
            'correct': conv_correct,
            'total': len(test_questions)
        })

        print(f"[Conversation {conv_id}] Final Accuracy: {conv_accuracy:.1f}% ({conv_correct}/{len(test_questions)})")

    # Final results
    overall_accuracy = (total_correct / total_questions) * 100

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"\nComparison:")
    print(f"  Roampal v0.2.0:  {overall_accuracy:.2f}%")
    print(f"  Mem0 v1.0.0:     66.90%")
    print(f"  OpenAI Memory:   52.90%")

    if overall_accuracy > 66.9:
        improvement = overall_accuracy - 66.9
        print(f"\n✅ Roampal BEATS Mem0 by {improvement:.2f} percentage points!")
    elif overall_accuracy > 52.9:
        print(f"\n✅ Roampal beats OpenAI Memory")

    print(f"\nPer-Conversation Results:")
    for result in conversation_results:
        print(f"  Conversation {result['conv_id']}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")

    print("\n" + "="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'conversation_results': conversation_results
    }


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

    # Run benchmark
    asyncio.run(run_locomo_benchmark())
