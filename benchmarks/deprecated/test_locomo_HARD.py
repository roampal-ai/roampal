"""
LOCOMO-HARD: Testing Learning Systems with Noisy, Contradictory Data

Standard LOCOMO: 100% with simple vector search (too easy)
LOCOMO-HARD: Inject semantic noise to test learning systems
"""
import asyncio
import json
import httpx
import random
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path


class SimpleMemoryStore:
    """Basic vector search - no learning"""
    def __init__(self):
        self.memories = []  # List[(text, embedding, metadata)]

    async def store(self, text: str, metadata: dict = None):
        embedding = await self.get_embedding(text)
        self.memories.append((text, embedding, metadata or {}))

    async def search(self, query: str, limit: int = 5) -> List[str]:
        query_emb = await self.get_embedding(query)
        # Cosine similarity
        results = []
        for text, emb, meta in self.memories:
            similarity = sum(a * b for a, b in zip(query_emb, emb))
            results.append((similarity, text))
        results.sort(reverse=True)
        return [text for _, text in results[:limit]]

    async def get_embedding(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}
            )
            return response.json()['embedding']


def generate_semantic_noise(real_turn: Dict, conversation_id: int, noise_id: int) -> Dict:
    """Generate semantically similar but factually wrong conversation turn (FAST - no async)"""

    # Name swapping pairs
    name_swaps = [
        ("John", "Mike"), ("Maria", "Sarah"), ("Caroline", "Lisa"),
        ("Melanie", "Emma"), ("Rob", "Tom"), ("Alex", "Chris")
    ]

    # Activity swaps
    activity_swaps = [
        ("yoga", "pilates"), ("pottery", "painting"), ("coding", "writing"),
        ("gym", "swimming"), ("hiking", "biking"), ("cooking", "baking")
    ]

    original_text = real_turn['text']
    speaker = real_turn['speaker']

    # Apply swaps deterministically based on noise_id
    noise_text = original_text
    noise_speaker = speaker

    # Swap names (deterministic based on noise_id)
    for i, (orig, swap) in enumerate(name_swaps):
        if (noise_id + i) % 2 == 0:
            noise_text = noise_text.replace(orig, f"__TEMP_{swap}__")
            noise_speaker = noise_speaker.replace(orig, f"__TEMP_{swap}__")
    for orig, swap in name_swaps:
        noise_text = noise_text.replace(f"__TEMP_{swap}__", swap)
        noise_speaker = noise_speaker.replace(f"__TEMP_{swap}__", swap)

    # Swap activities (different pattern)
    for i, (orig, swap) in enumerate(activity_swaps):
        if (noise_id * 2 + i) % 3 == 0:
            noise_text = noise_text.replace(orig, swap)

    return {
        'speaker': noise_speaker,
        'text': noise_text,
        'dia_id': f"NOISE_{conversation_id}_{noise_id}_{real_turn['dia_id']}",
        'is_noise': True
    }


async def inject_noise_into_conversation(conversation: Dict, noise_multiplier: int = 2) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (real_turns, noise_turns)
    noise_multiplier: 2 = 2× fake data, 4 = 4× fake data
    """
    conv_data = conversation['conversation']

    # Collect all real turns
    real_turns = []
    sessions = [key for key in conv_data.keys()
                if key.startswith('session_') and not key.endswith('_date_time')]
    sessions = sorted(sessions, key=lambda x: int(x.split('_')[1]))

    for session_key in sessions:
        session_data = conv_data[session_key]
        real_turns.extend(session_data)

    # Generate noise
    noise_turns = []
    conv_id = conversation.get('conversation_id', 0)

    for turn in real_turns:
        # Generate multiple noise versions of each turn
        for noise_idx in range(noise_multiplier):
            noise_turn = generate_semantic_noise(turn, conv_id, noise_idx)  # Not async anymore
            noise_turns.append(noise_turn)

    return real_turns, noise_turns


async def load_locomo_dataset(file_path: str = "locomo/data/locomo10.json") -> List[Dict]:
    """Load LOCOMO dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def answer_question(memory: SimpleMemoryStore, question: str) -> str:
    """Answer question using memory"""
    context_items = await memory.search(question, limit=5)
    context = "\n".join(context_items)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
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

        if response.status_code == 200:
            return response.json().get('response', '').strip()
        return "Error: Generation failed"


async def judge_answer(question: str, predicted: str, ground_truth: str) -> Tuple[float, str]:
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
            score = 1.0 if judgment == "CORRECT" else 0.0
            return score, judgment
        return 0.0, "ERROR"


async def run_locomo_hard_benchmark(noise_multiplier: int = 2):
    """
    Run LOCOMO-HARD with noise injection
    noise_multiplier: 2 = 2× noise (67% noise), 4 = 4× noise (80% noise)
    """

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"locomo_HARD_{noise_multiplier}x_noise_{timestamp}.log"

    print("="*80)
    print(f"LOCOMO-HARD BENCHMARK - {noise_multiplier}× NOISE INJECTION")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log File: {log_filename}")
    print(f"Noise Ratio: {noise_multiplier}× (Real:Noise = 1:{noise_multiplier})")
    print(f"Expected: SimpleMemoryStore struggles with semantic noise")
    print("="*80)

    # Load dataset
    dataset = await load_locomo_dataset()

    total_questions = 0
    total_correct = 0
    conversation_results = []

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("="*80 + "\n")
        log_file.write(f"LOCOMO-HARD BENCHMARK - {noise_multiplier}× NOISE\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Noise Multiplier: {noise_multiplier}× (1 real : {noise_multiplier} fake)\n")
        log_file.write(f"Model: qwen2.5:14b\n")
        log_file.write(f"Embeddings: nomic-embed-text\n")
        log_file.write(f"Memory System: SimpleMemoryStore (baseline - no learning)\n")
        log_file.write("="*80 + "\n\n")

        # Test first 2 conversations
        for conv_idx, conversation in enumerate(dataset[:2]):
            conv_id = conv_idx + 1

            print(f"\n{'='*80}")
            print(f"CONVERSATION {conv_id}/2 (TEST MODE)")
            print(f"{'='*80}")

            # Generate noisy dataset
            real_turns, noise_turns = await inject_noise_into_conversation(conversation, noise_multiplier)

            total_turns = len(real_turns) + len(noise_turns)
            noise_ratio = len(noise_turns) / total_turns * 100

            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"CONVERSATION {conv_id} - INGESTION\n")
            log_file.write(f"{'='*80}\n")
            log_file.write(f"Real turns: {len(real_turns)}\n")
            log_file.write(f"Noise turns: {len(noise_turns)}\n")
            log_file.write(f"Total: {total_turns} ({noise_ratio:.1f}% noise)\n\n")

            print(f"[Conv {conv_id}] Real: {len(real_turns)}, Noise: {len(noise_turns)} ({noise_ratio:.1f}% noise)")

            # Create fresh memory
            memory = SimpleMemoryStore()

            # Ingest ALL data (real + noise mixed together)
            all_turns = real_turns + noise_turns
            random.shuffle(all_turns)  # Mix them up

            for turn in all_turns:
                text = f"{turn['speaker']}: {turn['text']}"
                await memory.store(text, {'is_noise': turn.get('is_noise', False)})

            print(f"[Conv {conv_id}] Ingested {total_turns} turns (mixed real + noise)")

            # Get questions
            all_questions = conversation['qa']
            valid_questions = [q for q in all_questions if q.get('category', 0) in [1, 2, 3, 4]]

            log_file.write(f"Testing {len(valid_questions)} questions\n\n")

            conv_correct = 0
            conv_total = 0

            for q_idx, qa in enumerate(valid_questions, 1):
                question = qa.get('question', '')
                ground_truth = qa.get('answer', '')

                if not question or not ground_truth:
                    continue

                # Answer question
                predicted = await answer_question(memory, question)

                # Judge
                score, judgment = await judge_answer(question, predicted, str(ground_truth))

                result = "CORRECT" if score >= 1.0 else "INCORRECT"

                log_file.write(f"Q{q_idx}: {question}\n")
                log_file.write(f"  Ground Truth: {ground_truth}\n")
                log_file.write(f"  Predicted: {predicted}\n")
                log_file.write(f"  Result: {result}\n\n")

                if score >= 1.0:
                    conv_correct += 1
                    total_correct += 1

                conv_total += 1
                total_questions += 1

                # Progress
                if q_idx % 20 == 0:
                    current_acc = (conv_correct / conv_total) * 100
                    print(f"  Progress: {q_idx}/{len(valid_questions)} | Accuracy: {current_acc:.1f}%")

            # Conversation results
            conv_accuracy = (conv_correct / conv_total) * 100 if conv_total > 0 else 0
            conversation_results.append({
                'conv_id': conv_id,
                'accuracy': conv_accuracy,
                'correct': conv_correct,
                'total': conv_total
            })

            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"CONVERSATION {conv_id} RESULTS\n")
            log_file.write(f"Accuracy: {conv_accuracy:.1f}% ({conv_correct}/{conv_total})\n")
            log_file.write(f"{'='*80}\n\n")

            print(f"[Conv {conv_id}] Accuracy: {conv_accuracy:.1f}% ({conv_correct}/{conv_total})")

        # Final results
        overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        log_file.write("\n" + "="*80 + "\n")
        log_file.write("FINAL RESULTS\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})\n\n")
        log_file.write("Per-Conversation:\n")
        for result in conversation_results:
            log_file.write(f"  Conv {result['conv_id']}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})\n")

        log_file.write(f"\nComparison:\n")
        log_file.write(f"  SimpleMemoryStore (noisy):  {overall_accuracy:.2f}%\n")
        log_file.write(f"  SimpleMemoryStore (clean):  100.00%\n")
        log_file.write(f"  Degradation: {100.0 - overall_accuracy:.2f} points\n")

        log_file.write(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*80 + "\n")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"\nComparison:")
    print(f"  With {noise_multiplier}× noise: {overall_accuracy:.2f}%")
    print(f"  Without noise:   100.00%")
    print(f"  Degradation:     {100.0 - overall_accuracy:.2f} points")
    print(f"\nDetailed log: {log_filename}")
    print("="*80)

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'noise_multiplier': noise_multiplier,
        'log_file': log_filename
    }


if __name__ == "__main__":
    # Test with 2× noise (67% noise ratio)
    asyncio.run(run_locomo_hard_benchmark(noise_multiplier=2))
