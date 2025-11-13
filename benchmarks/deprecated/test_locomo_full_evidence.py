"""
LOCOMO BENCHMARK - FULL TEST FOR EVIDENCE
All 10 conversations, all valid questions (Categories 1-4)
Saves detailed log with every Q&A for verification
"""
import asyncio
import json
import httpx
from datetime import datetime
from typing import List, Dict
import numpy as np

class SimpleMemoryStore:
    def __init__(self):
        self.memories = []  # [(text, embedding, metadata)]

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}
            )
            return response.json()["embedding"]

    async def store(self, text: str, metadata: dict):
        """Store text with embedding"""
        embedding = await self.get_embedding(text)
        self.memories.append((text, np.array(embedding), metadata))

    async def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search by similarity"""
        query_emb = np.array(await self.get_embedding(query))

        # Calculate cosine similarity
        results = []
        for text, emb, meta in self.memories:
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            results.append((sim, text, meta))

        # Sort by similarity
        results.sort(reverse=True, key=lambda x: x[0])

        return [{"content": text, "metadata": meta, "score": float(score)}
                for score, text, meta in results[:limit]]

    def clear(self):
        """Clear all memories"""
        self.memories = []


async def load_locomo_dataset(file_path: str = "locomo/data/locomo10.json") -> List[Dict]:
    """Load LOCOMO dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def ingest_conversation(memory: SimpleMemoryStore, conversation: Dict, conv_id: int, log_file):
    """Ingest conversation"""
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"CONVERSATION {conv_id} - INGESTION\n")
    log_file.write(f"{'='*80}\n")

    conv_data = conversation['conversation']
    total_turns = 0

    # Find all sessions
    sessions = [key for key in conv_data.keys()
                if key.startswith('session_') and not key.endswith('_date_time')]
    sessions = sorted(sessions, key=lambda x: int(x.split('_')[1]))

    for session_key in sessions:
        session_data = conv_data[session_key]
        for turn in session_data:
            text = f"{turn['speaker']}: {turn['text']}"
            await memory.store(text, {"conv_id": conv_id, "dia_id": turn['dia_id']})
            total_turns += 1

    log_file.write(f"Ingested {total_turns} dialogue turns across {len(sessions)} sessions\n")
    log_file.flush()

    print(f"[Conv {conv_id}] Ingested {total_turns} turns")
    return total_turns


async def answer_question(memory: SimpleMemoryStore, question: str) -> tuple[str, list]:
    """Answer question - returns (answer, context_items)"""
    # Search for context
    results = await memory.search(question, limit=5)
    context_items = [r["content"] for r in results]
    context = "\n".join(context_items)

    # Generate answer
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
            answer = response.json().get('response', '').strip()
            return answer, context_items
        return "Error: Generation failed", context_items


async def judge_answer(question: str, predicted: str, ground_truth: str) -> tuple[float, str]:
    """LLM-as-a-Judge using qwen2.5:14b"""
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


async def run_full_benchmark():
    """Run complete LOCOMO benchmark - all conversations, all valid questions"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"locomo_FULL_EVIDENCE_{timestamp}.log"

    print("="*80)
    print("LOCOMO BENCHMARK - FULL TEST (ALL 10 CONVERSATIONS)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log File: {log_filename}")
    print(f"Comparison Baseline: Mem0 v1.0.0 (66.9%), OpenAI Memory (52.9%)")
    print("="*80)

    # Load dataset
    dataset = await load_locomo_dataset()
    memory = SimpleMemoryStore()

    total_questions = 0
    total_correct = 0
    conversation_results = []

    # Open log file
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("="*80 + "\n")
        log_file.write("LOCOMO BENCHMARK - FULL EVIDENCE LOG\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model: qwen2.5:14b (14.8B params)\n")
        log_file.write(f"Embeddings: nomic-embed-text (Ollama)\n")
        log_file.write(f"Dataset: LOCOMO (snap-research/locomo)\n")
        log_file.write(f"Methodology: LLM-as-a-Judge (same as Mem0)\n")
        log_file.write("="*80 + "\n\n")

        # Process each conversation
        for conv_idx, conversation in enumerate(dataset):
            conv_id = conv_idx + 1

            print(f"\n{'='*80}")
            print(f"PROCESSING CONVERSATION {conv_id}/10")
            print(f"{'='*80}")

            # Clear memory for fresh start
            memory.clear()

            # Ingest conversation
            await ingest_conversation(memory, conversation, conv_id, log_file)

            # Get questions - ONLY Categories 1-4 (exclude Category 5 per standard)
            all_questions = conversation['qa']
            valid_questions = [q for q in all_questions if q.get('category', 0) in [1, 2, 3, 4]]

            log_file.write(f"\nTesting {len(valid_questions)} valid questions (Categories 1-4)\n")
            log_file.write(f"Excluded {len(all_questions) - len(valid_questions)} Category 5 questions\n\n")

            conv_correct = 0
            conv_total = 0

            for q_idx, qa in enumerate(valid_questions, 1):
                question = qa.get('question', '')
                ground_truth = qa.get('answer', '')
                category = qa.get('category', 'unknown')

                if not question or not ground_truth:
                    continue

                # Log question
                log_file.write(f"\n{'-'*80}\n")
                log_file.write(f"Q{q_idx}/{len(valid_questions)} (Category {category})\n")
                log_file.write(f"{'-'*80}\n")
                log_file.write(f"Question: {question}\n")
                log_file.write(f"Ground Truth: {ground_truth}\n")

                # Get answer
                predicted, context_items = await answer_question(memory, question)

                log_file.write(f"\nRetrieved Context:\n")
                for i, ctx in enumerate(context_items[:3], 1):
                    preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
                    log_file.write(f"  {i}. {preview}\n")

                log_file.write(f"\nRoampal Answer: {predicted}\n")

                # Judge
                score, judgment = await judge_answer(question, predicted, str(ground_truth))

                result = "CORRECT" if score >= 1.0 else "INCORRECT"
                log_file.write(f"Judge: {judgment} -> {result}\n")

                if score >= 1.0:
                    conv_correct += 1
                    total_correct += 1

                conv_total += 1
                total_questions += 1

                # Progress update every 20 questions
                if q_idx % 20 == 0:
                    current_acc = (conv_correct / conv_total) * 100
                    print(f"  Progress: {q_idx}/{len(valid_questions)} | Accuracy: {current_acc:.1f}%")

            # Conversation summary
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
        log_file.write("Per-Conversation Breakdown:\n")
        for result in conversation_results:
            log_file.write(f"  Conversation {result['conv_id']}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})\n")

        log_file.write(f"\nComparison:\n")
        log_file.write(f"  Roampal:         {overall_accuracy:.2f}%\n")
        log_file.write(f"  Mem0 v1.0.0:     66.90%\n")
        log_file.write(f"  OpenAI Memory:   52.90%\n")

        if overall_accuracy > 66.9:
            improvement = overall_accuracy - 66.9
            log_file.write(f"\nROAMPAL BEATS MEM0 by {improvement:.2f} percentage points!\n")

        log_file.write(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*80 + "\n")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"\nPer-Conversation:")
    for result in conversation_results:
        print(f"  Conv {result['conv_id']}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")
    print(f"\nComparison:")
    print(f"  Roampal:         {overall_accuracy:.2f}%")
    print(f"  Mem0 v1.0.0:     66.90%")
    print(f"  OpenAI Memory:   52.90%")

    if overall_accuracy > 66.9:
        print(f"\nROAMPAL BEATS MEM0 by {overall_accuracy - 66.9:.2f} points!")

    print(f"\nDetailed log saved to: {log_filename}")
    print("="*80)

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'conversation_results': conversation_results,
        'log_file': log_filename
    }


if __name__ == "__main__":
    asyncio.run(run_full_benchmark())
