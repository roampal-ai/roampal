"""
LOCOMO Benchmark - Show me the answers so I can judge them myself
Uses simple in-memory store with Ollama embeddings
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


async def load_locomo_dataset(file_path: str = "locomo/data/locomo10.json") -> List[Dict]:
    """Load LOCOMO dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def ingest_conversation(memory: SimpleMemoryStore, conversation: Dict, conv_id: int):
    """Ingest conversation"""
    print(f"\n[Conversation {conv_id}] Ingesting conversation...")

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

    print(f"[Conversation {conv_id}] Ingested {total_turns} turns")
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
            score = 1.0 if "CORRECT" in judgment else 0.0
            return score, judgment
        return 0.0, "ERROR"


async def run_sample_test(num_conversations: int = 1, questions_per_conv: int = 10):
    """Run sample test with visible Q&A"""

    print("="*80)
    print("LOCOMO SAMPLE TEST - JUDGE THE ANSWERS YOURSELF")
    print("="*80)
    print(f"Testing {num_conversations} conversation(s), {questions_per_conv} questions each")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load dataset
    dataset = await load_locomo_dataset()
    memory = SimpleMemoryStore()

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
        await ingest_conversation(memory, conversation, conv_id)

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
            predicted, context_items = await answer_question(memory, question)

            print(f"\nRetrieved Context ({len(context_items)} items):")
            for i, ctx in enumerate(context_items, 1):
                preview = ctx[:120] + "..." if len(ctx) > 120 else ctx
                print(f"  {i}. {preview}")

            print(f"\nRoampal's Answer: {predicted}")

            # Judge
            score, judgment = await judge_answer(question, predicted, str(ground_truth))

            status = "[CORRECT]" if score >= 1.0 else "[INCORRECT]"
            print(f"\nLLM Judge: {judgment} -> {status}")

            if score >= 1.0:
                total_correct += 1

            total_questions += 1

        # Conversation summary
        conv_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        print(f"\n{'='*80}")
        print(f"Conversation {conv_id} Accuracy: {conv_accuracy:.1f}% ({total_correct}/{total_questions})")
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
    asyncio.run(run_sample_test(num_conversations=1, questions_per_conv=10))
