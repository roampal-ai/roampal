"""
NATURAL CONVERSATION LEARNING TEST

Simulates how Roampal would actually be used:
1. User uploads a book/document (stored in books collection)
2. User has natural conversations about the content
3. When wrong, system learns the correction naturally
4. Test if accuracy improves over time through real usage

This mimics: "I'm reading about Sarah, let me chat about her with my AI"
"""
import asyncio
import json
import httpx
import sys
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'ui-implementation/src-tauri/backend'))
from modules.memory.unified_memory_system import UnifiedMemorySystem


# Sample "book" content (like a user uploading a document)
SAMPLE_BOOK = """
Sarah's Life and Interests - A Biography

Sarah Thompson is a 32-year-old artist living in Portland, Oregon.

Chapter 1: Childhood
Sarah grew up in Seattle, where she first discovered her love for painting at age 8.
Her favorite color has always been red, which features prominently in all her artwork.
She attended Roosevelt High School and graduated in 2009.

Chapter 2: Career
After college, Sarah became a professional watercolor artist. She specializes in landscapes
and portraits. Her studio is located in the Pearl District of Portland.
She typically works from 6 AM to 2 PM, as she finds morning light ideal for painting.

Chapter 3: Personal Life
Sarah is married to David Chen, a software engineer. They have two cats named Pixel and Brush.
On weekends, Sarah enjoys hiking in the Columbia River Gorge and trying new restaurants.
Her favorite cuisine is Thai food, particularly from a local spot called Pok Pok.

Chapter 4: Recent Work
Sarah's latest exhibition, "Red Horizons," opened in March 2024 at the Portland Art Museum.
The collection features 15 large-scale watercolor landscapes, all incorporating her signature red tones.
She sold 12 pieces during the opening night, her most successful show to date.
"""


# Natural conversation scenarios (how a real user would ask)
CONVERSATION_SCENARIOS = [
    # ROUND 1: Basic recall (easy - should be in book memory)
    {
        "user_question": "What does Sarah do for work?",
        "ground_truth": "watercolor artist",
        "context": "Job/career question",
        "difficulty": "easy"
    },
    {
        "user_question": "Where does Sarah live?",
        "ground_truth": "Portland, Oregon",
        "context": "Location question",
        "difficulty": "easy"
    },
    {
        "user_question": "What's Sarah's favorite color?",
        "ground_truth": "red",
        "context": "Personal preference",
        "difficulty": "easy"
    },

    # ROUND 2: Inference (requires understanding, not just recall)
    {
        "user_question": "What time of day does Sarah prefer to paint?",
        "ground_truth": "morning (6 AM - 2 PM)",
        "context": "Daily routine inference",
        "difficulty": "medium"
    },
    {
        "user_question": "What kind of food does Sarah like?",
        "ground_truth": "Thai food",
        "context": "Food preference",
        "difficulty": "medium"
    },
    {
        "user_question": "Tell me about Sarah's recent art show",
        "ground_truth": "Red Horizons exhibition at Portland Art Museum in March 2024",
        "context": "Recent events",
        "difficulty": "medium"
    },

    # ROUND 3: Rephrased questions (tests if learning generalizes)
    {
        "user_question": "What's Sarah's profession?",  # Same as "what does she do"
        "ground_truth": "watercolor artist",
        "context": "Job question (rephrased)",
        "difficulty": "easy"
    },
    {
        "user_question": "Which city is Sarah based in?",  # Same as "where does she live"
        "ground_truth": "Portland",
        "context": "Location (rephrased)",
        "difficulty": "easy"
    },
    {
        "user_question": "What color shows up most in Sarah's work?",  # Related to favorite color
        "ground_truth": "red",
        "context": "Art style inference",
        "difficulty": "medium"
    },

    # ROUND 4: Multi-hop reasoning
    {
        "user_question": "What are Sarah's pets' names?",
        "ground_truth": "Pixel and Brush",
        "context": "Specific details",
        "difficulty": "hard"
    },
    {
        "user_question": "Where does Sarah work?",
        "ground_truth": "Pearl District studio in Portland",
        "context": "Work location",
        "difficulty": "medium"
    },
    {
        "user_question": "How successful was her latest exhibition?",
        "ground_truth": "Very successful - sold 12 out of 15 pieces on opening night",
        "context": "Success metrics",
        "difficulty": "hard"
    },

    # ROUND 5: Edge cases and tricky phrasings
    {
        "user_question": "I heard Sarah likes to paint. What medium does she use?",
        "ground_truth": "watercolor",
        "context": "Medium/technique",
        "difficulty": "easy"
    },
    {
        "user_question": "Does Sarah have any pets?",
        "ground_truth": "Yes, two cats named Pixel and Brush",
        "context": "Yes/no question with details",
        "difficulty": "medium"
    },
    {
        "user_question": "What's Sarah's husband's name and job?",
        "ground_truth": "David Chen, software engineer",
        "context": "Relationship details",
        "difficulty": "medium"
    },
]


async def upload_book_to_memory(memory: UnifiedMemorySystem, book_content: str, title: str):
    """Simulate user uploading a book/document"""
    print(f"\n📚 Uploading book: {title}")

    # Split into chunks (simulate real chunking)
    chunks = book_content.split('\n\n')
    chunks = [c.strip() for c in chunks if c.strip()]

    for i, chunk in enumerate(chunks):
        await memory.store(
            text=chunk,
            collection="books",
            metadata={
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "score": 0.7,  # Default book score
                "source": "user_upload"
            }
        )

    print(f"✓ Uploaded {len(chunks)} chunks to books collection")


async def have_conversation(
    memory: UnifiedMemorySystem,
    question: str,
    ground_truth: str,
    round_num: int
) -> Tuple[str, bool, List[Dict]]:
    """
    Simulate a natural conversation turn
    Returns: (answer, is_correct, retrieved_memories)
    """
    # Search memory naturally (books + patterns + history)
    results = await memory.search(question, collections=None, limit=5)

    # Build context from retrieved memories
    context_items = []
    for r in results:
        if isinstance(r, dict):
            content = r.get('content') or r.get('text', '')
            score = r.get('metadata', {}).get('score', 0.5)
            collection = r.get('collection', 'unknown')
            context_items.append({
                'content': content,
                'score': score,
                'collection': collection,
                'id': r.get('id')
            })

    context = "\n".join([c['content'] for c in context_items])

    # LLM generates answer based on context
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""You are Roampal, a helpful AI assistant with memory. Answer based on what you remember.

Your memories:
{context}

User: {question}

Answer naturally and conversationally:""",
                "stream": False
            }
        )

        if response.status_code == 200:
            answer = response.json().get('response', '').strip()
        else:
            answer = "Error generating response"

    # Judge if answer is correct (flexible matching)
    is_correct = await judge_answer_flexible(question, answer, ground_truth)

    return answer, is_correct, context_items


async def judge_answer_flexible(question: str, answer: str, ground_truth: str) -> bool:
    """
    More flexible judging - does the answer contain the key information?
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Does the answer contain the correct information?

Question: {question}
Expected information: {ground_truth}
Actual answer: {answer}

The answer doesn't need to match exactly - it just needs to convey the same key facts.
For example:
- Expected: "red" | Answer: "Her favorite color is red" → CORRECT
- Expected: "watercolor artist" | Answer: "She paints with watercolors" → CORRECT
- Expected: "Portland" | Answer: "She lives in Seattle" → INCORRECT

Answer ONLY with: CORRECT or INCORRECT""",
                "stream": False
            }
        )

        if response.status_code == 200:
            judgment = response.json().get('response', '').strip().upper()
            return "CORRECT" in judgment

        return False


async def learn_from_mistake(
    memory: UnifiedMemorySystem,
    question: str,
    answer: str,
    ground_truth: str,
    context_items: List[Dict],
    round_num: int
):
    """
    Natural learning: Store the correction as a conversational fact
    """
    # Extract fact from the ground truth
    fact = await extract_fact_natural(question, ground_truth)

    # Store as a natural fact memory
    await memory.store(
        text=fact,
        collection="patterns",  # Learned facts go to patterns
        metadata={
            "score": 0.9,  # High initial score
            "type": "learned_fact",
            "source": "conversation_correction",
            "round": round_num,
            "query": question  # For KG routing
        }
    )

    print(f"  💡 Learned: {fact}")

    # Punish memories that led to wrong answer
    for item in context_items[:3]:
        if item.get('id'):
            try:
                await memory.record_outcome(item['id'], "failed")
            except:
                pass


async def extract_fact_natural(question: str, ground_truth: str) -> str:
    """
    Extract a natural-sounding fact from question + ground truth
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": f"""Convert this Q&A into a natural statement of fact.

Question: {question}
Answer: {ground_truth}

Create a natural sentence that states this fact clearly. Examples:
- Q: "What's Sarah's favorite color?" A: "red" → "Sarah's favorite color is red"
- Q: "Where does Sarah live?" A: "Portland" → "Sarah lives in Portland, Oregon"
- Q: "What does Sarah do?" A: "watercolor artist" → "Sarah is a professional watercolor artist"

Write ONE natural sentence:""",
                "stream": False
            }
        )

        if response.status_code == 200:
            fact = response.json().get('response', '').strip()
            return fact

        # Fallback: simple template
        return f"Regarding '{question}': {ground_truth}"


async def reinforce_success(
    memory: UnifiedMemorySystem,
    context_items: List[Dict]
):
    """
    When answer is correct, boost the memories that helped
    """
    for item in context_items[:3]:
        if item.get('id'):
            try:
                await memory.record_outcome(item['id'], "worked")
            except:
                pass


async def run_natural_learning_test():
    """
    Main test: Simulate natural usage over multiple rounds
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"natural_learning_{timestamp}.log"

    print("="*80)
    print("🎭 NATURAL CONVERSATION LEARNING TEST")
    print("="*80)
    print("Simulating real-world usage:")
    print("1. Upload book about Sarah")
    print("2. Have natural conversations")
    print("3. System learns from corrections")
    print("4. Measure if accuracy improves")
    print("="*80)

    # Initialize memory
    memory = UnifiedMemorySystem(
        chroma_host="localhost",
        chroma_port=8000,
        collection_prefix="natural_learning_test"
    )
    await memory.initialize()

    # STEP 1: Upload the book
    await upload_book_to_memory(memory, SAMPLE_BOOK, "Sarah's Biography")

    # STEP 2: Run conversation rounds
    NUM_ROUNDS = 3  # Go through scenarios 3 times
    results_over_time = []

    with open(log_filename, 'w', encoding='utf-8') as log:
        log.write("="*80 + "\n")
        log.write("NATURAL CONVERSATION LEARNING TEST\n")
        log.write("="*80 + "\n\n")

        for round_num in range(1, NUM_ROUNDS + 1):
            print(f"\n{'='*80}")
            print(f"🔄 ROUND {round_num}/{NUM_ROUNDS}")
            print(f"{'='*80}")

            log.write(f"\n{'='*80}\n")
            log.write(f"ROUND {round_num}\n")
            log.write(f"{'='*80}\n\n")

            round_correct = 0
            round_total = 0

            for i, scenario in enumerate(CONVERSATION_SCENARIOS, 1):
                question = scenario['user_question']
                ground_truth = scenario['ground_truth']
                difficulty = scenario['difficulty']

                print(f"\n💬 Q{i}: {question}")
                log.write(f"\nQ{i} [{difficulty}]: {question}\n")
                log.write(f"Expected: {ground_truth}\n")

                # Have the conversation
                answer, is_correct, context_items = await have_conversation(
                    memory, question, ground_truth, round_num
                )

                print(f"🤖 Roampal: {answer}")
                log.write(f"Roampal: {answer}\n")

                if is_correct:
                    print(f"✓ CORRECT")
                    log.write(f"✓ CORRECT\n")
                    round_correct += 1
                    # Reinforce successful memories
                    await reinforce_success(memory, context_items)
                else:
                    print(f"✗ WRONG")
                    log.write(f"✗ WRONG\n")
                    # Learn from mistake
                    await learn_from_mistake(
                        memory, question, answer, ground_truth,
                        context_items, round_num
                    )

                round_total += 1

                # Small delay to not hammer the LLM
                await asyncio.sleep(0.5)

            # Round summary
            round_acc = (round_correct / round_total) * 100
            results_over_time.append(round_acc)

            print(f"\n📊 Round {round_num} Accuracy: {round_acc:.1f}% ({round_correct}/{round_total})")
            log.write(f"\n📊 Round {round_num} Accuracy: {round_acc:.1f}% ({round_correct}/{round_total})\n")

            # Show learning trend
            if round_num > 1:
                improvement = round_acc - results_over_time[0]
                print(f"📈 Improvement from Round 1: {improvement:+.1f}%")
                log.write(f"📈 Improvement from Round 1: {improvement:+.1f}%\n")

        # Final summary
        print(f"\n{'='*80}")
        print("📊 FINAL RESULTS")
        print(f"{'='*80}")
        log.write(f"\n{'='*80}\n")
        log.write("FINAL RESULTS\n")
        log.write(f"{'='*80}\n")

        for i, acc in enumerate(results_over_time, 1):
            print(f"Round {i}: {acc:.1f}%")
            log.write(f"Round {i}: {acc:.1f}%\n")

        if len(results_over_time) >= 2:
            total_improvement = results_over_time[-1] - results_over_time[0]
            print(f"\nTotal Learning: {total_improvement:+.1f}%")
            log.write(f"\nTotal Learning: {total_improvement:+.1f}%\n")

            if total_improvement > 10:
                print("✅ SYSTEM IS LEARNING! (>10% improvement)")
                log.write("✅ SYSTEM IS LEARNING! (>10% improvement)\n")
            elif total_improvement > 0:
                print("⚠️  Minimal learning (<10% improvement)")
                log.write("⚠️  Minimal learning (<10% improvement)\n")
            else:
                print("❌ NO LEARNING DETECTED")
                log.write("❌ NO LEARNING DETECTED\n")

    print(f"\n📝 Detailed log saved to: {log_filename}")


if __name__ == "__main__":
    asyncio.run(run_natural_learning_test())