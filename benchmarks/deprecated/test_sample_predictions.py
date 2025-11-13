"""
Test a small sample of LOCOMO questions to see actual predictions
"""
import asyncio
import json
import httpx

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

async def judge_answer(question: str, predicted: str, ground_truth: str) -> tuple:
    """LLM-as-a-Judge - returns (score, judgment_text)"""
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
            judgment = response.json().get('response', '').strip()
            score = 1.0 if "CORRECT" in judgment.upper() else 0.0
            return score, judgment
        return 0.0, "Error"

async def test_sample():
    """Test first 10 questions from conversation 1"""

    # Load dataset
    with open('locomo/data/locomo10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    conv1 = data[0]
    session_id = "sample_test"

    print("="*80)
    print("SAMPLE PREDICTION TEST - First 10 Questions from Conversation 1")
    print("="*80)

    # Ingest conversation first (simplified - just shows we're using existing data)
    print("\nUsing existing conversation data from previous LOCOMO run...")
    print("(Data should still be in Roampal's memory)\n")

    questions = conv1['qa']
    tested = 0

    for qa in questions:
        question = qa.get('question', '')
        ground_truth = qa.get('answer', '')

        if not question or not ground_truth:
            continue

        # Get prediction
        predicted = await answer_question_via_api(question, session_id)

        # Judge it
        score, judgment = await judge_answer(question, predicted, str(ground_truth))

        status = "✓ PASS" if score >= 1.0 else "✗ FAIL"

        print(f"\n{'-'*80}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted: {predicted}")
        print(f"Judge: {judgment}")
        print(f"Result: {status}")

        tested += 1
        if tested >= 10:
            break

    print("\n" + "="*80)
    print(f"Tested {tested} questions")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_sample())
