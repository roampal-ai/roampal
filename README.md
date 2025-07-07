Copyright © 2025 Logan Teague. All rights reserved.
This software and all associated materials are proprietary and may not be used, copied, modified, or distributed without explicit written permission from the copyright holder.

# Roampal

**An Agent with a Soul: Your Personalized, Learning AI Partner**  
_Created by Logan Teague. All intellectual property rights reserved to the creator._

---

Roampal OG is an advanced AI assistant built on the principle that true intelligence requires not just knowledge, but a persistent, evolving memory. Unlike stateless models that “forget” after every query, Roampal OG is designed to learn continuously—growing, pruning, and refining its knowledge with every interaction, and acting as a proactive, context-aware partner.

---

## Core Features

**Evolving Memory (“The Soul”)**  
Roampal’s core is a dynamic vector store (ChromaDB) representing its “soul.” This memory is *not* static: it expands with every conversation and document ingestion, and also prunes or decays fragments that are unhelpful or unused over time.

**Sentiment-Based Learning**  
User sentiment is inferred automatically from your responses. If your next message is positive or satisfied, the agent upvotes (reinforces) the relevant memory “neurons.” If it’s negative or critical, the fragment is penalized (downvoted). No explicit thumbs-up/down required.

**Hybrid Retrieval**  
Memory retrieval is based on both semantic similarity and learned “strength” scores, so the agent increasingly favors memories validated as useful.

**Multi-Source Knowledge**  
Roampal synthesizes information from:
- **Long-Term Memory:** Ingested documents (books, articles, etc.)
- **Episodic Memory:** Raw text from all past conversations, searchable by similarity.
- **Short-Term Memory:** The immediate context of your current session.
- **External Knowledge:** [Optional] Web search via Playwright scraper when needed.

**Agentic Reasoning & Tool Use**  
Roampal is not just a Q&A bot. It autonomously decides whether to answer from its internal memory, use a tool (like web search), or resolve conflicting info through an internal self-debate process (early-stage).

---

## Architecture Deep Dive

1. **Tool Decision:**  
   When you ask a question, Roampal first decides if its own memory is sufficient or if a tool (like web search) is needed.

2. **Memory Retrieval (RAG):**  
   It simultaneously queries its “soul” (ChromaDB) for the most relevant knowledge, and also searches past conversation logs for similar context.

3. **Hybrid Ranking:**  
   Results are ranked with a blend of semantic similarity and each fragment’s learned “strength.”

4. **Prompt Synthesis:**  
   The agent constructs a prompt for the LLM, including:
   - Your question
   - Short-term memory (recent conversation turns)
   - Top-ranked fragments from its soul (both book knowledge and past chats)
   - Any web search results (if used)

5. **Response Generation & Learning:**  
   - The LLM generates a reply from this combined context.
   - The answer is saved as a new memory fragment.
   - It is then queued for feedback scoring after your next message.

6. **The Feedback Loop:**  
   - When you reply again, sentiment is automatically analyzed.
   - The previous answer’s score is updated, reinforcing helpful knowledge and decaying the rest.

---

## Tech Stack

- **Backend:** Python, FastAPI
- **LLM Engine:** Ollama (default: Llama 3 8B, model swappable)
- **Vector Store:** ChromaDB
- **Embeddings:** `all-MiniLM-L6-v2` (via Sentence-Transformers)
- **Text Processing:** LangChain (for text splitting)
- **Sentiment Analysis:** Hugging Face Transformers pipeline
- **Web Scraping:** Playwright (optional for real-time search)

---

## Setup and Installation

**Clone the repository:**
```bash
git clone <your-repo-url>
cd RoampalAI




Create and activate a virtual environment:

bash
python -m venv venv
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Set up local services:

Ensure Ollama is installed, running, and you’ve pulled your desired model (e.g., ollama pull llama3).

[Optional] For web search: Ensure the Playwright web scraper service is running.
(You may need playwright install after pip install.)

Configure environment variables:

Copy .env.example to .env and configure key settings (like OLLAMA_BASE_URL).
[Refer to included docs or comments for required variables.]

Usage
1. Ingest Knowledge (Optional but recommended)
To add books or documents to the agent’s soul, place .txt files in backend/data/og_books/ and run:

bash
python -m utils.book_processor
Wait for processing to finish before starting the main app.

2. Start the Main Application
bash
uvicorn backend.app.main:app --reload
3. Interact with the API
Roampal OG is available at the /chat/og endpoint.

POST /chat/og
Body:

json
{
  "user_input": "Your question here...",
  "session_id": "optional-session-id"
}
Example cURL:

bash
curl -X POST http://localhost:8000/chat/og \
     -H "Content-Type: application/json" \
     -d '{"user_input":"Who wrote Meditations?","session_id":"mysession123"}'
[You can also use the provided frontend UI.]
