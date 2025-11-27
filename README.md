# Roampal

[![Status](https://img.shields.io/badge/status-alpha-orange)](https://roampal.ai)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri-FFC131?logo=tauri)](https://tauri.app/)
[![Multi-Provider](https://img.shields.io/badge/LLM-Ollama%20%7C%20LM%20Studio-blue)](https://roampal.ai)

THE Memory Layer That Actually Learns

Stop re-explaining yourself every conversation. Roampal remembers your context, learns what actually works for you, and gets smarter over time—all while keeping your data 100% private and local on your machine.

## Why Roampal?

**The Problem**: You've explained your setup to AI 47 times. It never learns what worked. You're paying $20/month(or more!) to re-train it daily.

**The Solution**: Roampal implements a **5-tier memory system** that:
- **Remembers you**: Your stack, preferences, projects stored permanently
- **Learns what works**: Tracks which solutions actually worked for YOU
- **Gets smarter over time**: Successful advice promotes to long-term memory, failures get deleted

Think of it as your personal AI that compounds in value the longer you use it.

### Performance Metrics

Validated performance characteristics:

| Metric | Result |
|--------|--------|
| **Search Latency (p95)** | **77ms** |
| **Token Efficiency** | **112 tokens/query** |
| **Learning Under Noise** | **80% precision @ 4:1 semantic confusion** |
| **Routing Accuracy** | **100% (cross-collection test)** |

[See benchmark methodology & results →](benchmarks/README.md) (12 test suites, 100+ tests)

---

### Key Features

Roampal includes advanced memory features:

- **Outcome-Based Learning**: Memories adapt based on feedback (+0.2 worked, -0.3 failed)
- **5-Tier Architecture**: Books, Working, History, Patterns, Memory Bank
- **Triple Knowledge Graphs**: Routing KG + Content KG + Action-Effectiveness KG
- **Local-First**: All processing on-device, no cloud dependencies

[See architecture details →](docs/architecture.md)

## Key Features

### 🧠 **Remembers Who You Are (Memory Bank)**
- **Autonomous identity storage** - AI automatically stores facts about YOU: identity, preferences, goals, projects
- **Permanent memory** - Never decays, always accessible (score fixed at 1.0)
- **Full control** - View, restore, or delete memories via Settings UI
- **Smart categorization** - Tags: identity, preference, goal, context, workflow
- *Example: "I prefer TypeScript" → AI stores permanently → Never suggests JavaScript again*

### 📚 **Learns Your Knowledge Base (Books)**
- **Upload your docs** - .txt, .md files become searchable permanent reference
- **Semantic chunking** - Smart document processing for accurate retrieval
- **Source attribution** - AI cites which document/page info came from
- **Persistent library** - Reference materials never expire or decay
- *Example: Upload architecture docs → AI references YOUR conventions when answering*

### 🎯 **Learns What Actually Works (Outcome Tracking)**
- **Automatic outcome detection** - Tracks when advice worked (+0.2) or failed (-0.3)
- **Smart promotion** - Score ≥0.7 + 2 uses → History. Score ≥0.9 + 3 uses → Patterns (permanent)
- **Auto-cleanup** - Bad advice (score <0.2) gets deleted automatically
- **Organic recall** - Proactively surfaces: *"You tried this 3 times before, here's what worked..."*

### 🔄 **Cross-Conversation Memory**
- **Global search** - Working memory searches across ALL conversations, not just current one
- **Pattern recognition** - Detects recurring issues across conversation boundaries
- **True continuity** - "You asked about this 3 weeks ago in a different chat..."

### 🎭 **Customizable Personality**
- **YAML templates** - Fully customize assistant tone, identity, behavior
- **Persistent preferences** - Your settings saved locally
- **Role flexibility** - Teacher, advisor, pair programmer, creative partner

### 🔒 **Privacy & Control**
- **100% local** - All data on your machine, zero cloud dependencies
- **Works offline** - No internet after model download
- **Full ownership** - Export, backup, or delete data anytime
- **No telemetry** - Your data never leaves your computer

## Real-World Use Cases

**For Developers**:
- "Remembers my entire stack. Never suggests Python when I use Rust."
- Learns debugging patterns that work for YOUR codebase
- Recalls past solutions: "This approach worked 3 weeks ago"

**For Students & Learners**:
- "My personal tutor that remembers what I struggle with"
- Tracks what concepts you've mastered
- Adapts explanations to your learning style over time

**For Writers & Creators**:
- "Remembers my story world, characters, and tone"
- Stores worldbuilding details permanently
- Tracks character arcs across conversations

**For Entrepreneurs & Founders**:
- "My business advisor that knows my entire strategy"
- Remembers your business model and goals
- Tracks which marketing approaches actually worked

## ⚠️ Important Notices

### AI Safety Disclaimer

**Roampal uses large language models (LLMs) which may:**
- Generate incorrect, outdated, or misleading information
- Produce inconsistent responses to similar queries
- Hallucinate facts, sources, or code that don't exist
- Reflect biases present in training data

**Always verify critical information** from authoritative sources. Do not rely on AI-generated content for:
- Medical, legal, or financial advice
- Safety-critical systems or decisions
- Production code without thorough review and testing

### Model Licensing Notice

**Downloaded models have separate licenses:**
- **Ollama models**: Llama (Meta - [License](https://ai.meta.com/llama/license/)), Qwen (Alibaba), etc. - Check [Ollama Library](https://ollama.com/library)
- **LM Studio models**: GGUF format from Hugging Face - Check individual model cards for licenses
- Models you download have their own terms of use - review before commercial use

---

## Performance Details

### Verified Metrics

**Search Performance:**
- p95 latency: 77ms
- Token efficiency: 112 tokens/query average
- Cross-collection routing: 100% accuracy (7/7 tests)

**Learning Capabilities:**
- Semantic confusion resistance: 80% precision under 4:1 noise ratio
- Outcome-based score adaptation: +0.2 (worked), -0.3 (failed)
- Smart promotion: Working → History (score ≥0.7, 2+ uses), History → Patterns (score ≥0.9, 3+ uses)

**Memory System:**
- 5-tier architecture: Books, Working, History, Patterns, Memory Bank
- Triple knowledge graphs: Routing KG + Content KG + Action-Effectiveness KG
- Quality-based ranking: importance × confidence scoring

> See [benchmarks/README.md](benchmarks/README.md) for test methodology

---

## Latest Release: v0.2.0

**Learning-Based Knowledge Graph Routing + Enhanced MCP Integration** 

### Major Features

**🎯 Intelligent KG Routing**: System learns which collections answer which queries
- Cold start (0-10 queries): Searches all collections
- Learning phase (10-20 queries): Focuses on top 2-3 successful collections
- Confident routing (20+ queries): Routes to single best collection with 80%+ success rate
- Progression: 60% precision → 80% precision → 100% precision achievable

**🔗 Enhanced MCP Integration**: Semantic learning storage with outcome-based scoring
- External LLMs (Claude Desktop, Cursor) store summaries, not verbatim transcripts
- Explicit outcome scoring (worked/failed/partial/unknown)
- Scores CURRENT learning immediately (enables optional tool calling)
- Cross-tool memory sharing across all MCP clients

**📊 Triple Knowledge Graph System**:
- **Routing KG** (blue nodes) - Learns query patterns → collection routing
- **Content KG** (green nodes) - Entity relationships extracted from memories
- **Action-Effectiveness KG** (orange nodes) - Learns which actions work in which contexts
- **Purple nodes** - Concepts appearing in multiple graphs

**🌐 Bundled Multilingual Embeddings**: Works offline in 50+ languages
- Model: `paraphrase-multilingual-mpnet-base-v2`
- No internet required after initial setup

### Performance
- Search latency: 77ms (p95)
- Token efficiency: 112 tokens/query
- Semantic confusion resistance: 80% precision @ 4:1 noise
- Routing accuracy: 100% (cross-collection KG test)

[View full changelog →](docs/RELEASE_NOTES_0.2.0.md)

---

## What Makes Roampal Different?

| Feature | Roampal Approach |
|---------|------------------|
| **Memory Type** | Learns what works for you, not just what you say |
| **Outcome Tracking** | Scores every result (+0.2 worked, -0.3 failed) |
| **Bad Advice** | Auto-deleted when score drops below threshold |
| **Context** | Recalls from all past conversations globally |
| **Privacy** | 100% local, zero telemetry, full data ownership |
| **Performance** | 77ms search latency (p95) |

## Getting Started

**Quick start:**
1. [Download from roampal.ai](https://roampal.ai) and extract
2. Install an LLM provider:
   - **Ollama** ([ollama.com](https://ollama.com)) - Recommended for beginners
   - **LM Studio** ([lmstudio.ai](https://lmstudio.ai)) - Advanced users with GUI preferences
3. Right-click `Roampal.exe` → **Run as administrator** (Windows requires this to avoid permission issues)
4. Download your first model in the UI (Roampal handles the rest!)

Your AI will start learning about you immediately.

### Updating Roampal

**To update to a new version:**
1. Download the latest release and extract it
2. Close Roampal if it's running
3. Replace your old Roampal folder with the new one
4. Run `Roampal.exe` - all your data is preserved!

**Your data is safe** - All conversations, memories, settings, and downloaded models are stored in AppData and remain intact across updates. Simply overwrite the program files and you're good to go.

## Architecture

Roampal uses a memory-first architecture with five tiers:

1. **Working Memory** (24h) - Current conversation context
2. **History** (30 days) - Recent conversations and interactions
3. **Patterns** (permanent) - Successful solutions and learned patterns
4. **Memory Bank** (permanent) - User preferences, identity, and project context
5. **Books** (permanent) - Uploaded reference documents

The LLM autonomously controls memory via tools (search_memory, create_memory, update_memory, archive_memory).

## MCP Integration

**Connect Roampal to Claude Desktop, Cursor, and other MCP-compatible tools** for persistent memory across applications.

### Setup (No Manual Config Required)

1. Open **Settings → Integrations** in Roampal
2. Click **"Connect"** next to Claude Desktop or Cursor
3. Restart your tool - memory tools are available immediately

**⚠️ Windows Admin Note**: If MCP connections fail, run both Roampal AND the connected application (Claude Desktop/Cursor) as administrator. Windows may block inter-process communication without elevated permissions.

Roampal auto-discovers MCP clients and writes the config for you. No manual JSON editing required.

### Available MCP Tools (6 tools)

- **`search_memory`** - Search across all memory tiers with optional metadata filtering
- **`add_to_memory_bank`** - Store permanent facts about the user
- **`update_memory`** - Modify existing memories by doc_id
- **`archive_memory`** - Remove outdated information
- **`get_context_insights`** - Get organic insights from Knowledge Graphs before searching (past patterns, failure warnings, action stats)
- **`record_response`** - Store semantic learnings with explicit outcome scoring (worked/failed/partial/unknown)

### How It Works

**Semantic Learning Storage**: External LLMs store summaries, not verbatim transcripts. The `record_response` tool accepts:
- `key_takeaway` (required) - 1-2 sentence summary of what was learned
- `outcome` (optional) - Explicit scoring: "worked", "failed", "partial", or "unknown" (default)

**Score CURRENT, not PREVIOUS**: Unlike Roampal's internal system (which scores previous exchanges), MCP scores the learning being recorded immediately. This allows optional tool calling - external LLMs only call `record_response` when clear outcomes occur.

**Scores retrieved memories too**: When you call `record_response`, it also scores all memories from your last search with the same outcome. If advice worked, those memories get upvoted (+0.2). If it failed, they get downvoted (-0.3). This helps good memories promote faster and bad advice get deleted.

**Cross-tool memory sharing**: Learnings recorded in Claude Desktop are searchable in Cursor, Roampal, and vice versa. All tools share the same local ChromaDB instance.

### Features

- ✅ **Auto-discovery** - Detects Claude Desktop, Cursor, and other MCP clients automatically
- ✅ **Semantic learning** - Stores concepts, not chat logs
- ✅ **Outcome-based scoring** - External LLM judges quality based on user feedback
- ✅ **50+ languages** - Bundled multilingual embedding model (paraphrase-multilingual-mpnet-base-v2)
- ✅ **100% local** - All data stays on your machine

## Pricing & Philosophy

### Why $9.99?

Roampal is an experiment in building sustainable technology without artificial scarcity or surveillance capitalism.

**Core principles:**
- ✅ Open source from day one (MIT License)
- ✅ One-time payment, not subscription trap
- ✅ Zero telemetry, zero tracking
- ✅ Your data stays on your machine
- ✅ Free to build from source forever

**The $9.99 pre-built version** includes:
- Tested, packaged executable with embedded Python
- Bundled dependencies (ChromaDB, FastAPI, multilingual embeddings)
- Ready-to-run on Windows with zero setup

**Building from source is free forever** - Technical users can clone the repo, install dependencies, and build for $0. The pre-built version exists to save you time, not lock you in.

### Supported Models

Works with any tool-calling capable model via Ollama or LM Studio:
- **Llama** - Meta's models (3B - 70B parameters)
- **Qwen** - Alibaba models (3B - 72B parameters)
- **GPT** - OpenAI models (20B - 120B parameters)
- **Mixtral** - Mistral's mixture-of-experts (8x7B)

Install models via Settings → Model Management in the UI.

## Support

For issues or feedback:
- **Discord**: https://discord.gg/F87za86R3v
- **Email**: roampal@protonmail.com
- **GitHub Issues**: https://github.com/roampal-ai/roampal/issues

---

**Made with ❤️ for people who want AI that actually remembers**
