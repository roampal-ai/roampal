# Roampal

[![Status](https://img.shields.io/badge/status-alpha-orange)](https://roampal.ai)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri-FFC131?logo=tauri)](https://tauri.app/)
[![Multi-Provider](https://img.shields.io/badge/LLM-Ollama%20%7C%20LM%20Studio-blue)](https://roampal.ai)

**AI that grows with you.**

Stop re-explaining yourself every conversation. Roampal remembers your context, learns what actually works for you, and gets smarter over time—all while keeping your data 100% private and local on your machine.

## Why Roampal?

**The Problem**: You've explained your setup to AI 47 times. It never learns what worked. You're paying $20/month(or more!) to re-train it daily.

**The Solution**: Roampal implements a **5-tier memory system** that:
- **Remembers you**: Your stack, preferences, projects stored permanently
- **Learns what works**: Tracks which solutions actually worked for YOU
- **Gets smarter over time**: Successful advice promotes to long-term memory, failures get deleted

Think of it as your personal AI that compounds in value the longer you use it.

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

## What Makes Roampal Different?

| Feature | Cloud AI Memory | Roampal |
|---------|----------------|---------|
| **Memory Type** | Stores what you say | Learns what works for you |
| **Outcome Tracking** | No feedback loop | Scores every result (+0.2 / -0.3) |
| **Bad Advice** | Stays in memory | Auto-deleted when score drops |
| **Context** | Limited to current chat | Recalls from all past conversations |
| **Privacy** | Cloud-based, data shared | 100% local, zero telemetry |
| **Control** | Black box algorithms | Full transparency |

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

## Support

For issues or feedback:
- Discord: https://discord.gg/qM4yEXfF
- Email: roampal@protonmail.com
- GitHub Issues: https://github.com/roampal-ai/roampal/issues

---

**Made with ❤️ for people who want AI that actually remembers**
