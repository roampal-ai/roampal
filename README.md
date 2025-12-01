# Roampal

[![Status](https://img.shields.io/badge/status-alpha-orange)](https://roampal.ai)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri-FFC131?logo=tauri)](https://tauri.app/)
[![Multi-Provider](https://img.shields.io/badge/LLM-Ollama%20%7C%20LM%20Studio-blue)](https://roampal.ai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**The memory layer that learns what actually works**

Stop re-explaining yourself every conversation. Roampal remembers your context, learns what actually works for you, and gets smarter over time—all while keeping your data 100% private and local.

<p align="center">
  <img src="docs/screenshot.png" alt="Roampal - Chat with Knowledge Graph" width="800">
</p>

> **Headline Result**: 130 adversarial scenarios. Plain vector search: 0-3%. Roampal: **100%**. 63% fewer tokens. [(Full benchmarks)](docs/BENCHMARKS.md)

<p align="center">
  <a href="https://github.com/roampal-ai/roampal">
    <img src="https://img.shields.io/github/stars/roampal-ai/roampal?style=social" alt="GitHub Stars">
  </a>
</p>

---

## Quick Start

1. **[Download from roampal.ai](https://roampal.ai)** and extract
2. Install [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai)
3. Right-click `Roampal.exe` → **Run as administrator**
4. Download a model in the UI → Start chatting!

Your AI starts learning about you immediately.

---

## Table of Contents

- [Why Roampal?](#why-roampal)
- [Key Features](#key-features)
- [MCP Integration](#mcp-integration)
- [Architecture](#architecture)
- [Supported Models](#supported-models)
- [Documentation](#documentation)
- [Pricing](#pricing)

---

## Why Roampal?

**The Problem**: You ask your AI "How do I debug this?" It suggests `print()` statements—the same advice that didn't help last time. Why? Because vector search matches **keywords**, not **what actually worked**.

**Why Vector Search Fails**:
```
Query: "How do I print and see variable values while debugging?"
   ↓
Vector DB returns: "Add print() statements to see variable values"
   ↓
But that advice FAILED last time. You needed the debugger.
```

**Roampal's Solution**: Track outcomes. When advice works, boost it (+0.2). When it fails, penalize it (-0.3). After a few conversations, the system **knows** debugger > print statements—for YOU.

**Result**: 100% vs 0-3% accuracy on 130 adversarial scenarios [(full stats)](docs/BENCHMARKS.md)

### Performance

| Metric | Result |
|--------|--------|
| **Accuracy** | **100% vs 0-3%** on 130 adversarial scenarios |
| **Token Efficiency** | **63% fewer** tokens per query (20 vs 55-93) |
| **Learning Curve** | **58% → 93%** accuracy as outcomes accumulate |
| **Latency (p95)** | Sub-100ms |

**Why this matters**: Better answers with less context. Lower API costs, faster responses, and it keeps getting smarter over time.

<details>
<summary>Statistical Details</summary>

- Coding (30 scenarios): p=0.001, Cohen's d=7.49
- Finance (100 scenarios): p<0.001, McNemar χ²=98
- Learning curve: p=0.005, Cohen's d=13.4

[Full methodology →](docs/BENCHMARKS.md)
</details>

---

## Key Features

**Memory That Learns**
- Outcome tracking: Scores every result (+0.2 worked, -0.3 failed)
- Smart promotion: Good advice becomes permanent, bad advice auto-deletes
- Cross-conversation: Recalls from ALL past chats

**Your Knowledge Base**
- Memory Bank: Permanent storage of preferences, identity, goals
- Books: Upload .txt/.md docs as searchable reference
- Pattern recognition: Detects what works across conversations

**Privacy First**
- 100% local: All data on your machine
- Works offline: No internet after model download
- No telemetry: Your data never leaves your computer

---

## MCP Integration

Connect Roampal to **Claude Desktop, Cursor**, and other MCP-compatible tools.

```
Settings → Integrations → Connect → Restart your tool
```

**6 tools available**: `search_memory`, `add_to_memory_bank`, `update_memory`, `archive_memory`, `get_context_insights`, `record_response`

[Full MCP documentation →](docs/architecture.md#mcp-integration)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    5-TIER MEMORY                        │
├─────────────┬─────────────┬─────────────┬──────────────┤
│   Books     │   Working   │   History   │   Patterns   │
│ (permanent) │   (24h)     │  (30 days)  │  (permanent) │
├─────────────┴─────────────┴─────────────┴──────────────┤
│                    Memory Bank                          │
│            (permanent user identity/prefs)              │
└─────────────────────────────────────────────────────────┘
```

**Core Technology:**
- Outcome-Based Learning: Memories adapt based on feedback
- Triple Knowledge Graphs: Routing + Content + Action-Effectiveness
- Hybrid Search: BM25 + Vector + Cross-Encoder reranking

[Architecture deep-dive →](docs/architecture.md)

---

## Supported Models

Works with any tool-calling model via Ollama or LM Studio:

| Model | Provider | Parameters |
|-------|----------|------------|
| Llama 3.x | Meta | 3B - 70B |
| Qwen 2.5 | Alibaba | 3B - 72B |
| Mistral/Mixtral | Mistral AI | 7B - 8x22B |
| GPT-OSS | OpenAI (Apache 2.0) | 20B - 120B |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | 5-tier memory, knowledge graphs, technical deep-dive |
| [Benchmarks](docs/BENCHMARKS.md) | Test methodology, statistical significance |
| [Release Notes](docs/RELEASE_NOTES_0.2.1.md) | Latest: Enhanced retrieval, causal learning |

---

## Important Notices

**AI Safety**: LLMs may generate incorrect information. Always verify critical information. Don't rely on AI for medical, legal, or financial advice.

**Model Licenses**: Downloaded models (Llama, Qwen, etc.) have their own licenses. Review before commercial use.

---

## Support

- **Discord**: https://discord.gg/F87za86R3v
- **Email**: roampal@protonmail.com
- **GitHub**: https://github.com/roampal-ai/roampal/issues
- **Author**: [Logan Teague](https://www.linkedin.com/in/logan-teague-mba-6909901a5/)

---

## Pricing

**Free & open-source** (MIT License)

- Build from source → completely free
- Pre-built executable: **$9.99 one-time** (saves hours of setup)
- Zero telemetry, full data ownership

---

**Made with love for people who want AI that actually remembers**
