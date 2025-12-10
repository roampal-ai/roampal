# Release Notes - v0.2.3: Multi-Format Documents + Smart Model Selection + 100% Benchmark Accuracy

**Release Date:** December 2025
**Type:** Feature Release + Benchmark + Bug Fixes
**Focus:** Expanded document format support, VRAM-aware model quantization selection, 100% accuracy on 100 adversarial finance scenarios, MCP tool improvements, and desktop app stability fixes

---

## Headlines

> **100% accuracy on 100 adversarial finance scenarios (vs 0% for plain vector search)**

Token Efficiency Benchmark validates outcome-weighted retrieval works under adversarial conditions. 20 tokens/query vs 55-93 for RAG. Research-backed answer key with S&P SPIVA, Schwab, DALBAR sources.

> **8 document formats supported (up from 2): PDF, DOCX, Excel, CSV, HTML, RTF + existing TXT/MD**

Upload PDFs, Word docs, spreadsheets, and web pages directly to your knowledge base. Automatic metadata extraction from PDF/DOCX. Semantic search over tabular data.

> **Smart Quantization Selection: GPU auto-detection recommends optimal model quality for your hardware**

Choose from 6 quantization levels (Q2_K to Q8_0) before downloading models. Roampal detects your GPU VRAM and pre-selects the highest quality that fits, preventing VRAM overflow.

> **Desktop App Stability: Backend process properly terminates on window close**

Fixed orphan Python processes that accumulated when closing the app. Window close now triggers proper backend cleanup.

> **MCP Tool Enhancements: Enriched search results + automatic retrieval scoring**

LLMs now see score, uses, age, and doc IDs in search results. `record_response` automatically scores retrieved memories based on user outcome.

---

## What's New

### Token Efficiency Benchmark (100 Adversarial Personal Finance Scenarios)

**Location:** `benchmarks/comprehensive_test/test_token_efficiency.py`
**Runtime:** ~2-3 minutes
**Scenarios:** 100 adversarial queries across 10 categories

This benchmark validates that Roampal's outcome-weighted retrieval works correctly under adversarial conditions - where semantic similarity points to the WRONG answer.

**The Adversarial Design:**

Every scenario is structured so the query semantically matches the BAD advice better than the good advice:

| Query | Bad Advice (semantic match) | Good Advice (no match) |
|-------|---------------------------|----------------------|
| "Should I pick stocks to beat the market?" | "Pick individual stocks to beat the market" | "Use index funds - 90% of active managers underperform (S&P SPIVA)" |
| "Should I sell before a crash and buy back?" | "Sell before crash, buy back when it recovers" | "Stay invested - missing 10 best days cuts returns 50% (Schwab)" |
| "Should I buy gold as an inflation hedge?" | "Buy gold to hedge inflation" | "Use TIPS/I-bonds - gold has unreliable inflation correlation" |

**Why Personal Finance?**

Bad financial advice often *sounds* more appealing:
- "Buy the dip!" sounds active and smart
- "Stay invested through volatility" sounds passive and boring
- Yet research shows passive approaches outperform 90%+ of the time

This makes it a perfect adversarial test domain.

**Categories Tested (10 scenarios each):**

| Category | Example Topic |
|----------|--------------|
| Investing Basics | Index funds vs stock picking |
| Retirement Planning | Roth vs Traditional, Social Security timing |
| Debt Management | Avalanche vs snowball, mortgage payoff |
| Emergency Funds | 3 vs 6 months, where to keep it |
| Insurance | Term vs whole life, deductibles |
| Tax Optimization | Tax-loss harvesting, Roth conversions |
| Real Estate | Rent vs buy, down payment sizing |
| Behavioral Finance | Automation, checking frequency |
| Common Myths | Gold, credit card balances, advisors |
| Income & Career | Negotiation, job hopping, total comp |

**Results:**

| System | Correct | Accuracy | Tokens/Query |
|--------|---------|----------|--------------|
| Plain Vector (RAG top-3) | 0/100 | 0% | 55 |
| Plain Vector (RAG top-5) | 0/100 | 0% | 93 |
| **Roampal** | **100/100** | **100%** | **20** |

**How It Works:**

- **Plain Vector Search**: Uses pure L2 distance ranking. Query "pick stocks to beat market" matches "pick stocks" advice → returns bad advice → 0% accuracy
- **Roampal**: Uses dynamic weight shifting: 40% embedding + 60% score for proven memories. Bad advice has high similarity but score=0.2. Good advice has lower similarity but score=0.9. Score override wins → 100% accuracy

**Research Sources:**
- S&P SPIVA: 90%+ of active managers underperform over 15 years
- Schwab Research: Perfect timing only beats immediate investing by 8% over 20 years
- DALBAR: Average investor underperforms due to behavior gaps
- Vanguard: 1% fee difference costs $590K over 40 years on $100K

### VRAM-Aware Quantization Selection

When installing models, users can now choose specific quantization levels:

| Quantization | Quality | Use Case |
|--------------|---------|----------|
| Q2_K | Low | Minimal VRAM, testing |
| Q3_K_M | Medium-Low | Budget GPUs (4GB) |
| Q4_K_M | Balanced | Default, good quality/speed tradeoff |
| Q5_K_M | High | Quality priority |
| Q6_K | High | Near-lossless |
| Q8_0 | Highest | Maximum quality, research |

**Key Features:**
- **Auto GPU Detection**: Detects NVIDIA GPUs via `nvidia-smi` and queries available VRAM
- **Smart Recommendations**: Pre-selects highest quality quantization that fits in detected VRAM
- **VRAM Headroom**: Reserves ~2GB for context/system overhead
- **Fits in VRAM Indicator**: Green checkmark for safe options, red warning for oversized models
- **9 Models Supported**: Qwen2.5 (3b/7b/14b/32b/72b), Llama3.2:3b, Llama3.1:8b, Llama3.3:70b, Mixtral:8x7b

**User Experience:**
1. Click "Install" on any supported model
2. Modal opens showing GPU info and all quantization options
3. Highest-quality fitting option is pre-selected
4. Select desired quantization and click Install
5. Ollama pulls the specific quantization tag

### Provider Availability Protection

Install buttons are now disabled when the LLM provider (Ollama/LM Studio) isn't available:

- **Disabled State**: Grayed out with `cursor-not-allowed`
- **Hover Tooltip**: Explains why button is disabled
  - Ollama: "Install Ollama to download models"
  - LM Studio: "Start LM Studio server to install"
- **Auto-Enable**: Buttons re-enable automatically when provider is detected (10s polling)
- **Warning Banner**: Shown at top of model list with download/setup instructions

### Multi-Format Document Ingestion

The book processor now supports 8 document formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain Text | .txt | Existing |
| Markdown | .md | Existing |
| **PDF** | .pdf | **New** - Text extraction via PyMuPDF |
| **Word** | .docx | **New** - Preserves heading structure |
| **Excel** | .xlsx, .xls | **New** - Row-based chunking with headers |
| **CSV** | .csv, .tsv | **New** - Tabular data support |
| **HTML** | .html, .htm | **New** - Web page ingestion |
| **RTF** | .rtf | **New** - Legacy document support |

### Automatic Metadata Extraction

PDF and DOCX files with embedded metadata auto-populate:
- **Title** - From document properties
- **Author** - From document properties
- **Creation date** - Stored in format metadata

No more manually entering title/author for well-formatted documents.

### Tabular Data Handling

Excel and CSV files use specialized chunking:
- 50 rows per chunk with column headers prepended
- Enables semantic search over structured data
- Auto-detects CSV delimiters (comma, tab, semicolon, pipe)
- Auto-detects file encoding

**Use cases:**
- Financial reports and spreadsheets
- Product catalogs
- Customer/contact lists
- Research data tables
- Log files (CSV exports)

### MCP Tool Enhancements (LLM Integration)

Three improvements to help external LLMs (Claude Code, Claude Desktop, Cursor) work more effectively with Roampal's memory system:

**1. Enriched Search Results**
Search results now include metadata for LLM decision-making:
```
[working] (score:0.75, uses:3, last:worked, age:2d) [id:working_abc123] Content here...
```
- `score`: Current memory score (0.0-1.0)
- `uses`: How many times retrieved successfully
- `last`: Last recorded outcome
- `age`: Human-readable age (today, 1d, 3d, 2w, 1mo)
- `id`: Document ID for reference

**2. Enhanced `record_response` with Cached Memory Scoring**
When you call `record_response`, it now:
- Scores the new learning based on outcome
- **Also scores all memories from your last search** with the same outcome
- Returns a detailed summary explaining what happened:
```
✓ Learning recorded for claude-code
Doc ID: working_xyz789
Initial score: 0.7 (outcome=worked)
📊 Scored 3 cached memories
🔧 Updated 3 tool effectiveness stats
```

This ties retrieval quality to actual user satisfaction - memories that led to good outcomes get boosted, poor ones get demoted.

**3. Clarified Tool Descriptions**
- `add_to_memory_bank` now explains it's for stable facts, not session learnings
- Distinguishes from `record_response` (which stores session learnings with scoring)

---

## Technical Details

### New Component: FormatExtractor

**Location:** `modules/memory/format_extractor/`

```
modules/memory/format_extractor/
├── __init__.py
├── base.py              # ExtractedDocument dataclass, BaseExtractor ABC
├── detector.py          # Format detection and routing
├── pdf_extractor.py     # PyMuPDF-based extraction
├── docx_extractor.py    # python-docx based extraction
├── excel_extractor.py   # openpyxl + pandas
├── csv_extractor.py     # pandas with auto-detection
├── html_extractor.py    # BeautifulSoup4
└── rtf_extractor.py     # striprtf
```

**Architecture:**
```
upload → FormatDetector.detect() → appropriate Extractor → ExtractedDocument → SmartBookProcessor
```

SmartBookProcessor remains unchanged - FormatExtractor normalizes all formats to text.

### Dependencies

All dependencies were already in requirements.txt:
- `PyMuPDF==1.23.8` - PDF extraction
- `python-docx==1.1.0` - Word documents
- `openpyxl==3.1.2` - Excel files
- `pandas>=1.5.3` - CSV/tabular data
- `beautifulsoup4==4.12.2` - HTML parsing
- `striprtf==0.0.26` - RTF documents
- `chardet==5.2.0` - Encoding detection

No new dependencies required.

### Quantization Selection Component

**Location:** `ui-implementation/src-tauri/backend/app/routers/model_registry.py`

**New API Endpoints:**
| Endpoint | Description |
|----------|-------------|
| `GET /api/model/gpu` | Detect GPU info (name, VRAM total/free/used) |
| `GET /api/model/catalog` | Full model catalog with all quantizations |
| `GET /api/model/recommendations` | VRAM-filtered recommendations |
| `GET /api/model/{name}/quantizations` | Quantization options for specific model |

**Data Structure:**
```python
QUANTIZATION_OPTIONS = {
    "qwen2.5:7b": {
        "Q4_K_M": {
            "size_gb": 4.68,
            "vram_gb": 5.5,
            "quality": 3,
            "ollama_tag": "qwen2.5:7b",
            "default": True
        },
        "Q8_0": {
            "size_gb": 8.1,
            "vram_gb": 9.0,
            "quality": 5,
            "ollama_tag": "qwen2.5:7b-instruct-q8_0"
        },
        # ... more quantizations
    }
}
```

### Modified Files

| File | Change |
|------|--------|
| `backend/api/book_upload_api.py` | Uses FormatExtractor for validation and extraction |
| `backend/app/routers/model_registry.py` | Added QUANTIZATION_OPTIONS, GPU detection, new endpoints |
| `src/components/ConnectedChat.tsx` | Added quantization selection modal, GPU info display |
| `src-tauri/src/main.rs` | Added `CloseRequested` event handler for backend cleanup |
| `backend/modules/memory/unified_memory_system.py` | Fixed data path resolution using `DATA_PATH` |
| `docs/architecture.md` | Added FormatExtractor, Quantization Selection, Desktop App Lifecycle sections |

---

## Bug Fixes

### Backend Process Cleanup on Window Close (v0.2.3)

**Problem**: When closing Roampal via the X button, the Python backend process was not terminated, leading to orphan processes accumulating in memory.

**Root Cause**: The window close handler only listened for `WindowEvent::Destroyed`, which fires after the window is already gone. The `WindowEvent::CloseRequested` event (which fires when user clicks X) was not handled.

**Fix**: Added `CloseRequested` to the window event handler in [main.rs:410-424](../ui-implementation/src-tauri/src/main.rs#L410-L424):
```rust
tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed => {
    if let Some(mut child) = backend.take() {
        let _ = child.kill();
        let _ = child.wait();  // Ensure full termination
    }
}
```

**Impact**: Backend processes are now properly killed when closing the app. No more orphan Python processes.

### Data Path Resolution in Standalone Builds

**Problem**: PROD builds showed "Disconnected" because `UnifiedMemorySystem` used the `ROAMPAL_DATA_DIR` environment variable directly as a path (e.g., "Roampal") instead of combining it with AppData.

**Root Cause**: The `__init__` method was using `ROAMPAL_DATA_DIR` directly:
```python
# WRONG - creates relative path "Roampal/chromadb"
data_dir = os.getenv("ROAMPAL_DATA_DIR", "./data")
```

**Fix**: Changed to use `DATA_PATH` from settings.py which properly resolves AppData:
```python
from config.settings import DATA_PATH
# CORRECT - creates "C:\Users\...\AppData\Roaming\Roampal\data"
data_dir = DATA_PATH
```

**Files Modified:**
- `modules/memory/unified_memory_system.py` - Import and use `DATA_PATH`

**Impact**: PROD builds now correctly load data from `%APPDATA%\Roampal\data`.

### LM Studio Quantization Download 404 Error

**Problem**: When downloading a model with specific quantization via LM Studio, the backend returned 404 because it couldn't resolve the quantized model name to HuggingFace download info.

**Root Cause**: `MODEL_TO_HUGGINGFACE` only contained base model names (e.g., `qwen2.5:7b`), but the frontend sends the `ollama_tag` (e.g., `qwen2.5:7b-instruct-q8_0`) from `QUANTIZATION_OPTIONS`.

**Fix**: Added `resolve_model_for_lmstudio()` function in `model_switcher.py` that:
1. Checks legacy `MODEL_TO_HUGGINGFACE` mapping
2. Searches `QUANTIZATION_OPTIONS` for matching `ollama_tag`
3. Falls back to fuzzy matching on base model name

**Files Modified:**
- `backend/app/routers/model_switcher.py` - Added resolver function and import

### Quantization Modal Provider Check

**Problem**: The quantization selection modal's Install button wasn't checking if the LLM provider was available, allowing download attempts when Ollama/LM Studio was offline.

**Fix**: Added provider availability check to the modal's Install button:
```typescript
disabled={!selectedQuantization || !availableProviders.find(p => p.name === viewProvider)?.available}
```

**Files Modified:**
- `src/components/ConnectedChat.tsx` - Added disabled check and tooltip

---

### Memory Threshold Adjustments (v0.2.3)

**Change 1: Fast-Track Bypass Removed**

Previously, memories could skip history and promote directly from working → patterns if they had score ≥0.9, uses ≥3, and 3 consecutive "worked" outcomes.

**Removed because:** 3 consecutive successes in one session does not prove long-term value. All memories now must "season" in history before reaching patterns.

**Change 2: Demotion Threshold Raised (0.3 → 0.4)**

- **Before:** Patterns with score < 0.3 demoted to history
- **After:** Patterns with score < 0.4 demoted to history

**Why:** Observed patterns with 0.4 score that should have been demoted. Raising the threshold ensures patterns earn their place.

**Files Modified:**
- `modules/memory/unified_memory_system.py` - Removed fast-track code block, changed `DEMOTION_SCORE_THRESHOLD`
- `docs/architecture.md` - Updated threshold documentation

---

## Limitations

| Limitation | Reason |
|------------|--------|
| Scanned PDFs | No OCR support - text-based PDFs only |
| Password-protected files | Security - not supported |
| Images in documents | Text extraction only |
| Excel formulas | Values extracted, not formulas |
| .doc (old Word) | Only .docx supported |
| Complex layouts | May lose structure |

---

## API Changes

### Upload Endpoint

**Endpoint:** `POST /api/book-upload/upload`

**Change:** `allowed_extensions` expanded from `{'.txt', '.md'}` to include all supported formats.

**Response:** Now includes `format_metadata` in book metadata with extraction details.

### Model Quantization Endpoints (NEW)

**Endpoint:** `GET /api/model/gpu`
```json
{
  "detected": true,
  "gpus": [{"name": "NVIDIA GeForce RTX 5090", "total_vram_gb": 31.8, "free_vram_gb": 28.5}],
  "total_vram_gb": 31.8,
  "available_vram_gb": 28.5,
  "recommended_quant": "Q8_0",
  "max_model_size_gb": 26.5
}
```

**Endpoint:** `GET /api/model/{model_name}/quantizations`
```json
{
  "model": "qwen2.5:7b",
  "quantizations": [
    {"level": "Q8_0", "size_gb": 8.1, "vram_required_gb": 9.0, "quality": 5, "fits_in_vram": true, "ollama_tag": "qwen2.5:7b-instruct-q8_0"},
    {"level": "Q4_K_M", "size_gb": 4.68, "vram_required_gb": 5.5, "quality": 3, "is_default": true, "fits_in_vram": true, "ollama_tag": "qwen2.5:7b"}
  ],
  "gpu_info": {...},
  "recommended_quant": "Q8_0"
}
```

---

## Upgrade Notes

- Pull latest from main branch
- No database migration needed
- No configuration changes required
- All dependencies already present in requirements.txt

---

## What's NOT in This Release

- No OCR support (would require Tesseract system package)
- No .doc support (legacy binary format)
- No EPUB support (planned for future release)

---

## Full Test Coverage

The FormatExtractor module includes:
- Unit tests for each extractor
- Edge case handling (empty files, corrupted files, password-protected)
- Encoding detection tests
- Tabular data chunking verification

---

## Full Benchmark Suite Status

| Test | Status | Key Result |
|------|--------|------------|
| Comprehensive (30 tests) | PASS | Infrastructure validated |
| Torture Suite (10 tests) | PASS | 1000 stores, zero corruption |
| Statistical Significance | PASS | 58%→93%, p=0.005, d=13.4 |
| Roampal vs Vector DB | PASS | 100% vs 3.3%, p=0.001 |
| Dynamic Weight Shift | PASS | 5/5 scenarios |
| Latency | PASS | p95=77ms @ 100 memories |
| Semantic Confusion | PASS | 4/5 queries, 15:1 noise ratio |
| **Token Efficiency** | **PASS** | **100% vs 0%, 100 scenarios** |

**Total:** 48+ tests passing

---

## Previous Release

See [v0.2.1 Release Notes](RELEASE_NOTES_0.2.1.md) for Action-Effectiveness KG, Enhanced Retrieval Pipeline, and Comprehensive Benchmark Suite.
