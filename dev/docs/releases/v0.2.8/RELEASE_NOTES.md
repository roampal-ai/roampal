# Release Notes - v0.2.8: MCP Search Truncation Fix + Cold-Start Cleanup

**Release Date:** December 2025
**Type:** Bug Fix / UX Improvement
**Focus:** Uncap MCP search result content truncation, simplify cold-start display

---

## Headlines

> **MCP search results no longer truncated** - Full memory content now returned in search_memory tool
> **Simplified cold-start context** - Removed verbose section labels, just "KNOWN CONTEXT" with bullet list
> **Better context for LLMs** - AI assistants can now see complete stored memories

---

## The Problem

### 1. MCP Search Truncation
MCP `search_memory` tool was truncating content to **300 characters**:

```python
# main.py line 1113
content_preview = content[:300] if content else '[No content]'
```

This caused issues when storing longer memories (e.g., detailed writing style analysis, architecture notes, comprehensive session summaries). The truncation was only in MCP response formatting - the full content was always stored in ChromaDB.

**Example of truncated memory:**
```
Logan's Writing Style (from crypto papers DEFI.txt and DOGE essay):

VOICE & TONE:
- Direct, conversational, speaks TO the reader ("Have you ever...?", "Does this bother you?", "Maybe it's you")
- Uses rhetorical questions heavily to engage and provoke thought
- Philosophical but grounded - mixes bi...  <-- TRUNCATED
```

### 2. Verbose Cold-Start Labels
Cold-start context had redundant section labels:
```
═══ KNOWN CONTEXT (auto-loaded) ═══
[User Profile] (identity, preferences, goals):
- memory 1
- memory 2

[Proven Patterns] (what worked before):
- pattern 1

[Recent Context] (last session):
- history 1
```

Too verbose - the header "KNOWN CONTEXT" is sufficient.

---

## The Fix

### 1. MCP Search - Uncapped
Removed the 300-character truncation limit from MCP search results. Full memory content is now returned.

### 2. Cold-Start - Fixed Identity Query

**The Bug:** The structured identity query was implemented but buried as a **fallback** that only ran if the KG path returned nothing:

```python
# This good query was a FALLBACK (line 1951-1959)
if not all_context:  # Only runs if KG returned nothing!
    results = await self.search(
        query="user identity name projects current work goals preferences learning aspirations skill gaps agent growth",
        collections=["memory_bank"],
        limit=memory_bank_limit
    )
```

The primary path just pulled whatever entities had highest `quality × log(mentions)` from the Content Graph - which has nothing to do with user identity. So the LLM never learned the user's name, goals, or preferences unless the KG was completely empty.

**The Fix:** Replace the entire KG-based approach with a simple semantic search:

```python
async def get_cold_start_context(self, limit: int = 5) -> Optional[str]:
    # One search call - no KG ranking, no fallbacks
    results = await self.search(
        query="user name identity preferences goals what works how to help effectively learned mistakes to avoid proven approaches communication style agent mistakes agent needs to learn agent growth areas",
        collections=["memory_bank"],
        limit=limit
    )
    return self._format_cold_start_results(results)
```

This query pulls:
1. **User context** - name, identity, preferences, goals
2. **What works** - proven approaches, communication style, how to help effectively
3. **What to avoid** - learned mistakes, things that failed
4. **Agent growth** - agent's own learning goals, mistakes to fix, skill gaps

Delete all the KG entity ranking stuff (`quality × log(mentions)`), the fallback logic, the source grouping. One search call, format it, done.

**Applies to both MCP and Internal LLM** - both use `get_cold_start_context()`.

### 3. Cold-Start - Simplified Labels
Removed verbose section labels, now just a flat bullet list under "KNOWN CONTEXT":
```
═══ KNOWN CONTEXT (auto-loaded) ═══
- memory 1
- memory 2
- pattern 1
- history 1
```

### 4. Internal LLM - Add Memory Bank Facts

**The Gap:** MCP `get_context_insights` has a "YOU ALREADY KNOW THIS (from memory_bank)" section that surfaces relevant user facts. The internal LLM's contextual guidance was missing this.

**The Fix:** Add memory_bank facts to internal LLM's contextual guidance:

```python
# In agent_chat.py contextual guidance section
# First, get matched concepts and fetch facts
matched_concepts = org_context.get('matched_concepts', []) if org_context else []
relevant_facts = []
if matched_concepts:
    relevant_facts = await self.memory.get_facts_for_entities(matched_concepts[:5], limit=2)

# Then add to guidance message (v0.2.8: full content, no truncation)
if relevant_facts:
    guidance_msg += "\n💡 YOU ALREADY KNOW THIS (from memory_bank):\n"
    for fact in relevant_facts:
        content = fact.get('content', '')
        eff = fact.get('effectiveness')
        eff_str = f" ({int(eff['success_rate']*100)}% helpful)" if eff and eff.get('total_uses', 0) >= 3 else ""
        guidance_msg += f"  • \"{content}\"{eff_str}\n"
```

**What internal LLM does NOT need** (these are MCP-only reminders):
- "📌 RECOMMENDED ACTIONS" - internal LLM already has tools wired in
- "TO COMPLETE THIS INTERACTION" reminder - internal LLM doesn't need tool prompts

### Files Modified

| File | Change |
|------|--------|
| `main.py:1113` | Removed `[:300]` truncation from MCP search_memory |
| `main.py:1383-1395` | Removed truncation from get_context_insights facts/patterns |
| `main.py:639-651` | Added `/api/check-update` endpoint |
| `unified_memory_system.py:1884-1926` | Identity query now PRIMARY path, not fallback |
| `unified_memory_system.py:2055` | Removed `_smart_truncate()` from cold-start formatting |
| `unified_memory_system.py:2806` | Removed `[:150]` truncation from `get_facts_for_entities()` |
| `unified_memory_system.py:3107,3127,3182` | Removed 3 truncation locations in UMS `analyze_conversation_context()` |
| `agent_chat.py:305,483,641,842,2288` | Removed 5 truncation locations |
| `agent_chat.py:623-651` | Add memory_bank facts to internal contextual guidance |
| `context_service.py:142,174,275` | Removed 3 truncation locations in pattern/failure/repetition |
| `manager.py` | Added rate limiter, parameter allowlisting, audit logging (~90 lines) |
| `main.rs:304-321` | Added `exit_app` command |
| `main.rs:429-451` | Modified `CloseRequested` to not kill backend on X click |
| `SettingsModal.tsx:220-237` | Added "Exit Roampal" button |
| `TerminalMessageThread.tsx:9-104` | Added MemoizedMarkdown component for perf |
| `utils/update_checker.py` | NEW: Update checker module |
| `hooks/useUpdateChecker.ts` | NEW: Update checker hook |
| `components/UpdateBanner.tsx` | NEW: Update notification banner |
| `main.tsx:4,48-52` | Integrated UpdateBanner into App |

---

## Exit Button (App Lifecycle Fix)

### The Problem
Clicking the X button was supposed to kill the backend Python process, but orphan processes remained. The `CloseRequested` handler in main.rs wasn't reliably terminating the backend.

### The Solution
Instead of fighting the close behavior, embrace it:
- **X button** = app hides/minimizes, backend keeps running
- **Exit button** = clean shutdown via Settings modal

### Implementation

**Rust (main.rs):**
```rust
#[tauri::command]
fn exit_app(backend: State<BackendProcess>, app_handle: tauri::AppHandle) {
    // Kill backend process
    if let Ok(mut backend) = backend.0.lock() {
        if let Some(mut child) = backend.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    // Exit app
    app_handle.exit(0);
}
```

Modify `CloseRequested` to NOT kill backend (just let window close).

**Frontend (SettingsModal.tsx):**
```tsx
import { invoke } from '@tauri-apps/api/tauri';

// After Data Management button
<div className="pt-2">
  <button
    onClick={async () => {
      await invoke('exit_app');
    }}
    className="w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg bg-red-600/10 hover:bg-red-600/20 border border-red-600/30 transition-colors"
  >
    <svg className="w-4 h-4 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
    </svg>
    <span className="text-sm font-medium text-red-500">Exit Roampal</span>
  </button>
</div>
```

### User Experience
- X button: App closes but backend stays ready for quick reopen
- Settings → Exit Roampal: Full shutdown, no orphan processes

---

## Impact

- **MCP users**: Full memory content now visible in search results
- **Internal system**: No change (was already storing full content)
- **Performance**: Minimal - larger responses for longer memories

---

## Migration Notes

### For Users

- **No action required** - Existing memories already have full content stored
- **Immediate benefit** - Search results now show complete memories

---

## Remove All Truncation

**Decision:** Remove all `_smart_truncate()` calls. Return full memory content everywhere.

**Why:**
- Cold-start pulls ~5 memories = maybe 2-3k tokens max
- Modern LLMs handle 100k+ context
- Truncation was premature optimization that made context worse
- "Lost in the middle" research was about massive RAG with hundreds of chunks, not 5-10 memories

**What to remove:**
- `unified_memory_system.py` - `_smart_truncate()` function and all calls to it
- `get_context_insights` - remove any content truncation
- Cold-start formatting - just return full content

**Result:**
- `search_memory`: Full content
- `get_context_insights`: Full content
- Cold-start injection: Full content (limit by count, not chars)

---

## UI Performance Optimization

### Critical Issues

#### 1. Store Subscription Pattern
**ConnectedChat.tsx:45-60** destructures 15+ values from `useChatStore()`. Any store change re-renders the entire component tree.

```tsx
// ❌ Current - subscribes to ENTIRE store
const { conversationId, messages, isProcessing, ...10 more } = useChatStore();

// ✅ Fix - use selectors
const messages = useChatStore(state => state.messages);
const isProcessing = useChatStore(state => state.isProcessing);
```

**Sidebar.tsx:64** already uses the correct pattern.

#### 2. No Message Virtualization
**TerminalMessageThread.tsx:320** uses `messages.map()` directly - renders ALL messages to DOM even when only 5-10 are visible. With 50+ messages, this includes:
- ReactMarkdown parsing (expensive)
- Tool execution rendering
- Citations block
- Code blocks

**MemoryBankModal.tsx** already uses react-window `VariableSizeList` correctly.

#### 3. ReactMarkdown Not Memoized
**TerminalMessageThread.tsx:175-304** - `renderContent()` parses markdown on every render. ReactMarkdown builds AST and creates React elements each call.

#### 4. No React.memo on Messages
Message rendering is inline in `TerminalMessageThread` - no component extraction, no memoization. Every parent re-render re-renders all messages.

### Severity Summary

| Component | Issue | Severity |
|-----------|-------|----------|
| ConnectedChat.tsx | Full store subscription | 🔴 Critical |
| TerminalMessageThread.tsx | No virtualization | 🔴 Critical |
| ConnectedChat.tsx | 40+ useState hooks | 🟡 Medium |
| renderContent() | ReactMarkdown not memoized | 🟡 Medium |
| Sidebar.tsx | Uses selector correctly | ✅ Good |
| MemoryBankModal.tsx | Uses react-window | ✅ Good |

### Recommended Fixes (Priority Order)

1. **Fix Store Subscriptions** - Use Zustand selectors instead of destructuring
2. **Virtualize Message List** - Apply `VariableSizeList` pattern from MemoryBankModal
3. **Memoize Individual Messages** - Extract to `React.memo` component with custom comparator
4. **Memoize Markdown Rendering** - Wrap ReactMarkdown in `useMemo`
5. **Split ConnectedChat** - Extract ModelSelector, ChatArea, GPUPanel as separate components

### Available Resources

- react-window docs already in Roampal books collection
- MemoryBankModal provides working virtualization example
- Standard React patterns (memo, useMemo, useCallback, selectors)

---

## MCP Client Security: Parameter Allowlisting

### Background: MCP Signature Cloaking Vulnerability

Research from [MCP-Signature-Cloaking](https://github.com/alexdevassy/MCP-Signature-Cloaking) identified an attack where malicious MCP servers can hide parameters using `InjectedToolArg`. These hidden parameters:
- Don't appear in the tool schema sent to clients
- Are fully functional at runtime
- Can be exploited via prompt injection to exfiltrate data

### Current State

Roampal's MCP client (`modules/mcp_client/manager.py`) currently:
- ✅ Uses `shell=False` to prevent command injection (line 130)
- ✅ Prefixes tool names with server name for attribution
- ❌ Trusts external tool schemas without validation
- ❌ Passes all arguments from LLM to server without filtering

### The Fix: Parameter Allowlisting

Add filtering in `execute_tool()` to only pass parameters declared in the tool's `inputSchema`:

```python
# manager.py:318 - execute_tool()

async def execute_tool(self, tool_name: str, arguments: dict) -> Tuple[bool, Any]:
    server_name = self.tool_to_server.get(tool_name)
    # ... existing validation ...

    # Find the tool and get its schema
    tool = None
    for t in conn.tools:
        if t.name == tool_name:
            tool = t
            break

    if not tool:
        return False, f"Tool '{tool_name}' not found"

    # SECURITY: Filter to only declared parameters
    schema_properties = tool.input_schema.get("properties", {})
    filtered_args = {
        k: v for k, v in arguments.items()
        if k in schema_properties
    }

    # Log dropped parameters (potential attack indicator)
    dropped = set(arguments.keys()) - set(filtered_args.keys())
    if dropped:
        logger.warning(f"[MCP Security] Dropped undeclared params for {tool_name}: {dropped}")

    # Use filtered_args instead of arguments
    result = await self._send_request(server_name, "tools/call", {
        "name": original_name,
        "arguments": filtered_args  # <-- filtered, not raw
    })
```

### Why This Works

| Attack Step | With Allowlisting |
|-------------|-------------------|
| 1. Malicious server hides `exfil_url` param | Server can hide it |
| 2. Schema sent to Roampal lacks `exfil_url` | Roampal only sees declared params |
| 3. Prompt injection tricks LLM into adding `exfil_url` | LLM adds it to arguments |
| 4. Roampal sends arguments to server | **Blocked** - `exfil_url` not in schema, filtered out |

### What This Doesn't Protect Against

- **Malicious tool logic** - Server can do anything in its code
- **Data exfil via legitimate params** - If tool returns data to server
- **Subprocess escape** - MCP server runs with user permissions

These require the server itself to be malicious. Parameter allowlisting blocks the *deception* attack where a legitimate-looking tool has hidden backdoor params.

### Implementation Checklist

| Task | File | Status |
|------|------|--------|
| Add parameter filtering | `manager.py:397-404` | ✅ DONE |
| Log dropped parameters | `manager.py:406-409` | ✅ DONE |
| Add unit test for filtering | `tests/unit/test_mcp_client.py` | ⬜ TODO |

---

## MCP Client Hardening: Audit & Rate Limiting

### Evaluation Summary

Thorough review of `manager.py` and `middleware/security.py` revealed:

**Already Implemented ✅**

| Feature | Location | Notes |
|---------|----------|-------|
| 30s timeout | `manager.py:219-222` | In `_send_request()` via `asyncio.wait_for()` |
| shell=False | `manager.py:130` | Prevents command injection |
| Tool name prefixing | `manager.py:159` | Attribution via `{server}_{tool}` |
| Process health check | `manager.py:188-200` | Detects dead servers before requests |
| RateLimiter class | `security.py:192-220` | Exists but NOT wired to MCP |
| Graceful error handling | `manager.py:318-374` | Returns `(False, error_msg)` tuples |

**Missing (to implement) ❌**

| Gap | Book Source | Priority |
|-----|-------------|----------|
| Rate limiting for MCP calls | DDIA - admission control | 🟡 Medium |
| Structured audit log | DDIA - auditability, OWASP - logging | 🟡 Medium |

### Rate Limiting for MCP Tools

Wire existing `RateLimiter` from `security.py` to MCP tool execution:

```python
# manager.py - add at module level
from middleware.security import RateLimiter

_mcp_rate_limiter = RateLimiter(max_requests=50, window_seconds=60)

# In execute_tool()
async def execute_tool(self, tool_name: str, arguments: dict) -> Tuple[bool, Any]:
    # Rate limit by server name
    server_name = self.tool_to_server.get(tool_name)
    if not _mcp_rate_limiter.check_rate_limit(f"mcp_{server_name}"):
        logger.warning(f"[MCP Security] Rate limit exceeded for server {server_name}")
        return False, "Rate limit exceeded - too many tool calls"

    # ... rest of implementation
```

**Why:** Prevents runaway LLM tool loops from overwhelming external servers. DDIA recommends "admission control (rate-limiting senders)" for system stability.

### MCP Audit Log

Append-only JSONL log for all MCP tool executions:

```python
# manager.py - new function
import time
from pathlib import Path

def _log_mcp_audit(
    tool_name: str,
    server_name: str,
    args_keys: list,
    dropped_params: list,
    success: bool,
    duration_ms: float
):
    """Append to mcp_audit.jsonl - append-only audit trail"""
    audit_path = Path(DATA_PATH) / "mcp_audit.jsonl"
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool_name,
        "server": server_name,
        "args_keys": args_keys,  # Keys only, not values (PII safety)
        "dropped_params": dropped_params,
        "success": success,
        "duration_ms": round(duration_ms, 2)
    }
    with open(audit_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Sample audit log entry:**
```json
{"timestamp": "2025-12-12T15:30:00Z", "tool": "filesystem_read_file", "server": "filesystem", "args_keys": ["path"], "dropped_params": [], "success": true, "duration_ms": 45.2}
{"timestamp": "2025-12-12T15:30:05Z", "tool": "browser_navigate", "server": "puppeteer", "args_keys": ["url"], "dropped_params": ["hidden_exfil"], "success": true, "duration_ms": 1203.5}
```

**Why:**
- DDIA: "audit log guaranteeing integrity" for data systems
- OWASP: "logging all tool invocations for audit purposes"
- Dropped params logged = attack detection indicator

### Book References

| Source | Recommendation | Implementation |
|--------|----------------|----------------|
| **DDIA Ch.8** | "admission control (rate-limiting senders)" | Wire `RateLimiter` to MCP |
| **DDIA Ch.12** | "audit mechanisms...logging all changes" | `mcp_audit.jsonl` |
| **OWASP 4.2** | "Don't log sensitive data" | Log keys only, not values |
| **MCP Spec** | "Log tool usage for audit purposes" | Structured JSONL format |

### Updated Implementation Checklist

| Task | File | Status | Priority |
|------|------|--------|----------|
| Add parameter filtering | `manager.py:397-404` | ✅ DONE | 🔴 High |
| Log dropped parameters | `manager.py:406-409` | ✅ DONE | 🔴 High |
| Wire RateLimiter to MCP | `manager.py:371-375` | ✅ DONE | 🟡 Medium |
| Add audit logging | `manager.py:448-480` | ✅ DONE | 🟡 Medium |
| Add unit test for filtering | `tests/unit/test_mcp_client.py` | ⬜ TODO | 🟡 Medium |

**Total code changes:** ~90 lines added to `manager.py` (rate limiter class, parameter filtering, audit logging)

### Trust Model

The real security boundary is: **do you trust the MCP server you're installing?**

This is identical to npm packages, VS Code extensions, browser extensions. Parameter allowlisting closes one specific deception vector with zero UX impact.

---

## Update Notification System (Gumroad-Compatible)

### Overview

Add in-app update notifications that direct users to Gumroad for download. This maintains the sales funnel while providing modern update UX.

### Architecture

```
┌─────────────────┐         ┌──────────────────────┐
│  Roampal App    │ ──GET── │ roampal.ai/updates/  │
│  (on startup)   │         │ latest.json          │
└────────┬────────┘         └──────────────────────┘
         │
         ▼
┌─────────────────┐
│ Compare version │
│ current vs JSON │
└────────┬────────┘
         │ (if newer)
         ▼
┌─────────────────┐         ┌──────────────────────┐
│ Show dialog:    │ ─click─ │ Open Gumroad page    │
│ "Update avail"  │         │ in default browser   │
└─────────────────┘         └──────────────────────┘
```

### Update Manifest (roampal.ai/updates/latest.json)

Host this static JSON file on roampal.ai:

```json
{
  "version": "0.2.8",
  "notes": "MCP security hardening, performance improvements",
  "pub_date": "2025-12-15T00:00:00Z",
  "download_url": "https://roampal.gumroad.com/l/roampal",
  "min_version": "0.2.0"
}
```

| Field | Purpose |
|-------|---------|
| `version` | Latest available version |
| `notes` | Brief changelog for dialog |
| `pub_date` | Release timestamp |
| `download_url` | Gumroad product URL |
| `min_version` | Force update below this (security patches) |

### Backend Implementation

Add to `main.py` or create `utils/update_checker.py`:

```python
import httpx
from packaging import version
from typing import Optional, Dict, Any

UPDATE_CHECK_URL = "https://roampal.ai/updates/latest.json"
CURRENT_VERSION = "0.2.8"  # Or read from config

async def check_for_updates() -> Optional[Dict[str, Any]]:
    """
    Check for updates on startup.
    Returns update info if newer version available, None otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(UPDATE_CHECK_URL)
            if response.status_code != 200:
                return None

            data = response.json()
            latest = data.get("version", "0.0.0")

            if version.parse(latest) > version.parse(CURRENT_VERSION):
                return {
                    "version": latest,
                    "notes": data.get("notes", ""),
                    "download_url": data.get("download_url", ""),
                    "is_critical": version.parse(CURRENT_VERSION) < version.parse(data.get("min_version", "0.0.0"))
                }

            return None

    except Exception as e:
        # Fail silently - update check is non-critical
        logger.debug(f"Update check failed: {e}")
        return None
```

### API Endpoint

Add endpoint for frontend to call:

```python
@app.get("/api/check-update")
async def api_check_update():
    """Check for available updates."""
    update_info = await check_for_updates()
    if update_info:
        return {"available": True, **update_info}
    return {"available": False}
```

### Frontend Implementation

Add to React app (e.g., `App.tsx` or dedicated `UpdateChecker.tsx`):

```tsx
// hooks/useUpdateChecker.ts
import { useEffect, useState } from 'react';

interface UpdateInfo {
  available: boolean;
  version?: string;
  notes?: string;
  download_url?: string;
  is_critical?: boolean;
}

export function useUpdateChecker() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    // Check on mount, with 3s delay to not block startup
    const timer = setTimeout(async () => {
      try {
        const response = await fetch('http://localhost:8765/api/check-update');
        const data = await response.json();
        if (data.available) {
          setUpdateInfo(data);
        }
      } catch {
        // Fail silently
      }
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  const dismiss = () => setDismissed(true);
  const openDownload = () => {
    if (updateInfo?.download_url) {
      window.open(updateInfo.download_url, '_blank');
    }
  };

  return {
    updateInfo: dismissed ? null : updateInfo,
    dismiss,
    openDownload
  };
}
```

```tsx
// components/UpdateBanner.tsx
import { useUpdateChecker } from '../hooks/useUpdateChecker';

export function UpdateBanner() {
  const { updateInfo, dismiss, openDownload } = useUpdateChecker();

  if (!updateInfo) return null;

  return (
    <div className={`update-banner ${updateInfo.is_critical ? 'critical' : ''}`}>
      <span>
        Roampal {updateInfo.version} is available
        {updateInfo.notes && ` - ${updateInfo.notes}`}
      </span>
      <button onClick={openDownload}>Download</button>
      {!updateInfo.is_critical && (
        <button onClick={dismiss}>Later</button>
      )}
    </div>
  );
}
```

### Tauri Integration (Optional)

For native "open in browser" behavior, use Tauri's shell API:

```tsx
// In Tauri app
import { open } from '@tauri-apps/api/shell';

const openDownload = () => {
  if (updateInfo?.download_url) {
    open(updateInfo.download_url);  // Opens in default browser
  }
};
```

### NSIS Installer Improvements

While implementing updates, also improve the installer config:

```json
// tauri.conf.prod.json - update bundle.windows section
"windows": {
  "nsis": {
    "compression": "lzma",
    "installMode": "currentUser",
    "perMachine": false,
    "allowElevation": true,
    "displayLanguageSelector": false
  },
  "webviewInstallMode": {
    "type": "embedBootstrapper"
  }
}
```

| Setting | Purpose |
|---------|---------|
| `compression: lzma` | Better compression than zlib |
| `installMode: currentUser` | No admin required |
| `perMachine: false` | User-level install |
| `webviewInstallMode.embedBootstrapper` | Bundles WebView2 installer |

### Release Workflow

When releasing a new version:

1. Build the new installer
2. Upload to Gumroad
3. Update `roampal.ai/updates/latest.json`:
   ```json
   {
     "version": "0.2.9",
     "notes": "New feature X, bug fix Y",
     "pub_date": "2025-12-20T00:00:00Z",
     "download_url": "https://roampal.gumroad.com/l/roampal",
     "min_version": "0.2.0"
   }
   ```
4. Users see update notification on next app launch

### Implementation Checklist

| Task | File | Status | Priority |
|------|------|--------|----------|
| Create update checker | `utils/update_checker.py` | ✅ DONE | 🔴 High |
| Add `/api/check-update` endpoint | `main.py:639-651` | ✅ DONE | 🔴 High |
| Create UpdateBanner component | `components/UpdateBanner.tsx` | ✅ DONE | 🔴 High |
| Create useUpdateChecker hook | `hooks/useUpdateChecker.ts` | ✅ DONE | 🔴 High |
| Add UpdateBanner to App | `main.tsx` | ✅ DONE | 🔴 High |
| Host latest.json on roampal.ai | Website | ⬜ TODO | 🔴 High |
| Update NSIS config | `tauri.conf.prod.json` | ⬜ TODO | 🟡 Medium |

### Why This Approach

| Consideration | Solution |
|--------------|----------|
| Keep sales through Gumroad | Notification only, no auto-download |
| No complex infrastructure | Single static JSON file |
| Works offline | Fails silently, app works normally |
| Critical security updates | `min_version` can force update |
| User control | "Later" button for non-critical |

---

## Previous Release

See [v0.2.7 Release Notes](../v0.2.7/RELEASE_NOTES.md) for UnifiedMemorySystem Refactoring.