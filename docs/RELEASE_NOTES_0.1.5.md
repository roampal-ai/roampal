# Roampal v0.1.5 - Multi-Provider Support & Performance Fixes

**Release Date:** October 31, 2025
**Build:** `Roampal_0.1.5_x64-setup.exe`

## What's New

### Multi-Provider Support
- **LM Studio Integration**: Roampal now supports both Ollama and LM Studio as LLM providers
- **Provider Auto-Detection**: Automatically detects which providers are running on your system (polls every 10s)
- **Seamless Switching**: Switch between providers without losing your chat context
- **Provider Status Display**: Settings modal shows active providers with model counts
- **Unified Model Library**: Browse and download models from both providers in one interface
- **Model Dropdown Filtering**: Only shows models for the currently selected provider

### Performance & Reliability Improvements
- **Fixed Redundant Memory Searches**: AI was searching memory 3 times for the same query - now searches once and reuses results
- **Collection-Specific Memory Search**: AI can now target specific memory collections (books, history, patterns) instead of always searching everything
- **Fixed Cancel Button**: Cancel button now properly stops generation and cleans up backend tasks

## Technical Details

### Architecture Changes
- Multi-provider architecture supporting Ollama and LM Studio
- Provider auto-detection system with periodic polling
- Unified model management across providers
- Enhanced error handling and logging throughout backend

### API Changes
- **New Endpoints**:
  - `GET /api/model/lmstudio/status` - Check LM Studio availability
  - `GET /api/model/providers/detect` - Detect all running providers
  - `GET /api/model/providers/all/models` - List models from all providers
- **Enhanced Endpoints**:
  - `POST /api/model/switch` - Now supports `provider` parameter for multi-provider switching

## Upgrade Notes

- No breaking changes
- Existing configurations will be preserved
- Provider preference saved to localStorage
- Model selection persists across updates
- All your data (conversations, memories, models) remains intact in AppData
