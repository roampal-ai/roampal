ROAMPAL v0.2.7 - Your Private Intelligence
=====================================

INSTALLATION
------------
1. Right-click Roampal_v0.2.7_alpha.zip
2. Select "Extract All..."
3. Choose a location (e.g., C:\Roampal)
4. Click "Extract"
5. Open the extracted folder

IMPORTANT: Run as Administrator
--------------------------------
For best performance and to avoid permission issues, RIGHT-CLICK on Roampal.exe
and select "Run as administrator".

System Requirements
-------------------
- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- For chat features: Ollama OR LM Studio with at least one model
- For MCP server: No additional requirements (embedding model bundled)

First Time Setup
----------------
For Chat Features:
1. Install either:
   - Ollama from https://ollama.com, OR
   - LM Studio from https://lmstudio.ai
2. Pull/download at least one chat model:
   - Ollama: ollama pull qwen2.5:7b
   - LM Studio: Download from built-in model browser
3. Right-click Roampal.exe and select "Run as administrator"
4. The app will auto-start the backend and open in your browser

For MCP Server Only (No Chat):
1. Right-click Roampal.exe and select "Run as administrator"
2. Configure MCP client (Claude Desktop, Cursor, etc.) to use Roampal
3. No additional setup needed - embedding model bundled (paraphrase-multilingual-mpnet-base-v2)

Troubleshooting
---------------
If the application doesn't start properly:
1. Run TROUBLESHOOT.bat to diagnose issues
2. Check that Ollama is running (should see icon in system tray)
3. Ensure no other application is using ports 8000 or 11434
4. Check logs/backend_stderr.log for error messages

Features
--------
- Smart memory management with auto-categorization
- Real-time chat with memory context
- Book/document upload and processing
- Memory visualization and search
- Multiple AI model support (Ollama + LM Studio)
- MCP (Model Context Protocol) server for AI tool integrations
- Automatic session management
- Multilingual embeddings (50+ languages)

Security Note
-------------
Only add MCP servers from sources you trust. MCP servers run with your
user permissions and can execute code on your machine.

Data Storage
------------
Your data is stored locally at:
%APPDATA%\Roampal\data\

This ensures your data persists across updates and remains private.

Support
-------
For issues or feedback, please visit:
https://github.com/roampal-ai/roampal/issues

Notes
-----
- The app automatically rotates logs to prevent disk bloat (max 40MB)
- First launch may take longer as the system initializes
- Memory search improves over time as you add more content
- All processing happens locally - no data leaves your computer

Open Source
-----------
Roampal is open source (Apache 2.0 License).
Source code: https://github.com/roampal-ai/roampal
