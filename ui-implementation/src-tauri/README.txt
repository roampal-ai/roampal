ROAMPAL v1.0.0 - Your Private Intelligence
=====================================

INSTALLATION
------------
1. Right-click Roampal-v1.0.0-Windows.zip
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
- Ollama installed with at least one model

First Time Setup
----------------
1. Install Ollama from https://ollama.com if not already installed
2. Pull required models:
   - Open Command Prompt/PowerShell
   - Run: ollama pull qwen2.5:0.5b (minimum required for chat)
   - Run: ollama pull nomic-embed-text (required for embeddings)
3. Right-click Roampal.exe and select "Run as administrator"
4. The app will auto-start the backend and open in your browser

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
- Multiple AI model support
- Automatic session management

Data Storage
------------
Your data is stored locally at:
%APPDATA%\Roampal\data\

This ensures your data persists across updates and remains private.

Support
-------
For issues or feedback, please visit:
https://github.com/[your-repo]/roampal/issues

Notes
-----
- The app automatically rotates logs to prevent disk bloat (max 40MB)
- First launch may take longer as the system initializes
- Memory search improves over time as you add more content
- All processing happens locally - no data leaves your computer

Open Source
-----------
Roampal is open source (MIT License).
Source code: https://github.com/[your-repo]/roampal
