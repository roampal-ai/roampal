@echo off
echo ========================================
echo ROAMPAL TROUBLESHOOTING SCRIPT
echo ========================================
echo.

echo [1/8] Checking Ollama installation...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama not found in PATH
    echo Download from: https://ollama.com
    echo.
) else (
    echo [OK] Ollama found
    ollama --version
    echo.
)

echo [2/8] Checking if Ollama is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not running
    echo Start Ollama from Start Menu or run: ollama serve
    echo.
) else (
    echo [OK] Ollama is running on port 11434
    echo.
)

echo [3/8] Checking models (optional)...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    curl -s http://localhost:11434/api/tags 2>nul | findstr "qwen2.5" >nul
    if %errorlevel% neq 0 (
        echo [INFO] qwen2.5 model not found (will be prompted to download via UI)
        echo.
    ) else (
        echo [OK] qwen2.5 model found
        echo.
    )

    curl -s http://localhost:11434/api/tags 2>nul | findstr "nomic-embed-text" >nul
    if %errorlevel% neq 0 (
        echo [INFO] nomic-embed-text model not found (will be prompted to download via UI)
        echo.
    ) else (
        echo [OK] nomic-embed-text model found
        echo.
    )
) else (
    echo [INFO] Ollama not installed - models check skipped
    echo       Install Ollama first, then download models via the UI
    echo.
)

echo [4/8] Checking port availability...
netstat -ano | findstr ":8000" >nul
if %errorlevel% equ 0 (
    echo [WARNING] Port 8000 is in use
    netstat -ano | findstr ":8000"
    echo.
) else (
    echo [OK] Port 8000 is available
    echo.
)

echo [5/8] Checking Python runtime...
if exist "binaries\python\python.exe" (
    echo [OK] Python runtime found
    binaries\python\python.exe --version
    echo.
) else (
    echo [ERROR] Python runtime missing in binaries\python\
    echo.
)

echo [6/8] Checking backend files...
if exist "backend\main.py" (
    echo [OK] Backend main.py found
) else (
    echo [ERROR] backend\main.py is missing
)

if exist "backend\requirements.txt" (
    echo [OK] requirements.txt found
) else (
    echo [WARNING] requirements.txt not found
)
echo.

echo [7/8] Checking log files...
if exist "logs\backend_stderr.log" (
    echo [OK] Backend error log exists
    echo Last 10 lines:
    echo ----------------------------------------
    powershell -command "Get-Content logs\backend_stderr.log -Tail 10"
    echo ----------------------------------------
    echo.
) else (
    echo [INFO] No error log yet (expected on first run)
    echo.
)

echo [8/8] Checking data directories...
if exist "backend\data" (
    echo [OK] Data directory exists
) else (
    echo [INFO] Data directory will be created on first run
)
echo.

echo ========================================
echo TROUBLESHOOTING SUMMARY
echo ========================================
echo.
echo Common Issues:
echo 1. Ollama not installed or not running
echo    - Download from https://ollama.com
echo    - Make sure Ollama icon is in system tray
echo.
echo 2. Models not downloaded
echo    - Run: ollama pull qwen2.5:0.5b
echo    - Run: ollama pull nomic-embed-text
echo.
echo 3. Port 8000 in use
echo    - Close other applications using port 8000
echo    - Or restart your computer
echo.
echo 4. Backend won't start
echo    - Right-click Roampal.exe and "Run as Administrator"
echo    - Check logs\backend_stderr.log for errors
echo.
echo For more help: https://github.com/[your-repo]/roampal/issues
echo.
pause
