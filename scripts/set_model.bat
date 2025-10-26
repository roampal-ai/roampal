@echo off
echo Available models:
echo 1. codellama:latest (3.8 GB) - Good for coding
echo 2. mistral-nemo:12b-instruct-2407-q4_0 (7.1 GB) - Best overall
echo 3. gemma:2b (1.7 GB) - Fast, lightweight
echo 4. tinyllama:1.1b (637 MB) - Very fast

set /p choice="Select model (1-4): "

if "%choice%"=="1" (
    set OLLAMA_MODEL=codellama:latest
) else if "%choice%"=="2" (
    set OLLAMA_MODEL=mistral-nemo:12b-instruct-2407-q4_0
) else if "%choice%"=="3" (
    set OLLAMA_MODEL=gemma:2b
) else if "%choice%"=="4" (
    set OLLAMA_MODEL=tinyllama:1.1b
) else (
    echo Invalid choice
    exit /b 1
)

echo.
echo Selected: %OLLAMA_MODEL%
echo.
echo Starting LoopSmith with %OLLAMA_MODEL%...
python main.py