@echo off
echo.
echo ========================================
echo     LoopSmith Model Selector
echo ========================================
echo.

ollama list

echo.
echo ========================================
echo.
set /p model="Enter model name (e.g., codellama:latest): "

if "%model%"=="" (
    echo No model selected.
    exit /b 1
)

echo.
echo Setting OLLAMA_MODEL=%model%
set OLLAMA_MODEL=%model%

REM Update .env file if it exists
if exist .env (
    powershell -Command "(Get-Content .env) -replace 'OLLAMA_MODEL=.*', 'OLLAMA_MODEL=%model%' | Set-Content .env"
    echo Updated .env file
)

echo.
echo Model set to: %model%
echo.
echo You can now start LoopSmith with: start_loopsmith.bat
echo.