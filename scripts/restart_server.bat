@echo off
echo Stopping LoopSmith server...
taskkill /F /IM python.exe /FI "MEMUSAGE gt 500000"
timeout /t 2 /nobreak >nul

echo Starting LoopSmith server...
start /B python main.py

echo Waiting for server to initialize...
timeout /t 5 /nobreak >nul

echo Testing server health...
curl http://localhost:8000/health

echo.
echo Server restarted. Check the output above for any errors.