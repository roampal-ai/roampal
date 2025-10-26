@echo off
echo ╔══════════════════════════════════════╗
echo ║       LoopSmith Quick Start          ║
echo ║   Master Craftsman of Code Loops     ║
echo ╚══════════════════════════════════════╝
echo.

echo [1/4] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo ✓ Docker found

echo.
echo [2/4] Building LoopSmith image...
docker-compose build
if %errorlevel% neq 0 (
    echo ERROR: Failed to build LoopSmith image
    pause
    exit /b 1
)
echo ✓ LoopSmith image built

echo.
echo [3/4] Starting services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)
echo ✓ Services started

echo.
echo [4/4] Pulling Ollama models...
docker-compose run --rm ollama-pull
echo ✓ Models ready

echo.
echo ════════════════════════════════════════
echo ✅ LoopSmith is ready!
echo.
echo 🌐 Access LoopSmith at: http://localhost:8000
echo 📚 API Docs at: http://localhost:8000/docs
echo.
echo To stop LoopSmith: docker-compose down
echo To view logs: docker-compose logs -f
echo ════════════════════════════════════════
echo.
echo Opening LoopSmith in your browser...
timeout /t 3 >nul
start http://localhost:8000

pause