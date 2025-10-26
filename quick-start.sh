#!/bin/bash

echo "╔══════════════════════════════════════╗"
echo "║       LoopSmith Quick Start          ║"
echo "║   Master Craftsman of Code Loops     ║"
echo "╚══════════════════════════════════════╝"
echo

echo "[1/4] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✓ Docker found"

echo
echo "[2/4] Building LoopSmith image..."
docker-compose build
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build LoopSmith image"
    exit 1
fi
echo "✓ LoopSmith image built"

echo
echo "[3/4] Starting services..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start services"
    exit 1
fi
echo "✓ Services started"

echo
echo "[4/4] Pulling Ollama models..."
docker-compose run --rm ollama-pull
echo "✓ Models ready"

echo
echo "════════════════════════════════════════"
echo "✅ LoopSmith is ready!"
echo
echo "🌐 Access LoopSmith at: http://localhost:8000"
echo "📚 API Docs at: http://localhost:8000/docs"
echo
echo "To stop LoopSmith: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo "════════════════════════════════════════"
echo

# Open in browser based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening LoopSmith in your browser..."
    sleep 3
    open "http://localhost:8000"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Opening LoopSmith in your browser..."
    sleep 3
    xdg-open "http://localhost:8000" 2>/dev/null || echo "Please open http://localhost:8000 in your browser"
fi