# Building from Source

Roampal is fully open source. You can build from source for development or personal use.

## Architecture

- **Backend:** Python FastAPI + embedded Python runtime at `ui-implementation/src-tauri/binaries/python/`
- **Frontend:** Tauri (Rust) + React + TypeScript
- **Database:** ChromaDB (vector store) + SQLite
- **LLM:** Ollama (local)

## Development

```bash
# Clone the repository
git clone https://github.com/roampal-ai/roampal.git
cd roampal/ui-implementation

# Install dependencies
npm install

# Run in development mode
npm run tauri dev
```

The embedded Python runtime and backend start automatically via Tauri.

## Production Build

See the GitHub Actions workflow (`.github/workflows/build.yml`) for the automated build process used for official releases.

For manual builds:
```bash
cd ui-implementation
npm run tauri build
```

**Note:** We recommend downloading the official release from [roampal.ai](https://roampal.ai) for the best experience.

---

*For support, see [README.md](README.md)*
