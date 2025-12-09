"""
MCP Integration Router
Handles auto-detection and connection to MCP-compatible AI tools
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import sys
import json
import platform
from pathlib import Path
from typing import List, Optional
import logging
from config.settings import DATA_PATH

# File to store manually-added custom MCP client paths
CUSTOM_PATHS_FILE = Path(DATA_PATH) / "mcp_custom_paths.json"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class MCPTool(BaseModel):
    name: str
    status: str  # "connected", "available", "not_installed"
    config_path: Optional[str] = None


class ConnectRequest(BaseModel):
    config_path: str


def create_default_mcp_config(directory: Path, filename: str = "mcp.json") -> Path:
    """Create a default MCP config file if it doesn't exist

    Args:
        directory: Directory to create config in
        filename: Config filename (default: mcp.json)

    Returns:
        Path to created/existing config file
    """
    config_path = directory / filename

    if not config_path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        default_config = {"mcpServers": {}}
        config_path.write_text(json.dumps(default_config, indent=2))
        logger.info(f"[MCP] Created default config at {config_path}")

    return config_path


def get_roampal_command() -> dict:
    """Get platform-specific command to run Roampal MCP server

    Returns dict with 'command' and 'args' keys for MCP config
    """
    current_dir = Path(__file__).parent
    system = platform.system()

    # Try to find bundled executable first
    if system == "Windows":
        exe_path = current_dir.parent.parent.parent / "Roampal.exe"
        if exe_path.exists():
            return {
                "command": str(exe_path.absolute()),
                "args": ["--mcp"],
                "env": {
                    "ROAMPAL_HEADLESS": "1"  # Signal to hide window
                }
            }
    elif system == "Darwin":  # macOS
        # Check for .app bundle
        app_path = current_dir.parent.parent.parent / "Roampal.app" / "Contents" / "MacOS" / "Roampal"
        if app_path.exists():
            return {
                "command": str(app_path.absolute()),
                "args": ["--mcp"]
            }
    else:  # Linux
        bin_path = current_dir.parent.parent.parent / "roampal"
        if bin_path.exists():
            return {
                "command": str(bin_path.absolute()),
                "args": ["--mcp"]
            }

    # Fallback: Development mode with Python
    # Try to find embedded Python first (prefer pythonw.exe on Windows to hide console)
    python_paths = []

    if system == "Windows":
        # Try pythonw.exe first (windowless) for embedded Python
        pythonw_embedded = current_dir.parent.parent / "binaries" / "python" / "pythonw.exe"
        python_embedded = current_dir.parent.parent / "binaries" / "python" / "python.exe"
        python_paths = [pythonw_embedded, python_embedded, Path(sys.executable)]
    else:
        python_paths = [
            current_dir.parent.parent / "binaries" / "python" / "python",  # Linux/macOS embedded
            Path(sys.executable),  # System Python
        ]

    python_exec = None
    for py_path in python_paths:
        if py_path.exists():
            python_exec = str(py_path.absolute())
            break

    if not python_exec:
        python_exec = "pythonw" if system == "Windows" else "python"  # Fallback to system PATH

    main_py = current_dir.parent.parent / "main.py"

    return {
        "command": python_exec,
        "args": [str(main_py.absolute()), "--mcp"]
    }




def load_custom_paths() -> List[str]:
    """Load manually-added custom MCP client config paths"""
    if not CUSTOM_PATHS_FILE.exists():
        return []
    try:
        data = json.loads(CUSTOM_PATHS_FILE.read_text())
        return data.get("custom_paths", [])
    except Exception as e:
        logger.error(f"[MCP] Error loading custom paths: {e}")
        return []


def save_custom_path(config_path: str):
    """Save a manually-added custom MCP client config path"""
    try:
        paths = load_custom_paths()
        if config_path not in paths:
            paths.append(config_path)
            CUSTOM_PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
            CUSTOM_PATHS_FILE.write_text(json.dumps({"custom_paths": paths}, indent=2))
            logger.info(f"[MCP] Saved custom path: {config_path}")
    except Exception as e:
        logger.error(f"[MCP] Error saving custom path: {e}")


def migrate_claude_config_if_needed():
    """
    Fix misplaced roampal config in Claude Desktop.
    If roampal is in config.json (UI settings), move it to claude_desktop_config.json (MCP config).
    """
    try:
        claude_dir = Path.home() / "AppData/Roaming/Claude" if platform.system() == "Windows" else \
                     Path.home() / "Library/Application Support/Claude" if platform.system() == "Darwin" else \
                     Path.home() / ".config/Claude"

        config_json = claude_dir / "config.json"
        mcp_config = claude_dir / "claude_desktop_config.json"

        if not config_json.exists():
            return

        # Read config.json
        config_data = json.loads(config_json.read_text())
        mcp_servers = config_data.get("mcpServers", {})

        # If roampal is in config.json (wrong file)
        if "roampal" in mcp_servers:
            roampal_config = mcp_servers["roampal"]
            logger.info("[MCP] Found roampal in config.json (UI settings file) - migrating to claude_desktop_config.json")

            # Read or create claude_desktop_config.json
            if mcp_config.exists():
                mcp_data = json.loads(mcp_config.read_text())
            else:
                mcp_data = {"mcpServers": {}}

            # Add roampal to correct file
            if "mcpServers" not in mcp_data:
                mcp_data["mcpServers"] = {}
            mcp_data["mcpServers"]["roampal"] = roampal_config

            # Write to claude_desktop_config.json
            mcp_config.write_text(json.dumps(mcp_data, indent=2))
            logger.info(f"[MCP] ✓ Migrated roampal config to {mcp_config}")

            # Remove from config.json (clean up wrong location)
            del config_data["mcpServers"]["roampal"]
            config_json.write_text(json.dumps(config_data, indent=2))
            logger.info("[MCP] ✓ Removed roampal from config.json")

    except Exception as e:
        logger.warning(f"[MCP] Could not migrate Claude config: {e}")


@router.get("/scan")
async def scan_for_mcp_tools():
    """Auto-detect MCP-compatible tools by scanning common config locations"""
    # First, fix any misplaced configs for Claude Desktop
    migrate_claude_config_if_needed()

    tools = []
    seen_paths = set()
    seen_tools = {}  # Deduplicate by tool name

    # MCP config filename patterns to search for
    mcp_config_patterns = [
        "*mcp*.json",        # Any file with "mcp" in name
        "config.json",       # Generic config files
        "*_config.json",     # Any app_config.json pattern
    ]

    # Broad search directories (cross-platform)
    system = platform.system()
    search_dirs = []

    if system == "Windows":
        search_dirs.extend([
            Path.home(),  # Home directory root for .cursor, .vscode, etc.
            Path.home() / "AppData/Roaming",
            Path.home() / "AppData/Local",
            Path.home() / ".config",
        ])
    elif system == "Darwin":  # macOS
        search_dirs.extend([
            Path.home(),  # Home directory root
            Path.home() / "Library/Application Support",
            Path.home() / ".config",
        ])
    else:  # Linux
        search_dirs.extend([
            Path.home(),  # Home directory root
            Path.home() / ".config",
            Path.home() / ".local/share",
        ])

    # Helper function to get config file priority for deduplication
    def get_config_priority(config_path: Path, tool_name: str) -> int:
        """
        Return priority score for this config file.
        Higher score = more likely to be the official MCP config.

        This ensures we pick the correct config when multiple files exist.
        """
        filename = config_path.name.lower()
        tool_lower = tool_name.lower()

        # Known MCP-specific config files (highest priority)
        if tool_lower == "claude" and filename == "claude_desktop_config.json":
            return 100  # Claude Desktop's official MCP config

        # Pattern: *_desktop_config.json (likely MCP-specific)
        if filename.endswith("_desktop_config.json"):
            return 90

        # Files with "mcp" in name (explicitly MCP configs)
        if "mcp" in filename:
            return 80

        # Generic config.json files (lowest priority - usually UI settings)
        if filename == "config.json":
            return 10

        # Other *_config.json files (medium priority)
        if filename.endswith("_config.json"):
            return 50

        return 50  # Default priority

    # Helper function to process config file
    def process_config_file(config_path: Path, tool_name_override: str = None):
        """Process a single config file and add to seen_tools if valid"""
        logger.debug(f"[MCP] Processing: {config_path.name} (full path: {config_path})")

        if not config_path.is_file():
            logger.debug(f"[MCP] Skipped: {config_path.name} (not a file)")
            return

        # CRITICAL: Skip config.json if claude_desktop_config.json exists (Claude Desktop specific)
        # Claude uses config.json for UI settings and claude_desktop_config.json for MCP
        if config_path.name == "config.json" and config_path.parent.name == "Claude":
            correct_config = config_path.parent / "claude_desktop_config.json"
            if correct_config.exists():
                logger.debug(f"[MCP] Skipped: config.json (claude_desktop_config.json exists - using that instead)")
                return

        # Skip if already seen
        if str(config_path) in seen_paths:
            logger.debug(f"[MCP] Skipped: {config_path.name} (already processed)")
            return
        seen_paths.add(str(config_path))

        # Try to parse as JSON
        try:
            config = json.loads(config_path.read_text())

            # Must be a dict with mcpServers key
            if not isinstance(config, dict):
                logger.debug(f"[MCP] Skipped: {config_path.name} (not a dict)")
                return

            if "mcpServers" not in config:
                logger.debug(f"[MCP] Skipped: {config_path.name} (no mcpServers key)")
                return

            # Tool name = override or parent directory name
            if tool_name_override:
                tool_name = tool_name_override
            else:
                tool_name = config_path.parent.name
                # Clean up tool names (remove dots, capitalize)
                if tool_name.startswith('.'):
                    tool_name = tool_name[1:].capitalize()
                # Special handling: .claude folder is Claude Code, not Claude Desktop
                if tool_name == "Claude" and ".claude" in str(config_path):
                    tool_name = "Claude Code"

            # Check if roampal is configured
            connected = "roampal" in config.get("mcpServers", {})

            # Smart deduplication - prefer higher priority configs
            if tool_name in seen_tools:
                existing = seen_tools[tool_name]
                existing_priority = get_config_priority(Path(existing["config_path"]), tool_name)
                new_priority = get_config_priority(config_path, tool_name)

                logger.debug(f"[MCP] Priority check for {tool_name}: existing={Path(existing['config_path']).name} (priority={existing_priority}) vs new={config_path.name} (priority={new_priority})")

                # Keep higher priority config
                if new_priority < existing_priority:
                    logger.debug(f"[MCP] Skipping {config_path.name} - lower priority ({new_priority} < {existing_priority})")
                    return
                elif new_priority == existing_priority:
                    # Tie - prefer connected configs
                    if not (connected and existing["status"] != "connected"):
                        logger.debug(f"[MCP] Skipping {config_path.name} - same priority but not preferred by connection status")
                        return
                else:
                    logger.info(f"[MCP] Replacing {Path(existing['config_path']).name} with {config_path.name} (higher priority: {new_priority} > {existing_priority})")

            seen_tools[tool_name] = {
                "name": tool_name,
                "status": "connected" if connected else "available",
                "config_path": str(config_path)
            }

            logger.info(f"[MCP] Discovered: {tool_name} at {config_path}")

        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not a valid JSON file, skip
            pass
        except Exception as e:
            logger.debug(f"[MCP] Error reading {config_path}: {e}")

    # Scan directories for MCP config files
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Only scan first level subdirectories (e.g., AppData/Roaming/Claude, not deeper)
        try:
            subdirs = [d for d in search_dir.iterdir() if d.is_dir()]
        except PermissionError:
            continue

        for subdir in subdirs:
            for pattern in mcp_config_patterns:
                try:
                    for config_path in subdir.glob(pattern):
                        process_config_file(config_path)
                except Exception as e:
                    logger.debug(f"[MCP] Error globbing {pattern} in {subdir}: {e}")
    # Also scan manually-added custom paths
    custom_paths = load_custom_paths()
    logger.info(f"[MCP] Loading custom paths from {CUSTOM_PATHS_FILE}")
    logger.info(f"[MCP] Loaded {len(custom_paths)} custom paths: {custom_paths}")
    for custom_path_str in custom_paths:
        try:
            custom_path = Path(custom_path_str).expanduser().resolve()
            if custom_path.exists():
                process_config_file(custom_path)
        except Exception as e:
            logger.debug(f"[MCP] Error processing custom path {custom_path_str}: {e}")

    # Convert seen_tools dict to list
    tools = list(seen_tools.values())

    return {"tools": tools}


@router.post("/connect")
async def connect_to_tool(request: ConnectRequest):
    """Add Roampal MCP server to tool's config"""
    config_path = Path(request.config_path)

    # Security: Validate and resolve path
    try:
        config_path = config_path.expanduser().resolve()
        # Must be in user's home directory for security
        home_dir = Path.home().resolve()
        # Use os.sep to prevent path traversal attacks (e.g., /home/alice-evil when home is /home/alice)
        if not str(config_path).startswith(str(home_dir) + os.sep) and config_path != home_dir:
            raise HTTPException(status_code=400, detail="Config file must be in your home directory")
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"[MCP] Invalid path: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")

    # CRITICAL FIX: If connecting to Claude Desktop via config.json, redirect to claude_desktop_config.json
    # Claude Desktop uses config.json for UI settings and claude_desktop_config.json for MCP servers
    if config_path.name == "config.json" and config_path.parent.name == "Claude":
        correct_config = config_path.parent / "claude_desktop_config.json"
        logger.info(f"[MCP] Detected Claude Desktop connection - using {correct_config.name} instead of config.json")
        config_path = correct_config

    # Create config file if it doesn't exist
    if not config_path.exists():
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"mcpServers": {}}, indent=2))
            logger.info(f"[MCP] Created new config file at {config_path}")
        except Exception as e:
            logger.error(f"[MCP] Failed to create config: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create config file: {str(e)}")

    try:
        # Read existing config
        config = json.loads(config_path.read_text())

        # Add Roampal MCP server
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        roampal_cmd = get_roampal_command()

        # Check if roampal already exists (for informational message)
        was_update = "roampal" in config["mcpServers"]
        config["mcpServers"]["roampal"] = roampal_cmd

        # Write back
        # Save custom path for future scans
        save_custom_path(str(config_path))
        config_path.write_text(json.dumps(config, indent=2))

        tool_name = config_path.parent.name
        if was_update:
            logger.info(f"[MCP] Updated Roampal connection for {tool_name}")
            return {
                "success": True,
                "message": f"Updated Roampal connection for {tool_name}. Restart the tool to apply changes."
            }
        else:
            logger.info(f"[MCP] Connected Roampal to {tool_name}")
            return {
                "success": True,
                "message": f"Connected to {tool_name}. Restart the tool to use Roampal memory."
            }

    except Exception as e:
        logger.error(f"[MCP] Error connecting to config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect")
async def disconnect_from_tool(request: ConnectRequest):
    """Remove Roampal from tool's config"""
    config_path = Path(request.config_path)

    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config file not found")

    try:
        config = json.loads(config_path.read_text())
        tool_name = config_path.parent.name

        if "mcpServers" in config and "roampal" in config["mcpServers"]:
            del config["mcpServers"]["roampal"]
            config_path.write_text(json.dumps(config, indent=2))
            logger.info(f"[MCP] Disconnected Roampal from {tool_name}")
            return {"success": True, "message": f"Disconnected from {tool_name}. Restart {tool_name} to remove Roampal access."}
        else:
            return {"success": True, "message": f"Roampal was not connected to {tool_name}"}

    except Exception as e:
        logger.error(f"[MCP] Error disconnecting from config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug-custom-paths")
async def debug_custom_paths():
    """Debug endpoint to check custom paths loading"""
    try:
        custom_paths = load_custom_paths()
        results = []
        for path_str in custom_paths:
            path = Path(path_str).expanduser().resolve()
            results.append({
                "path": path_str,
                "resolved": str(path),
                "exists": path.exists(),
                "is_file": path.is_file() if path.exists() else False
            })
        return {
            "custom_paths_file": str(CUSTOM_PATHS_FILE),
            "file_exists": CUSTOM_PATHS_FILE.exists(),
            "loaded_paths": custom_paths,
            "path_details": results
        }
    except Exception as e:
        return {"error": str(e)}
