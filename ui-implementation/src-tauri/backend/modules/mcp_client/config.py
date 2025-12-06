"""
MCP Server Configuration Management
Handles loading/saving server configurations from mcp_servers.json
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> 'MCPServerConfig':
        return cls(
            name=name,
            command=data.get('command', ''),
            args=data.get('args', []),
            env=data.get('env', {}),
            enabled=data.get('enabled', True)
        )


def get_config_path(data_path: Path) -> Path:
    """Get the path to mcp_servers.json"""
    return data_path / "mcp_servers.json"


def load_mcp_config(data_path: Path) -> Dict[str, MCPServerConfig]:
    """Load MCP server configurations from file"""
    config_path = get_config_path(data_path)

    if not config_path.exists():
        logger.info(f"No MCP config found at {config_path}, starting with empty config")
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        servers = {}
        for name, server_data in data.get('servers', {}).items():
            servers[name] = MCPServerConfig.from_dict(name, server_data)

        logger.info(f"Loaded {len(servers)} MCP server configs from {config_path}")
        return servers

    except Exception as e:
        logger.error(f"Failed to load MCP config: {e}")
        return {}


def save_mcp_config(data_path: Path, servers: Dict[str, MCPServerConfig]) -> bool:
    """Save MCP server configurations to file"""
    config_path = get_config_path(data_path)

    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Build config structure
        config = {
            "servers": {
                name: {
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                    "enabled": server.enabled
                }
                for name, server in servers.items()
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved {len(servers)} MCP server configs to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save MCP config: {e}")
        return False


POPULAR_SERVERS = []
