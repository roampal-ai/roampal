"""
MCP Client Manager - v0.2.5
Connects to external MCP tool servers and makes their tools available to Ollama/LM Studio

Architecture:
- Spawns MCP servers as subprocesses via stdio transport
- Discovers tools via list_tools() on each server
- Converts MCP tools to OpenAI function format for Ollama
- Routes tool calls to the correct server
"""
import asyncio
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .config import MCPServerConfig, load_mcp_config, save_mcp_config

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents a tool from an MCP server"""
    name: str  # Prefixed name (e.g., "filesystem_read_file")
    original_name: str  # Original name from server
    server_name: str  # Which server this tool belongs to
    description: str
    input_schema: dict  # JSON Schema for parameters


@dataclass
class MCPServerConnection:
    """Represents an active connection to an MCP server"""
    config: MCPServerConfig
    process: Optional[subprocess.Popen] = None
    tools: List[MCPTool] = field(default_factory=list)
    status: str = "disconnected"  # connected, disconnected, error
    last_error: Optional[str] = None
    _request_id: int = 0

    def next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id


class MCPClientManager:
    """
    Manages connections to external MCP tool servers.

    Usage:
        manager = MCPClientManager(data_path)
        await manager.connect_all()

        # Get tools for Ollama
        tools = manager.get_all_tools_openai_format()

        # Execute a tool call
        result = await manager.execute_tool("filesystem_read_file", {"path": "/etc/hosts"})
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.servers: Dict[str, MCPServerConnection] = {}
        self.tool_to_server: Dict[str, str] = {}  # tool_name -> server_name
        self._initialized = False

    async def initialize(self):
        """Load config and connect to all enabled servers"""
        if self._initialized:
            return

        configs = load_mcp_config(self.data_path)
        for name, config in configs.items():
            self.servers[name] = MCPServerConnection(config=config)

        await self.connect_all()
        self._initialized = True
        logger.info(f"MCPClientManager initialized with {len(self.servers)} servers")

    async def connect_all(self):
        """Connect to all enabled servers"""
        tasks = []
        for name, conn in self.servers.items():
            if conn.config.enabled:
                tasks.append(self._connect_server(name))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_server(self, name: str) -> bool:
        """Connect to a single MCP server"""
        conn = self.servers.get(name)
        if not conn:
            return False

        config = conn.config
        logger.info(f"[MCP] Connecting to server '{name}': {config.command} {' '.join(config.args)}")

        try:
            # Build environment
            env = dict(subprocess.os.environ)
            env.update(config.env)

            # v0.2.5 Security: Always use shell=False to prevent command injection
            # Use shutil.which() to resolve command path (works on Windows for npx/npm/node/uvx)
            command = config.command
            if sys.platform == 'win32':
                # Resolve full path to avoid needing shell=True
                resolved = shutil.which(command)
                if resolved:
                    command = resolved
                else:
                    # Try with .cmd extension for npm scripts
                    resolved = shutil.which(f"{command}.cmd")
                    if resolved:
                        command = resolved

            # Start the process - always use list args and shell=False for security
            process = subprocess.Popen(
                [command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=False,  # Security: Never use shell=True
                bufsize=0  # Unbuffered for real-time communication
            )

            conn.process = process
            conn.status = "connecting"

            # Initialize the MCP connection
            init_result = await self._send_request(name, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "Roampal",
                    "version": "0.2.5"
                }
            })

            if init_result is None:
                raise Exception("Initialize request failed")

            # Send initialized notification
            await self._send_notification(name, "notifications/initialized", {})

            # Discover tools
            tools_result = await self._send_request(name, "tools/list", {})
            if tools_result and "tools" in tools_result:
                conn.tools = []
                for tool_data in tools_result["tools"]:
                    tool = MCPTool(
                        name=f"{name}_{tool_data['name']}",  # Prefix with server name
                        original_name=tool_data['name'],
                        server_name=name,
                        description=tool_data.get('description', ''),
                        input_schema=tool_data.get('inputSchema', {})
                    )
                    conn.tools.append(tool)
                    self.tool_to_server[tool.name] = name

            conn.status = "connected"
            logger.info(f"[MCP] Connected to '{name}' with {len(conn.tools)} tools")
            return True

        except Exception as e:
            conn.status = "error"
            conn.last_error = str(e)
            logger.error(f"[MCP] Failed to connect to '{name}': {e}")
            if conn.process:
                conn.process.terminate()
                conn.process = None
            return False

    async def _send_request(self, server_name: str, method: str, params: dict) -> Optional[dict]:
        """Send a JSON-RPC request to an MCP server"""
        conn = self.servers.get(server_name)
        if not conn or not conn.process:
            logger.error(f"[MCP] No connection/process for {server_name}")
            return None

        # Check if process is still alive
        if conn.process.poll() is not None:
            exit_code = conn.process.returncode
            # Try to read stderr for error details
            stderr_output = ""
            try:
                stderr_output = conn.process.stderr.read().decode() if conn.process.stderr else ""
            except:
                pass
            logger.error(f"[MCP] Server '{server_name}' process died (exit code {exit_code}). stderr: {stderr_output[:500]}")
            conn.status = "error"
            conn.last_error = f"Process exited with code {exit_code}"
            return None

        request_id = conn.next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        try:
            # Write request
            request_line = json.dumps(request) + "\n"
            logger.debug(f"[MCP] Sending to {server_name}: {method}")
            conn.process.stdin.write(request_line.encode())
            conn.process.stdin.flush()

            # Read response (with timeout)
            loop = asyncio.get_event_loop()
            response_line = await asyncio.wait_for(
                loop.run_in_executor(None, conn.process.stdout.readline),
                timeout=30.0
            )

            if not response_line:
                logger.error(f"[MCP] Empty response from {server_name} for {method}")
                return None

            response = json.loads(response_line.decode())

            if "error" in response:
                error_msg = response['error']
                logger.error(f"[MCP] Error from {server_name}: {error_msg}")
                conn.last_error = str(error_msg)
                return None

            return response.get("result")

        except asyncio.TimeoutError:
            logger.error(f"[MCP] Timeout (30s) waiting for response from {server_name} for {method}")
            return None
        except Exception as e:
            logger.error(f"[MCP] Request failed for {server_name}: {e}")
            return None

    async def _send_notification(self, server_name: str, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)"""
        conn = self.servers.get(server_name)
        if not conn or not conn.process:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        try:
            notification_line = json.dumps(notification) + "\n"
            conn.process.stdin.write(notification_line.encode())
            conn.process.stdin.flush()
        except Exception as e:
            logger.error(f"[MCP] Failed to send notification to {server_name}: {e}")

    async def disconnect_server(self, name: str):
        """Disconnect from a server"""
        conn = self.servers.get(name)
        if not conn:
            return

        # Remove tools from mapping
        for tool in conn.tools:
            self.tool_to_server.pop(tool.name, None)

        # Terminate process
        if conn.process:
            try:
                conn.process.terminate()
                conn.process.wait(timeout=5)
            except:
                conn.process.kill()
            conn.process = None

        conn.tools = []
        conn.status = "disconnected"
        logger.info(f"[MCP] Disconnected from '{name}'")

    async def disconnect_all(self):
        """Disconnect from all servers"""
        for name in list(self.servers.keys()):
            await self.disconnect_server(name)

    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools from connected servers"""
        tools = []
        for conn in self.servers.values():
            if conn.status == "connected":
                tools.extend(conn.tools)
        return tools

    def get_all_tools_openai_format(self) -> List[dict]:
        """Get all tools in OpenAI function calling format for Ollama/LM Studio"""
        tools = []
        for tool in self.get_all_tools():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": f"[{tool.server_name}] {tool.description}",
                    "parameters": tool.input_schema
                }
            })
        return tools

    def is_external_tool(self, tool_name: str) -> bool:
        """Check if a tool name belongs to an external MCP server"""
        return tool_name in self.tool_to_server

    async def execute_tool(self, tool_name: str, arguments: dict) -> Tuple[bool, Any]:
        """
        Execute a tool on the appropriate MCP server.

        Returns:
            Tuple of (success: bool, result: Any)
        """
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            return False, f"Unknown tool: {tool_name}"

        conn = self.servers.get(server_name)
        if not conn:
            return False, f"Server '{server_name}' not found"

        if conn.status != "connected":
            error_detail = conn.last_error or "No connection"
            return False, f"Server '{server_name}' not connected: {error_detail}"

        # Find the original tool name
        original_name = None
        for tool in conn.tools:
            if tool.name == tool_name:
                original_name = tool.original_name
                break

        if not original_name:
            return False, f"Tool '{tool_name}' not found on server '{server_name}'"

        logger.info(f"[MCP] Executing {tool_name} on {server_name} with args: {list(arguments.keys())}")

        try:
            result = await self._send_request(server_name, "tools/call", {
                "name": original_name,
                "arguments": arguments
            })

            if result is None:
                # Get more specific error from connection
                error_detail = conn.last_error or "No response from server (process may have crashed)"
                return False, f"Tool execution failed: {error_detail}"

            # Extract content from MCP response
            content = result.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                # Return the first text content
                for item in content:
                    if item.get("type") == "text":
                        return True, item.get("text", "")
                # If no text, return the raw content
                return True, content

            return True, result

        except Exception as e:
            logger.error(f"[MCP] Tool execution failed: {e}")
            return False, str(e)

    # === Server Management API ===

    async def add_server(self, name: str, command: str, args: List[str], env: dict = None) -> bool:
        """Add a new MCP server configuration"""
        if name in self.servers:
            logger.warning(f"[MCP] Server '{name}' already exists")
            return False

        config = MCPServerConfig(
            name=name,
            command=command,
            args=args,
            env=env or {},
            enabled=True
        )

        self.servers[name] = MCPServerConnection(config=config)

        # Save to disk
        all_configs = {n: c.config for n, c in self.servers.items()}
        save_mcp_config(self.data_path, all_configs)

        # Connect
        await self._connect_server(name)
        return True

    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server"""
        if name not in self.servers:
            return False

        await self.disconnect_server(name)
        del self.servers[name]

        # Save to disk
        all_configs = {n: c.config for n, c in self.servers.items()}
        save_mcp_config(self.data_path, all_configs)

        return True

    async def toggle_server(self, name: str, enabled: bool) -> bool:
        """Enable or disable a server"""
        conn = self.servers.get(name)
        if not conn:
            return False

        conn.config.enabled = enabled

        if enabled and conn.status == "disconnected":
            await self._connect_server(name)
        elif not enabled and conn.status == "connected":
            await self.disconnect_server(name)

        # Save to disk
        all_configs = {n: c.config for n, c in self.servers.items()}
        save_mcp_config(self.data_path, all_configs)

        return True

    async def test_connection(self, name: str) -> Tuple[bool, str]:
        """Test connection to a server"""
        conn = self.servers.get(name)
        if not conn:
            return False, "Server not found"

        if conn.status == "connected":
            return True, f"Connected with {len(conn.tools)} tools"

        # Try to reconnect
        success = await self._connect_server(name)
        if success:
            return True, f"Connected with {len(conn.tools)} tools"
        else:
            return False, conn.last_error or "Connection failed"

    def get_server_status(self) -> List[dict]:
        """Get status of all servers"""
        return [
            {
                "name": name,
                "command": conn.config.command,
                "args": conn.config.args,
                "enabled": conn.config.enabled,
                "status": conn.status,
                "toolCount": len(conn.tools),
                "tools": [t.name for t in conn.tools],
                "lastError": conn.last_error
            }
            for name, conn in self.servers.items()
        ]


# Global instance (initialized in main.py)
_mcp_manager: Optional[MCPClientManager] = None


def get_mcp_manager() -> Optional[MCPClientManager]:
    """Get the global MCP manager instance"""
    return _mcp_manager


def set_mcp_manager(manager: MCPClientManager):
    """Set the global MCP manager instance"""
    global _mcp_manager
    _mcp_manager = manager
