"""
MCP Servers API Router - v0.2.5
Endpoints for managing external MCP tool server connections
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class AddServerRequest(BaseModel):
    name: str
    command: str
    args: List[str] = []
    env: dict = {}


class ToggleServerRequest(BaseModel):
    enabled: bool


@router.get("/servers")
async def list_servers(request: Request):
    """List all configured MCP servers with their status"""
    mcp_manager = getattr(request.app.state, 'mcp_manager', None)
    if not mcp_manager:
        return {"servers": [], "message": "MCP manager not initialized"}

    return {"servers": mcp_manager.get_server_status()}


@router.post("/servers")
async def add_server(request: Request, body: AddServerRequest):
    """Add a new MCP server configuration"""
    mcp_manager = getattr(request.app.state, 'mcp_manager', None)
    if not mcp_manager:
        raise HTTPException(500, "MCP manager not initialized")

    if not body.name or not body.command:
        raise HTTPException(400, "Name and command are required")

    success = await mcp_manager.add_server(
        name=body.name,
        command=body.command,
        args=body.args,
        env=body.env
    )

    if not success:
        raise HTTPException(400, f"Server '{body.name}' already exists")

    # Get the new server's status
    servers = mcp_manager.get_server_status()
    new_server = next((s for s in servers if s["name"] == body.name), None)

    return {
        "success": True,
        "server": new_server
    }


@router.delete("/servers/{name}")
async def remove_server(request: Request, name: str):
    """Remove an MCP server"""
    mcp_manager = getattr(request.app.state, 'mcp_manager', None)
    if not mcp_manager:
        raise HTTPException(500, "MCP manager not initialized")

    success = await mcp_manager.remove_server(name)
    if not success:
        raise HTTPException(404, f"Server '{name}' not found")

    return {"success": True, "message": f"Server '{name}' removed"}


@router.post("/servers/{name}/toggle")
async def toggle_server(request: Request, name: str, body: ToggleServerRequest):
    """Enable or disable an MCP server"""
    mcp_manager = getattr(request.app.state, 'mcp_manager', None)
    if not mcp_manager:
        raise HTTPException(500, "MCP manager not initialized")

    success = await mcp_manager.toggle_server(name, body.enabled)
    if not success:
        raise HTTPException(404, f"Server '{name}' not found")

    return {"success": True, "enabled": body.enabled}


@router.post("/servers/{name}/test")
async def test_server_connection(request: Request, name: str):
    """Test connection to an MCP server"""
    mcp_manager = getattr(request.app.state, 'mcp_manager', None)
    if not mcp_manager:
        raise HTTPException(500, "MCP manager not initialized")

    success, message = await mcp_manager.test_connection(name)

    return {
        "success": success,
        "message": message
    }


@router.post("/servers/{name}/reconnect")
async def reconnect_server(request: Request, name: str):
    """Reconnect to an MCP server"""
    mcp_manager = getattr(request.app.state, 'mcp_manager', None)
    if not mcp_manager:
        raise HTTPException(500, "MCP manager not initialized")

    if name not in mcp_manager.servers:
        raise HTTPException(404, f"Server '{name}' not found")

    # Disconnect first if connected
    await mcp_manager.disconnect_server(name)

    # Reconnect
    success = await mcp_manager._connect_server(name)

    servers = mcp_manager.get_server_status()
    server = next((s for s in servers if s["name"] == name), None)

    return {
        "success": success,
        "server": server
    }


@router.get("/tools")
async def list_all_tools(request: Request):
    """List all available tools from connected MCP servers"""
    mcp_manager = getattr(request.app.state, 'mcp_manager', None)
    if not mcp_manager:
        return {"tools": [], "message": "MCP manager not initialized"}

    tools = []
    for tool in mcp_manager.get_all_tools():
        tools.append({
            "name": tool.name,
            "originalName": tool.original_name,
            "server": tool.server_name,
            "description": tool.description
        })

    return {"tools": tools, "count": len(tools)}


@router.get("/popular")
async def get_popular_servers():
    """Get list of popular MCP servers for quick setup"""
    from modules.mcp_client.config import POPULAR_SERVERS
    return {"servers": POPULAR_SERVERS}
