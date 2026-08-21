"""Tests for MCP server setup and tool registration."""

from mcp.server.mcpserver.server import MCPServer

from self_rag.mcp.server import mcp


def test_server_is_mcpserver_instance():
    assert isinstance(mcp, MCPServer)


def test_all_tools_registered():
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert {"rag_answer", "retrieve", "server_health"} == tool_names


def test_rag_answer_tool_description():
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "Self-RAG" in tools["rag_answer"].description


def test_retrieve_tool_description():
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert tools["retrieve"].description


def test_server_health_tool_description():
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert tools["server_health"].description
