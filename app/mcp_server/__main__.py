import os

from app.mcp_server.server import mcp

mcp.run(
    "streamable-http",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
    stateless_http=True,
)
