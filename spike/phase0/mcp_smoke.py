"""Phase 0: start the stylesync MCP server over stdio, list tools and call rewrite_style."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "stylesync.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            print("health:", (await session.call_tool("health", {})).content[0].text)
            for text, task in [
                ("fix(auth): Sửa lỗi đăng nhập do thiếu kiểm tra null trong `get_user`.\n\n"
                 "Co-Authored-By: Claude <noreply@anthropic.com>", "commit"),
                ("Tôi đã kiểm tra lại và lỗi này không còn xuất hiện sau khi merge PR trước.", "comment"),
            ]:
                res = await session.call_tool("rewrite_style", {"text": text, "task": task})
                print(json.dumps(json.loads(res.content[0].text), ensure_ascii=False))


asyncio.run(main())
