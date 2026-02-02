import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession
import os
from dotenv import load_dotenv
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

load_dotenv()

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.api_key = os.getenv("LLM_API_KEY", "sk-68cf8fad5e2e43b5a05dc960aa7e0259")

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        try:
            if "url" in self.config:
                # SSE协议处理
                url = self.config["url"]
                sse_transport = await self.exit_stack.enter_async_context(sse_client(url=url))
                read, write = sse_transport
            elif "command" in self.config:
                # stdio协议处理
                params = StdioServerParameters(
                    command=self.config["command"],
                    args=self.config.get("args", [])
                )
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
                read, write = stdio_transport
            else:
                raise ValueError("Invalid server configuration: must have either 'url' or 'command'")

            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
        except Exception as e:
            print(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                print(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                print(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                print(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class MCPLLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        url = "https://api.deepseek.com/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }
        print(f"#### request to LLM: {messages}")

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload, timeout=3600.0)
                response.raise_for_status()
                data = response.json()
                print(f"#### response from LLM: {data}")
                return data["choices"][0]["message"]["content"]

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            print(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                print(f"Status code: {status_code}")
                print(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: MCPLLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: MCPLLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                print(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                print(f"Executing tool: {tool_call['tool']}")
                print(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                print(
                                    f"Progress: {progress}/{total} ({percentage:.1f}%)"
                                )

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            print(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    print(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "你是一个有帮助的中文助手，可以使用以下工具:\n\n"
                f"{tools_description}\n"
                "根据用户的问题选择合适的工具。"
                "如果不需要使用工具，直接回复即可。\n"
                "如果问题已解决，请只回复'Done'。\n\n"
                "重要提示：当你需要使用工具时，必须严格按照以下JSON格式回复，不要包含其他内容:\n"
                "{\n"
                '    "tool": "工具名称",\n'
                '    "arguments": {\n'
                '        "参数名": "参数值"\n'
                "    }\n"
                "}\n\n"
                "收到工具响应后:\n"
                "1. 将原始数据转换为自然、对话式的响应\n"
                "2. 保持回复简洁但信息丰富\n"
                "3. 聚焦最相关的信息\n"
                "4. 使用用户问题中的适当上下文\n"
                "5. 避免简单重复原始数据\n\n"
                "请仅使用上面明确定义的工具。"
            )

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        print("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})

                    while True:
                        llm_response = self.llm_client.get_response(messages)
                        print("\nAssistant: %s", llm_response)

                        if llm_response== "Done":
                            print(f"\nAssistant: Problem solved.")
                            break

                        # patch to split prefix "```json" and suffix "```"
                        llm_response = llm_response.removeprefix("```json")
                        llm_response = llm_response.removesuffix("```")
                        # tool call
                        result = await self.process_llm_response(llm_response)
                        messages.append({"role": "assistant", "content": llm_response})

                        if result != llm_response: # success
                            print(f"\n#### tool call success: {result}")
                            messages.append({"role": "system", "content": result})
                        else: # fail
                            messages.append({"role": "system", "content": "Tool invocation failed, please consider other tools or alternative methods."})

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()

class MCPClientSession:
    def __init__(self, server_config_path: str):
        self.server_config_path = server_config_path

    # @timeit
    async def start(self):
        config = Configuration()
        server_config = config.load_config(self.server_config_path)

        # === 新增：修正配置文件中的相对路径 ===
        from pathlib import Path
        config_dir = Path(self.server_config_path).parent
        for server_name, server_config_item in server_config["mcpServers"].items():
            if "command" in server_config_item and "args" in server_config_item:
                args = server_config_item["args"]
                if args and len(args) > 0:
                    # 第一个参数通常是脚本文件
                    first_arg = args[0]
                    if first_arg.endswith('.py') and not Path(first_arg).is_absolute():
                        # 转换为绝对路径
                        abs_path = config_dir / first_arg
                        if abs_path.exists():
                            server_config_item["args"][0] = str(abs_path)
                            print(f"[INFO] 修正路径 {server_name}: {first_arg} -> {abs_path}")
                        else:
                            print(f"[WARNING] 文件不存在: {abs_path}")

        self.servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]
        for server in self.servers:
            try:
                await server.initialize()
                print(f"Success to initialize server: {server}")
            except Exception as e:
                print(f"Failed to initialize server: {e}")
                await self._cleanup_servers()
                return

    # @timeit
    async def get_all_tools(self) -> list[Tool]:
        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)
        return all_tools

    # @timeit
    async def stop(self):
        await self._cleanup_servers()
        print(f"Success to cleanup all servers")

    # @timeit
    async def _cleanup_servers(self):
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                print(f"Warning during final cleanup: {e}")

    # @timeit
    async def process_tool_call(self, tool_call_json_str: str) -> tuple[bool, str]:
        import json
        try:
            tool_call = json.loads(tool_call_json_str)
            if not isinstance(tool_call, dict) or "tool" not in tool_call or "arguments" not in tool_call:
                return (False, "tool call json str format error")

            print(f"Executing tool: {tool_call['tool']}")
            print(f"With arguments: {tool_call['arguments']}")

            for server in self.servers:
                tools = await server.list_tools()
                if any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        raw_response = await server.execute_tool(
                            tool_call["tool"], tool_call["arguments"]
                        )

                        if getattr(raw_response, "isError", False):
                            error_msg = f"Tool execution failed: {raw_response}"
                            print(error_msg)
                            return (False, error_msg)

                        content = getattr(raw_response, "content", [])
                        json_str = None
                        for item in content:
                            if getattr(item, "type", "") == "text":
                                json_str = getattr(item, "text", "")
                                break
                        if not json_str:
                            return (False, "No valid JSON content found in tool response")

                        return (True, json_str)

                    except json.JSONDecodeError as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        print(error_msg, exc_info=True)
                        return (False, error_msg)

            return (False, f"No server found with tool: {tool_call['tool']}")

        except json.JSONDecodeError:
            return (False, "JSONDecodeError")

async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = MCPLLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
