#claude choosing tools 
import os
import requests
from typing import Dict, List, Any, Optional
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5050")
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229")
DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "4096"))
DEFAULT_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "20"))
class ClaudeMCPClient:
    def __init__(
        self,
        api_key: str = CLAUDE_API_KEY,
        model: str = DEFAULT_MODEL,
        enable_tools: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        request_timeout: int = DEFAULT_TIMEOUT,
        server_url: Optional[str] = None,
        progress_callback=None,
        **kwargs,
    ):
        if not api_key:
            raise ValueError("CLAUDE_API_KEY is missing.")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.mcp_server_url = (server_url or MCP_SERVER_URL).rstrip("/")
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self.tools_free_chat = [
            {
                "name": "research_lookup",
                "description": (
                    "Call this tool ONLY if the user asks to search for or ingest a research paper "
                    "(e.g., 'find a paper on...', 'load the paper titled...', 'can you get the paper...').\n"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paper_title": {"type": "string"},
                        "question": {"type": "string"}
                    },
                    "required": ["paper_title"]
                },
            },
            {
                "name": "web_search",
                "description": (
                    "Use ONLY when user asks for current or up-to-date information  and you think it requires a web search but if you can answer from your own knowledge, do that instead. "
                    "(latest news, current price, recent events). Uses SerpAPI."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                },
            }
        ]
    def _normalize_history(self, history):
        if not history:
            return []
        out = []
        for h in history:
            role = h.get("role")
            content = h.get("content")

            if role in ("user", "assistant") and isinstance(content, str):
                out.append({"role": role, "content": content})
        return out
    def _call_claude(self, message, history=None):
        system_prompt = f"""
You are a strict state-aware orchestrator for a research assistant app.

Your job is to decide which tool to use:
   - Allowed actions:
       • direct_answer using your own knowledge
       • web_search tool
       • research_lookup tool
   - If the user explicitly asks to search for or ingest a new research paper (e.g., “find paper on…”, “load the paper titled…”):
        CALL research_lookup.
   - Do NOT call research_lookup just because a paper is mentioned.
   - If user asks "current", "latest", "recent", "today", "now" or you think a web search can better answer the question:
        CALL web_search.
   - Otherwise: answer directly. direct answers are allowed.
"""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": self._normalize_history(history) + [
                {"role": "user", "content": message}
            ],
            "tools": self.tools_free_chat,
            "tool_choice": {"type": "auto"},
        }
        resp = requests.post(
            CLAUDE_API_URL,
            headers=self.headers,
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()
    def send_message(self, message: str, conversation_history=None):
        result = self._call_claude(message, history=conversation_history)
        
        for block in result.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_name = block["name"]
                tool_input = block["input"]
                result["__tool_name"] = tool_name
                result["__tool_payload"] = tool_input
                return result
        return result