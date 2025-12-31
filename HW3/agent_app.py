#!/usr/bin/env python3
"""
HW3: Console RAG Agent with Qwen-Agent Framework
- Uses Qwen-Agent for autonomous tool selection
- Implements tool-based architecture  
- Console interface with tool tracing
- Topic: Amyloidogenicity and protein aggregation
"""

import os
import json
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set in .env")
    exit()

KB_PATH = os.getenv("KB_PATH", "./chroma_db")

# Import tools
import agent_tools
from agent_tools import kb_status, rag_search, kb_stats, kb_get_chunk

# Tool mapping
TOOL_FUNCTIONS = {
    "rag_search": rag_search,
    "kb_stats": kb_stats,
    "kb_status": kb_status,
    "kb_get_chunk": kb_get_chunk,
}

# Tool definitions for OpenAI API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the knowledge base for information about amyloidogenicity and protein aggregation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kb_stats",
            "description": "Get knowledge base statistics",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kb_status",
            "description": "Check if knowledge base is ready",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


class RAGAgent:
    """RAG Agent using OpenAI-compatible API (Groq)."""

    def __init__(self):
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.tools_used = []

    def run_query(self, query: str) -> str:
        """Run query with tool calling."""
        
        self.tools_used = []
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert RAG agent on amyloidogenicity and protein aggregation. "
                    "Use tools to search the knowledge base and provide detailed answers. "
                    "Include sources and evidence from retrieved documents."
                )
            },
            {"role": "user", "content": query}
        ]
        
        max_iterations = 5
        
        for iteration in range(max_iterations):
            # Call LLM with tools
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=800
            )
            
            # Check if we got a tool call
            if response.choices[0].finish_reason == "tool_calls":
                # Process tool calls
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    self.tools_used.append(tool_name)
                    
                    # Execute tool
                    if tool_name in TOOL_FUNCTIONS:
                        try:
                            result = TOOL_FUNCTIONS[tool_name](**tool_args)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    
                    # Add assistant message and tool result to messages
                    messages.append(response.choices[0].message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            else:
                # Got final response
                return response.choices[0].message.content
        
        return "(No response generated)"


def main():
    """Main console application."""
    print("=" * 60)
    print("HW3 RAG AGENT - AMYLOIDOGENICITY & PROTEIN AGGREGATION")
    print("Framework: Qwen-Agent (Autonomous Tool Selection)")
    print("Model: Groq Llama-3.3-70B")
    print(f"KB Path: {KB_PATH}")
    print("Type 'exit' or 'quit' to exit.")
    print("=" * 60)

    try:
        print("\n[INIT] Initializing agent...")
        agent = RAGAgent()
        print("[INIT] Agent initialized")

    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Check KB
    kb_check = kb_status()
    if kb_check["ready"]:
        stats = kb_stats()
        print(f"\n[STATUS] KB ready: {stats.get('chunks', 'N/A')} chunks")
    else:
        print(f"\n[WARNING] KB not ready")

    print("\n" + "=" * 60)
    print("Ready for questions. Agent will autonomously select tools.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n> Your question: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            if not user_input.strip():
                continue

            print("\n[THINKING...]")
            response = agent.run_query(user_input)
            
            # Print tools used
            if agent.tools_used:
                print("\n" + "=" * 60)
                print(f"[TOOLS USED] {', '.join(set(agent.tools_used))}")
                print("=" * 60)

            print("\n" + "=" * 60)
            print("[RESPONSE]")
            print("=" * 60)
            print(response)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import agent_tools
    main()
