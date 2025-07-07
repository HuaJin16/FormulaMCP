import asyncio
import os
import json
import uuid

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import TypedDict
from typing import List

class AgentRepairState(TypedDict):
    input: str
    messages: List[BaseMessage]

class MCPRepair:
    def __init__(self, mcp_server_url="http://127.0.0.1:8000"):
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False,
            num_ctx=16384
        )
        server_config = {
            "default": {
                "url": f"{mcp_server_url}/sse",
                "transport": "sse",
                "timeout": 10.0,
            }
        }
        print(f"Connecting to MCP server at {mcp_server_url}...")
        self.mcp_client = MultiServerMCPClient(server_config)

        self.SYSTEM_PROMPT = """You are an assistant that can call FORMULA tools.
        Only respond with a tool call when the user explicitly asks to **"load" a model file** and provides a filename (e.g., "docs/conflicts/MappingExample.4ml")
        OR when the user explicity asks to **"solve" a model file** and provides provides a filename (e.g., "docs/conflicts/MappingExample.4ml")
        OR when the user explicity asks to **"extract" a solution**" such as "extract 0".
        Always use what the user types after "load" or "solve" or "extract" as the argument.
        
        When loading a model file, do not respond in natural language. Instead, return a JSON object in this format:
        {{
        "tool_calls": [
            {{
                "name": "load_file",
                "arguments": {{
                    "filename": "<filename here>"
                }}
            }}
        ]
        }}
        Do not include any explanation or text outside of this JSON.

        When solving a model file, do not respond in natural language. Instead, return a JSON object in this format:
        {{
        "tool_calls": [
            {{
                "name": "solve_file",
                "arguments": {{
                    "filename": "<filename here>"
                }}
            }}
        ]
        }}
        Do not include any explanation or text outside of this JSON.

        When extracting a solution, do not respond in natural language. Instead, return a JSON object in this format:
        {{
        "tool_calls": [
            {{
                "name": "extract_solution",
                "arguments": {{
                    "id": "<task id>"
                }}
            }}
        ]
        }}
        Do not include any explanation or text outside of this JSON.

        If the user does **not** ask to "load" or "solve" a model file, or "extact" a solution, 
        or if no input is provided, respond as a general-purpose language model.
        Do not attempt to call any tools.
        """
        system_token_count = count_tokens_approximately(self.SYSTEM_PROMPT)
        print(f"[DEBUG] Prompt tokens estimate: {system_token_count}")
    
    async def initialize_graph(self):
        mcp_tools = await self.mcp_client.get_tools()
        print("[DEBUG] Obtaining tools:")
        for tool in mcp_tools:
            print(f"[DEBUG] - {tool.name}: {tool.description}")

        # defines what should happen when the graph visits the llm node
        # accepts the current AgentState
        def llm_node(state: AgentRepairState) -> AgentRepairState:
            print("\n[DEBUG] Entering llm_node...")
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", "{input}")
            ])
            # combines prompt and model into a single pipeline
            chain = prompt | self.llm
            input_text = state["input"]
            print(f"[DEBUG] Sending prompt to LLM: {input_text}")
            # LangChain's Runnable syntax - executes the pipeline with the given input 
            result = chain.invoke({"input": input_text})
            print("[DEBUG] Raw LLM response object:", result)
            response_tokens = count_tokens_approximately(result.content)
            print(f"[DEBUG] LLM response tokens estimate: {response_tokens}")

            # Convert any JSON string tool_calls outputted from the LLM in "content" into tool_calls JSON object
            tool_calls = []
            try:
                parsed = json.loads(result.content)
                if "tool_calls" in parsed:
                    tool_calls = [
                        # manually create LangChain ToolCall objects
                        ToolCall(
                            name=call["name"],
                            args=call["arguments"],
                            id=str(uuid.uuid4())  # tool_call_id
                        )
                        for call in parsed["tool_calls"]
                    ]
                    print("[DEBUG] Parsed tool_calls:", tool_calls)
            except Exception as e:
                print("[DEBUG] No valid JSON tool_call structure found:", e)

            return {
                "input": input_text,
                "messages": state.get("messages", []) + [
                    AIMessage(
                        content="" if tool_calls else result.content,
                        additional_kwargs=getattr(result, "additional_kwargs", {}),
                        tool_calls=tool_calls if tool_calls else None
                    )
                ]
            }
        
        # returns the name of the next node to go to 
        def should_call_tool(state: AgentRepairState) -> str:
            print("\n[DEBUG] Entering should_call_tool...")
            if not state["messages"]:
                return END
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                print("[DEBUG] Tool call found. Routing to 'tools' node.")
                return "tools"
            print("[DEBUG] No tool call found. Routing to END.")
            return END

        graph = StateGraph(AgentRepairState)
        graph.add_node("llm", llm_node)
        # ToolNode is a prebuilt node that automatically reads tool_calls from AgentState["messages"],
        # finds the matching MCP defined tool, and executes it with the given arguments.
        # It then wraps the reutrn value of the tool call in a ToolMessage and adds it to AgentState["messages"]
        graph.add_node("tools", ToolNode(mcp_tools))

        graph.set_entry_point("llm")
        graph.add_conditional_edges("llm", should_call_tool)
        graph.add_edge("tools", END)

        print("[DEBUG] Compiling LangGraph...")
        self.app = graph.compile()
        print("[DEBUG] Graph compiled successfully.")

    async def interactive_chat(self):
        print("Chat session started. Type 'exit' to quit.")
        state = {"input": "", "messages": []}

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Ending chat session...")
                break

            print("[DEBUG] Invoking agent with input:", user_input)
            state["input"] = user_input
            # runs the full state graph - processes each node, follows graph logic, and returns the final AgentState
            result = await self.app.ainvoke(state)
            print("[DEBUG] Agent returned state:", result)
            last_message = result["messages"][-1]
            print(f"\nAgent: {last_message.content}\n")

async def main():
    print("[DEBUG] Initializing MCPRepair...")
    client = MCPRepair()

    print("\n[DEBUG] Initializing LangGraph agent...")
    await client.initialize_graph()

    print("\n[DEBUG] Starting interactive loop...")
    await client.interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())