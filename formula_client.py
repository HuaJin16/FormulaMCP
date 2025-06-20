import asyncio
import nest_asyncio
import json
import uuid

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from typing_extensions import TypedDict
from typing import List

nest_asyncio.apply()

# state - a way to maintain and track information as the llm flows through the LangGraph system 
class AgentState(TypedDict):
    input: str
    # LangGraph and ToolNode expects list of BaseMessage as "messages"
    messages: List[BaseMessage]

class MCPClient:
    def __init__(self, mcp_server_url="http://127.0.0.1:8000"):
        # initialize ollama
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False
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
        # use double curly so LangChain knows it isn't a variable that needs to be substittued
        self.SYSTEM_PROMPT = """You are an assistant that can call FORMULA tools.
        If the user asks to "load" a model file, use the `load_file` tool. The tool takes a filename like 'examples/MappingExample.4ml'.
        You should only respond with a tool call when loading a file.
        If the user wants to load a model file, do not respond in plain text. Instead, respond with a JSON object like this:
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
        Do not include any explanation or natural language. Only output this JSON tool call.
        """

    async def initialize_graph(self):
        mcp_tools = await self.mcp_client.get_tools()
        print("[DEBUG] Obtaining tools:")
        for tool in mcp_tools:
            print(f"[DEBUG] - {tool.name}: {tool.description}")

        # defines what should happen when the graph visits the llm node
        # accepts the current AgentState
        def llm_node(state: AgentState) -> AgentState:
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
            print("[DEBUG] LLM content:", result.content)
            print("[DEBUG] LLM tool_calls:", getattr(result, "tool_calls", None))
            print("[DEBUG] LLM additional_kwargs:", getattr(result, "additional_kwargs", {}))

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
        def should_call_tool(state: AgentState) -> str:
            print("\n[DEBUG] Entering should_call_tool...")
            if not state["messages"]:
                return END
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                print("[DEBUG] Tool call found. Routing to 'tools' node.")
                return "tools"
            print("[DEBUG] No tool call found. Routing to END.")
            return END

        graph = StateGraph(AgentState)
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

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Ending chat session...")
                break

            print("[DEBUG] Invoking agent with input:", user_input)
            state = {"input": user_input}
            # runs the full state graph - processes each node, follows graph logic, and returns the final AgentState
            result = await self.app.ainvoke(state)
            print("[DEBUG] Agent returned state:", result)
            last_message = result["messages"][-1]
            print(f"\nAgent: {last_message.content}\n")

async def main():
    print("[DEBUG] Initializing MCPClient...")
    client = MCPClient()

    print("\n[DEBUG] Initializing LangGraph agent...")
    await client.initialize_graph()

    print("\n[DEBUG] Starting interactive loop...")
    await client.interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())
