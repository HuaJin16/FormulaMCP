import asyncio
import os
import json
import uuid

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
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

        self.SYSTEM_TOOL_PROMPT = """You are an assistant that can call FORMULA tools.
        When the user asks to **provide a solution for** a model file, you must perform ONLY one of these steps and in order:
        - ONLY if the value of "previous_tool:" is empty, respond ONLY with a tool call named "load_file" with argument:
          "filename" = the user-provided file path (e.g., "docs/conflicts/MappingExample.4ml").
        - ONLY if "previous_tool" is "load_file", respond ONLY with a tool call named "solve_file" with argument:
          "filename" = the same user-provided file path (e.g., "docs/conflicts/MappingExample.4ml").
        - ONLY if "previous_tool" is "solve_file", you will receive its output as a "tool_result:": "content". 
          Look at content to find a "solve task with Id <task_id>".
          Then respond ONLY with a tool call named `"extract_solution"` with argument: `"id"` = the `"task_id"` you found.

        For each step, respond ONLY with a JSON object in this format:
        {{
        "tool_calls": [
            {{
                "name": "load_file" | "solve_file" | "extract_solution",
                "arguments": {{ ... }}
            }}
        ]
        }}
        Do NOT include any explanation or text outside of this JSON.
        If the user does NOT ask to "solve" a model file, or if no argument is provided, do NOT make any tool calls. 
        """
        system_token_count = count_tokens_approximately(self.SYSTEM_TOOL_PROMPT)
        print(f"[DEBUG] Prompt tokens estimate: {system_token_count}")
    
    async def initialize_graph(self):
        mcp_tools = await self.mcp_client.get_tools()
        print("[DEBUG] Obtaining tools:")
        for tool in mcp_tools:
            print(f"[DEBUG] - {tool.name}: {tool.description}")

        # creates tool_calls for loading or solving a model, or extracting a task solution
        def tool_planner_node(state: AgentRepairState) -> AgentRepairState:
            print("\n[DEBUG] Entering tool_planner_node...")

            # extract the last tool message for the solve command
            last_tool_call = ""
            last_tool_result = ""
            if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
                last_tool_call = state["messages"][-1].name
                print(f"[DEBUG] Last tool '{last_tool_call}'")
                if last_tool_call == "solve_file":
                    last_tool_result = state["messages"][-1].content

            # construct human prompt
            input_text = state["input"]
            human_prompt = "\n".join([
                "[USER PROMPT]",
                input_text,
                "",
                "[PREVIOUS TOOL CALL]",
                f"previous_tool = {last_tool_call}",
                "",
                "[PREVIOUS TOOL RESULT]",
                f"tool_result = {last_tool_result}\n",
            ])
            print(f"[DEBUG] Human prompt:\n{human_prompt}")

            # construct full prompt
            prompt_parts = [
                ("system", self.SYSTEM_TOOL_PROMPT),
                ("human", "{input}")
            ]
            prompt = ChatPromptTemplate.from_messages(prompt_parts)
            
            # invoke llm
            chain = prompt | self.llm
            result = chain.invoke({"input": human_prompt})
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
                        tool_calls=tool_calls if tool_calls else []
                    )
                ]
            }
        
        # invoke the tool if tool_calls is found, else END
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
        
        # loops back to tool planner until the execute tool call is made
        def should_continue_tool_call(state: AgentRepairState) -> str:
            print("\n[DEBUG] Entering should_continue_tool_call...")
            if not state["messages"]:
                return END
            last_tool_message = state["messages"][-1]
            if isinstance(last_tool_message, ToolMessage):
                last_tool = last_tool_message.name
                if last_tool == "load_file" or last_tool== "solve_file":
                    print(f"[DEBUG] {last_tool} called. Routing to 'tool_planner' node...")
                    return "tool_planner"
            print("\n[DEBUG] Execute tool called. Routing to 'END'...")
            return END

        graph = StateGraph(AgentRepairState)
        graph.add_node("tool_planner", tool_planner_node)
        graph.add_node("tools", ToolNode(mcp_tools))

        graph.set_entry_point("tool_planner")
        graph.add_conditional_edges("tool_planner", should_call_tool)
        graph.add_conditional_edges("tools", should_continue_tool_call)

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