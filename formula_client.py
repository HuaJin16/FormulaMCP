import asyncio
import nest_asyncio
import json
import uuid
import pprint
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import TypedDict
from typing import List

nest_asyncio.apply()

# state - a way to maintain and track information as the llm flows through the LangGraph system 
class AgentState(TypedDict):
    input: str
    # LangGraph and ToolNode expects list of BaseMessage as "messages"
    messages: List[BaseMessage]
    # Supporting FORMULA documents needed by the llm for lifting
    formula_manual: str
    formula_examples: str

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
        self.FORMULA_GEN_SYSTEM_PROMPT = """You are a FORMULA code generator.
        Your task is to convert:
        - A C source code
        - A natural language description of the C source code
        into a valid .4ml FORMULA model using the supporting FORMULA documents:
        - The FORMULA manual for syntax and semantics (e.g., "docs/Manual.pdf")
        - Reference FORMULA examples (e.g., "docs/examples/MappingExample.4ml"...)

        **Instructions:**
        - DO NOT output anything else except valid FORMULA code
        - DO NOT include explanations, markdown, comments, or formatting tips.
        - The output must contain with a valid domain and, model or partial model declaration.
        - Assume the user prompt includes the relevant C code and description.
        """
        # use double curly so LangChain knows it isn't a variable that needs to be substittued
        self.SYSTEM_PROMPT = """You are an assistant that can call FORMULA tools.
        Only respond with a tool call when the user explicitly asks to **"load" a model file** and provides a filename (e.g., "docs/examples/MappingExample.4ml").
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

        If the user does **not** ask to "load" a model file, or if no filename is given, respond normally as a general-purpose language model.
        Do not attempt to call any tools.
        """

    async def initialize_graph(self):
        mcp_tools = await self.mcp_client.get_tools()
        print("[DEBUG] Obtaining tools:")
        for tool in mcp_tools:
            print(f"[DEBUG] - {tool.name}: {tool.description}")

        # loads supporting FORMULA documents into AgentState
        def load_docs_node(state: AgentState) -> AgentState:
            print("\n[DEBUG] Entering load_docs_node...")
            # load Manual.pdf
            loader = PyMuPDFLoader(
                "docs/Manual.pdf", 
                mode = "single"
            )
            manual_doc = loader.load()
            print(f"[DEBUG] Loaded {len(manual_doc)} pages.")
            pprint.pp(manual_doc[0].metadata)

            # load formula examples
            example_files = ["BatteryExample.4ml", "Arith.4ml", "MappingExample.4ml"]
            formula_example_docs = []
            for filename in example_files:
                loader = TextLoader(os.path.join("docs/examples/", filename))
                example_doc = loader.load()
                formula_example_docs.append((filename, example_doc[0].page_content))
            
            for filename, content in formula_example_docs:
                print(f"\n[DEBUG] --- Example File: {filename} ---")
                print(content)

            return {
                "input": state["input"],
                "formula_manual": manual_doc[0].page_content,
                "formula_examples": formula_example_docs
            }
        
        # creates and writes a FORMULA model 
        def FormulaGen_node(state: AgentState) -> AgentState:
            print("\n[DEBUG] Entering FormulaGen_node (generating formula model)...")
            # print(f"\n[DEBUG] Input: {state["input"]}")
            # print(f"\n[DEBUG] Manual\n: {state["formula_manual"]}")
            # print(f"\n[DEBUG] Examples\n {state["formula_examples"]}")
            full_prompt = f"""
            [C SOURCE CODE + NATURAL LANGUAGE DESCRIPTION]\n
            {state["input"]}\n

            [FORMULA MANUAL]\n
            {state["formula_manual"]}\n

            [FORMULA EXAMPLES]\n
            {state["formula_examples"]}

            [INSTRUCTION]
            Using the above materials, generate a valid .4ml FORMULA model that captures the logic of the C source code.
            Do not explain or summarize the manual or examples. Only return valid FORMULA code.
            """

            # creates a prompt template and composes it in a single pipeline with the LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.FORMULA_GEN_SYSTEM_PROMPT),
                ("human", "{input}")
            ])
            chain = prompt | self.llm

            print("[DEBUG] Sending FORMULA generation prompt to model...")
            print(f"PROMPT: {full_prompt}")
            result = chain.invoke({"input": full_prompt})
            print(f"[DEBUG] FORMULA MODEL: {result}")
            token_count = count_tokens_approximately(full_prompt)
            print(f"[TOKENS]: {token_count}")

            return " "
        
        # defines what should happen when the graph visits the llm node
        def model_loader_node(state: AgentState) -> AgentState:
            print("\n[DEBUG] Entering model_loader_node...")
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
                        tool_calls=tool_calls if tool_calls else []
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
        graph.add_node("load_docs", load_docs_node)
        graph.add_node("FormulaGen", FormulaGen_node)
        graph.add_node("model_loader", model_loader_node)
        # ToolNode is a prebuilt node that automatically reads tool_calls from AgentState["messages"],
        # finds the matching MCP defined tool, and executes it with the given arguments.
        # It then wraps the reutrn value of the tool call in a ToolMessage and adds it to AgentState["messages"]
        graph.add_node("tools", ToolNode(mcp_tools))

        graph.set_entry_point("load_docs")
        graph.add_edge("load_docs", "FormulaGen")
        graph.add_edge("FormulaGen", "model_loader")
        graph.add_conditional_edges("model_loader", should_call_tool)
        graph.add_edge("tools", END)

        print("[DEBUG] Compiling LangGraph...")
        self.app = graph.compile()
        print("[DEBUG] Graph compiled successfully.")

    def read_multiline_input(self) -> str:
        print("Paste your C code and natural language description. End with an \"END\" on a new line")
        lines = []
        while True:
            try:
                line = input("\nYou: ")
                lines.append(line)
                if line.strip() == "END":
                    break
            except EOFError:
                break
        return "\n".join(lines)

    async def interactive_chat(self) -> str:
        print("Chat session started. Type 'exit' to quit.")
        while True:
            user_input = self.read_multiline_input()
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
