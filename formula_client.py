import asyncio
import nest_asyncio
import json
import uuid
import os
import shutil

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_community.document_loaders import TextLoader
from langchain_core.messages.utils import count_tokens_approximately
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from typing_extensions import TypedDict
from typing import List, Dict

nest_asyncio.apply()

# tracks shared context as the agent flows through the LangGraph graph
class AgentState(TypedDict):
    input: str
    # LangGraph and ToolNode expects list of BaseMessage as "messages"
    messages: List[BaseMessage]
    models_results: List[Dict[str, str]]
    iterations: int
    latest_model: str

class MCPClient:
    def __init__(self, mcp_server_url="http://127.0.0.1:8000"):
        self.tool_llm = ChatOllama(
            model="llama3.1",
            temperature=0.6,
            streaming=False,
            num_ctx=16384
        )
        self.model_llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0.6,
            streaming=False,
            num_ctx=45000
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

        # Load FORMULA Manual PDF for context
        """
        loader = PyMuPDFLoader(
            "docs/formula/Manual.pdf", 
            mode = "single"
        )
        manual_doc = loader.load()
        safe_manual = manual_doc[0].page_content.replace("{", "{{").replace("}", "}}")

        # parse chapters 1-3
        config = {
            "output_format": "markdown",
            "page_range": "17-54"
        }
        config_parser = ConfigParser(config)

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict()
        )
        rendered = converter("docs/formula/Manual.pdf")
        text, _, _ = text_from_rendered(rendered)
        safe_manual = text.replace("{", "{{").replace("}", "}}")
        print(safe_manual)
        manual_token_count = count_tokens_approximately(safe_manual)
        print(f"[DEBUG] System prompt tokens estimate: {manual_token_count}")
        """
        
        # Load FORMULA examples from local files
        """
        "TenDigitNumber": [
            "TenDigitNumberDescription.txt",
            "TenDigitNumberC.txt",
            "TenDigitNumberFormula.4ml"
        ]
        """
        self.examples = {
            "ArmStrong": [
                "ArmStrongDescription.txt",
                "ArmStrongExampleC.txt",
                "ArmStrongExampleFormula.4ml"
            ],
        }
        joined_examples = self.load_examples()

        self.FORMULA_GEN_SYSTEM_PROMPT = f"""\n
        You are a FORMULA code generator.
        Your goal is to translate user input from `[C SOURCE CODE + NATURAL LANGUAGE DESCRIPTION]` into a 
        syntactially and semantically correct FORMULA `.4ml` model that represents the logic of the source code.
        
        If `[PREVIOUS RESULT]` contains errors or incorrect assumptions, your output **MUST** correct them.

        You are **required** to follow strict formatting rules below:
        1. Begin with a `domain` block
            - Define data constructors using `"name" ::= new (...).` syntax.
            - Define rules (constraint) using `"name" :- ... .` syntax.
            - Include a single `conforms` clause that listing all rules defined.
        2. Follow with a `partial model` block:
            - Instantiate each constructor used and assign variables to all of their arguments. 
            - Ensure variable naming is consistent and shared across all parts of the model.

        [OUTPUT INSTRUCTIONS]
        1. Output **ONLY** valid FORMULA `.4ml` code — do **NOT** include explanations, markdown, or comments.
        2. Avoid using example code or placeholders. Every line must serve a functional purpose in the model.
        3. Ensure your output compiles and respects the FORMULA formatting rules defined above. 
        4. Do **NOT** wrap the ouptput in backticks, markdown formatting, or code blocks of any kind.

        [REFERENCE EXAMPLES]
        Use the example translation pairs below to learn **patterns** of abstraction and constraint formulation from C code and natural language
        into FORMULA models.
        {joined_examples}
        """
        print(f"[DEBUG] SYSTEM PROMPT: ", self.FORMULA_GEN_SYSTEM_PROMPT)
        formula_gen_token_count = count_tokens_approximately(self.FORMULA_GEN_SYSTEM_PROMPT)
        print(f"[DEBUG] Formula_Gen prompt tokens estimate: {formula_gen_token_count}")
        
        # use double curly so LangChain knows it isn't a variable that needs to be substittued
        self.SYSTEM_TOOL_PROMPT = """You are an assistant that can call FORMULA tools.
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
        system_token_count = count_tokens_approximately(self.SYSTEM_TOOL_PROMPT)
        print(f"[DEBUG] System prompt tokens estimate: {system_token_count}")

    # loads reference translation examples and formats them for the FORMULA_GEN_SYSTEM_PROMPT
    def load_examples(self) -> str:
        example_texts = []
        for example_name, files in self.examples.items():
            group_text = f"\n--- EXAMPLE: {example_name} ---"
            for filename in files:
                path = os.path.join("docs", "examples", example_name, filename)
                loader = TextLoader(path)
                doc = loader.load()
                safe_content = doc[0].page_content.replace("{", "{{").replace("}", "}}")
                group_text += f"\n### {filename} ###\n{safe_content}"
            example_texts.append(group_text)
        return "\n".join(example_texts)

    async def initialize_graph(self):
        mcp_tools = await self.mcp_client.get_tools()
        print("[DEBUG] Obtaining tools:")
        for tool in mcp_tools:
            print(f"[DEBUG] - {tool.name}: {tool.description}")
        
        # generates a FORMULA model from C code and natural language description
        def FormulaGen_node(state: AgentState) -> AgentState:
            print("\n[DEBUG] Entering FormulaGen_node (generating formula model)...")

            # incorporate previous models and their errors into the human prompt
            previous_attempts = ""
            for attempt in state["models_results"]:
                previous_attempts += f"\n[PREVIOUS MODEL]\n{attempt['model']}\n\n[PREVIOUS RESULT]\n{attempt['result']}\n"

            human_prompt = f"""
            [C SOURCE CODE + NATURAL LANGUAGE DESCRIPTION]\n
            {state["input"]}\n
            [PREVIOUS ATTEMPTS]
            {previous_attempts}\n
            """
            print(f"[DEBUG] Human prompt:\n{human_prompt}")
            human_token_count = count_tokens_approximately(human_prompt)
            print(f"[DEBUG] Human prompt tokens estimate: {human_token_count}")

            # creates a prompt template and composes it in a single pipeline with the LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.FORMULA_GEN_SYSTEM_PROMPT),
                ("human", "{input}")
            ])
            chain = prompt | self.model_llm

            result = chain.invoke({"input": human_prompt})
            print("[DEBUG] Raw LLM response object:", result)
            response_tokens = count_tokens_approximately(result.content)
            print(f"[DEBUG] LLM response tokens estimate: {response_tokens}")

            return {
                "input": state["input"],
                "messages": state.get("messages", []) + [
                    AIMessage(content = result.content)
                ],
                "models_results": state["models_results"],
                "iterations": state["iterations"],
                "latest_model": state["latest_model"] 
            }
        
        # saves the generated model to disk and generates a tool_call for "load" command
        def model_loader_node(state: AgentState) -> AgentState:
            print("\n[DEBUG] Entering model_loader_node...")
            input_text = ""
            latest_model = state["latest_model"]

            # check if FormulaGen produced a FORMULA model
            last_message = state["messages"][-1]
            if last_message:
                if isinstance(last_message, AIMessage):
                    model = last_message.content
                    print("[DEBUG] Model:\n", model)
                    if "domain" in model and "model" in model:
                        # create unique filename for generated model
                        filename = f"{uuid.uuid4().hex}.4ml"

                        # write model to local file
                        output_dir = os.path.join("docs", "generated_models")
                        os.makedirs(output_dir, exist_ok = True)
                        with open(os.path.join(output_dir, filename), "w") as f:
                            f.write(model)
                        print(f"[DEBUG] Saved generated FORMULA model as: {filename}")
                
                        input_text = f"load docs/generated_models/{filename}"
                        latest_model = model

            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_TOOL_PROMPT),
                ("human", "{input}")
            ])
            # combines prompt and model into a single pipeline
            chain = prompt | self.tool_llm
            print(f"[DEBUG] Sending prompt to LLM: {input_text}")
            # LangChain's Runnable syntax - executes the pipeline with the given input 
            result = chain.invoke({"input": input_text})
            print("[DEBUG] Raw LLM response object:", result)

            # Convert any JSON string tool_calls outputted from the LLM in "content" into tool_calls JSON object
            tool_calls = []
            try:
                try:
                    parsed = json.loads(result.content)
                    print("[DEBUG] JSON data successfully parsed")
                except json.JSONDecodeError as e:
                    print("[DEBUG] Error decoding JSON:", e)
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
                "input": state["input"],
                "messages": state.get("messages", []) + [
                    AIMessage(
                        content="" if tool_calls else result.content,
                        additional_kwargs=getattr(result, "additional_kwargs", {}),
                        tool_calls=tool_calls if tool_calls else []
                    )
                ],
                "models_results": state["models_results"],
                "iterations": state["iterations"],
                "latest_model": latest_model
            }

        # routes to 'tools' if the last message contains a tool call; otherwise, ends execution
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
        
        # records the result of the tool call and updates model history in AgentState
        def load_result_node(state: AgentState) -> AgentState:
            print("\n[DEBUG] Entering load_result_node...")

            last_message = state["messages"][-1]
            tool_output = last_message.content if isinstance(last_message, ToolMessage) else ""
            print(f"[DEBUG] Generated Model\n{state['latest_model']}")
            print(f"[DEBUG] Tool Result\n{tool_output}")

            models_results = state.get("models_results", [])
            models_results.append({
                "model": state["latest_model"],
                "result": tool_output
            })
            iterations = state.get("iterations", 0) + 1

            return {
                "input": state["input"],
                "messages": state["messages"],
                "models_results": models_results,
                "iterations": iterations,
                "latest_model": state["latest_model"]
            }
        
        # determines whether to retry FORMULA model generation based on "load" tool feedback
        def should_attempt_again(state: AgentState) -> str:
            print("\n[DEBUG] Entering should_attempt_again_node...")

            last_model_result = state["models_results"][-1]
            result = last_model_result["result"]
            print(f"[DEBUG] {result}")
            if "(Compiled)" in result:
                print("[DEBUG] Model compiled successfully. Ending.")
                return END
        
            if "Error" in last_model_result["result"]:
                if state.get("iterations", 0) >= 1:
                    print("[DEBUG] Max iterations reached")
                    return END
                print("[DEBUG] Retrying FORMULA generation.")
                return "FormulaGen"
            
            print("[DEBUG] No syntax errors found. Ending.")
            return END

        graph = StateGraph(AgentState)
        graph.add_node("FormulaGen", FormulaGen_node)
        graph.add_node("model_loader", model_loader_node)
        # ToolNode is a prebuilt node that automatically reads tool_calls from AgentState["messages"],
        # finds the matching MCP defined tool, and executes it with the given arguments.
        # It then wraps the reutrn value of the tool call in a ToolMessage and adds it to AgentState["messages"]
        graph.add_node("tools", ToolNode(mcp_tools))
        graph.add_node("load_result", load_result_node)

        graph.set_entry_point("FormulaGen")
        graph.add_edge("FormulaGen", "model_loader")
        graph.add_conditional_edges("model_loader", should_call_tool)
        graph.add_edge("tools", "load_result")
        graph.add_conditional_edges("load_result", should_attempt_again)

        print("[DEBUG] Compiling LangGraph...")
        self.app = graph.compile()
        print("[DEBUG] Graph compiled successfully.")

    def read_multiline_input(self) -> str:
        print("Chat session started. Type 'exit' to quit.")
        print("Paste your C code and natural language description. End with an \"END\" on a new line")
        lines = []
        while True:
            try:
                line = input("\nYou: ")
                lines.append(line)
                if line.strip() == "END" or line.strip() == "exit":
                    break
            except EOFError:
                break
        return "\n".join(lines)

    def cleanup_generated_models(self):
        generated_dir = os.path.join("docs", "generated_models")
        if os.path.exists(generated_dir):
            shutil.rmtree(generated_dir)
            print(f"[DEBUG] Deleted {generated_dir} and its files")

    async def interactive_chat(self) -> str:
        while True:
            user_input = self.read_multiline_input()
            if user_input.lower() == "exit":
                print("Ending chat session...")
                break

            state = {
                "input": user_input,
                "messages": [],
                "models_results": [],
                "iterations": 0,
                "latest_model": ""
            }
            # runs the full state graph - processes each node, follows graph logic, and returns the final AgentState
            result = await self.app.ainvoke(state)
            print("\n[DEBUG] Agent returned state:", result)
            last_message = result["messages"][-1]
            print(
                f"\n[Agent Output]:\n-------- MODEL --------\n{result['latest_model']}\n"
                f"-------- RESULT --------\n{last_message.content}\n{'-' * 23}\n"
            )
        self.cleanup_generated_models()

async def main():
    print("[DEBUG] Initializing MCPClient...")
    client = MCPClient()

    print("\n[DEBUG] Initializing LangGraph agent...")
    await client.initialize_graph()

    print("\n[DEBUG] Starting interactive loop...")
    await client.interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())