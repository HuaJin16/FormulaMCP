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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import TypedDict
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from marker.converters.pdf import PdfConverter
from marker.settings import settings            
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from langchain.schema import Document

CONTEXT_LIMIT_TOKENS = 131072;   
nest_asyncio.apply()




# state - a way to maintain and track information as the llm flows through the LangGraph system 
class AgentState(TypedDict):
    input: str
    # LangGraph and ToolNode expects list of BaseMessage as "messages"
    messages: List[BaseMessage]
    models_results: List[Dict[str, str]]
    iterations: int
    latest_model: str

class MCPClient:                                                                                        
    def __init__(self, mcp_server_url="http://127.0.0.1:8000"):
        # initialize ollama
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False,
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

        summary_path = "docs/formula_grammar_summary.txt"       

        def load_and_escape_grammar(file_path: str) -> str:
            with open(file_path, "r") as f:
                content = f.read()
            return content.replace("{", "{{").replace("}", "}}")

        if os.path.exists(summary_path):
            print("[DEBUG] Loading cached FORMULA grammar summary...")
            with open(summary_path, "r") as f:
                grammar_summary = f.read()
        else:
            print("[DEBUG] Grammar summary not found — generating from Manual.pdf...")

            # ---- USE MARKER-PDF HERE ----
            print("[DEBUG] Converting Manual.pdf to Markdown using marker-pdf...")
                # init converter        
            config_parser = ConfigParser({"output_format": "markdown"})
            conv = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        # run conversion
            res = conv("docs/Manual.pdf")
            markdown_output = res.markdown


            doc = Document(page_content=markdown_output)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents([doc])

            # Compress the chunks with LLM
            llm = ChatOllama(model="llama3.2", temperature=0.3)
            compressor = LLMChainExtractor.from_llm(llm)
            compressed_chunks = compressor.compress_documents(
                chunks, 
                query="""
                Extract a **technical cheat sheet** of the FORMULA programming language.
                Focus only on:
                    - The structure of `domain`, `model`, `transform`, `machine`, `rule`, `type`, `constraint` blocks.
                    - Grammar keywords (like `new`, `ensures`, `no`, `conforms`, `:-`) and their valid use.
                    - Nesting rules and what’s required inside each block.
                DO NOT include historical background, tooling usage, or example explanations.
                """
            )

            def escape_curly_braces(text: str) -> str:
                return text.replace("{", "{{").replace("}", "}}")

            grammar_summary = escape_curly_braces(
                "\n".join([doc.page_content for doc in compressed_chunks])
            )
            with open(summary_path, "w") as f:
                f.write(grammar_summary)                

        safe_lexer = load_and_escape_grammar("docs/FormulaLexer.g4")
        safe_parser = load_and_escape_grammar("docs/FormulaParser.g4")     

        # Escape braces for f-string


        # Load FORUMLA Manual PDF for context
        # loader = PyMuPDFLoader(
        #     "docs/Manual.pdf", 
        #     mode = "single"
        # )
        # manual_doc = loader.load()
        # safe_manual = manual_doc[0].page_content.replace("{", "{{").replace("}", "}}")

        # Load FORMULA examples from local files
        self.examples = {
            "ArmStrong": [
                "ArmStrongDescription.txt",
                "ArmStrongExampleC.txt",
                "ArmStrongExampleFormula.4ml"
            ],
            "TenDigitNumber": [
                "TenDigitNumberDescription.txt",
                "TenDigitNumberC.txt",
                "TenDigitNumberFormula.4ml"
            ],
            "SendMoreMoney": [
                "SendMoreMoneyDescription",           # No file extension, fix if needed
                "SendMoreMoney.c",
                "SendMoreMoneyExample.4ml"
            ],
           "Ldp": [
                "LdpDescription.txt",
                "Ldp.c",
                "ldp.4ml"
            ]
        }
        joined_examples = self.load_examples()

        """
        You are also given the FORMULA model. Learn the language's syntax and semantics
        {safe_manual}
        """
        self.FORMULA_GEN_SYSTEM_PROMPT = f"""\n
        You are a FORMULA code generator.
        Your task is to convert a C source code **and** its natural language description into a valid `.4ml` FORMULA model.

        You must follow strict formatting:
        - Begin with a `domain` block
        - Follow with a `model` or `partial model` block
        - Do NOT include any explanation, markdown, or comments
        - Use examples below for **syntax patterns only**   

        [FORMULA GRAMMAR SUMMARY — Learn syntax from this only, do not repeat]
        {grammar_summary}
                                    
        [LEXER GRAMMAR — Token definitions for FORMULA language]
        {safe_lexer}

        - Use this to understand which words are reserved keywords (e.g., `domain`, `model`, `ensures`, `new`, etc.)
        - Learn how literals (e.g., numbers, strings, ranges like `0..9`) and identifiers (e.g., `V`, `Map.left`) must be written
        - Avoid generating invalid tokens or unknown symbols

        [PARSER GRAMMAR — Full syntax rules for FORMULA language]
        {safe_parser}

        - Use this to learn the structure of valid `.4ml` models
        - Know what grammar rules must be followed for:
            - `domain`, `model`, `transform`, `machine` blocks
            - constraint logic using `no`, `ensures`, `requires`, `conforms`
            - rule syntax using `:-`, function calls, comprehensions
        - Use this as your guide for ordering, nesting, and completing code constructs correctly


        You are given example translation pairs below. Learn the mapping pattern from description and C logic to FORMULA constraints.
        [EXAMPLEs — FOR REFERENCE ONLY. DO NOT COPY]
        {joined_examples}
        """

        print(f"[DEBUG] SYSTEM PROMPT: ", self.FORMULA_GEN_SYSTEM_PROMPT)
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

    # load C & Natural Language Description -> FORMULA model examples and format for system prompt
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
        
        # creates and writes a FORMULA model 
        def FormulaGen_node(state: AgentState) -> AgentState:
            print("\n[DEBUG] Entering FormulaGen_node (generating formula model)...")
            
            def token_debug(name: str, content: str) -> int:
                tokens = count_tokens_approximately(content)
                print(f"[TOKENS] {name:<30}: {tokens:>5} tokens")
                return tokens

            previous_attempts = "\n".join(
                f"[PREVIOUS MODEL]\n{attempt['model']}\n\n[PREVIOUS RESULT]\n{attempt['result']}\n"
                for attempt in state["models_results"]
            )

            full_prompt = f"""
            Below is a string consisting of a C source code, a natural language description, and any previous attempts made
            at generating a FORMULA model.

            [C SOURCE CODE + NATURAL LANGUAGE DESCRIPTION]\n
            {state["input"]}\n

            [PREVIOUS ATTEMPTS]
            {previous_attempts}\n

            [YOUR TASK]
            Write a valid FORMULA `.4ml` model that reflects the logic of the user provided C source code and description.
       
            """

            # creates a prompt template and composes it in a single pipeline with the LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.FORMULA_GEN_SYSTEM_PROMPT),
                ("human", "{input}")
            ])
            chain = prompt | self.llm



             # Token usage breakdown
            print("\n[TOKEN USAGE ANALYSIS]")
            token_debug("System Prompt", self.FORMULA_GEN_SYSTEM_PROMPT)
            token_debug("User Input (C + NL)", state["input"])
            token_debug("Previous Attempts", previous_attempts)
            total_tokens = token_debug("Full Prompt", full_prompt)

            remaining_tokens = CONTEXT_LIMIT_TOKENS - total_tokens
            if remaining_tokens < 0:
                print(f"[WARNING] Full prompt exceeds context limit by {-remaining_tokens} tokens")
            elif remaining_tokens < 500:
                print(f"[CAUTION] You are near the context limit. Only {remaining_tokens} tokens left.")    
    
    
            if total_tokens > CONTEXT_LIMIT_TOKENS:
                print(f"[WARNING] Token count exceeds {CONTEXT_LIMIT_TOKENS} — prompt may be truncated!")

            print("\n[DEBUG] Sending FORMULA generation prompt to model...")
            result = chain.invoke({"input": full_prompt})
            print("[DEBUG] LLM Response:", result)

            return {
                "input": state["input"],
                "messages": state.get("messages", []) + [
                    AIMessage(content = result.content)
                ],
                "models_results": state["models_results"],
                "iterations": state["iterations"],
                "latest_model": state["latest_model"] 
            }
        
        # write the generated model to disk, generates tool call for "load"
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
                ("system", self.SYSTEM_PROMPT),
                ("human", "{input}")
            ])
            # combines prompt and model into a single pipeline
            chain = prompt | self.llm
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
        
        # adds the results of loading the generated model to AgentState
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
        
        # determines if the LLM need to attempt creating the model again
        def should_attempt_again(state: AgentState) -> str:
            print("\n[DEBUG] Entering should_attempt_again_node...")
            if state.get("iterations", 0) >= 2:
                print("[DEBUG] Max iterations reached")
                return END
        
            last_model_result = state["models_results"][-1]
            print(f"[DEBUG] {last_model_result}")
            if "Error" in last_model_result["result"]:
                print("[DEBUG] Retrying FORMULA generation.")
                return "FormulaGen"
            
            print("[DEBUG] No syntax error, done.")
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
            print(f"""\n[Agent Output]:\n----- MODEL -----{state['latest_model']}---- RESULT -----{last_message.content}------------------""")
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