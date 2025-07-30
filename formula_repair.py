import asyncio
import os
import json
import uuid
import shutil
import stat

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
    latest_model: str
    tool_reset: bool

class MCPRepair:
    def __init__(self, mcp_server_url="http://127.0.0.1:8000"):
        self.tool_llm = ChatOllama(
            model="llama3.1:latest",
            temperature=0.6,
            streaming=False,
            num_ctx=16384
        )
        self.repair_llm = ChatOllama(
            # qwen2.5-coder:7b performs better for SendMoreMoney.4ml
            # yi-coder:latest performs better for MappingExample.4ml
            model="yi-coder:latest",
            temperature=0.6,
            streaming=False,
            num_ctx=50000
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
        When the user asks to **generate a solution for** a model file, you must:
        - Create ONLY **one** `tool_call` per response.
        - NEVER create more than **one** `tool_call` in a single response, even if you know the next step.
        - Always create a new `tool_call` based on the `previous_tool`. 
        - When creating any `tool_call`, you MUST extract and reuse the **exact file path** from the `[USER PROMPT]`.
          Do NOT guess or substitute another path. Always preserve forward slashes `/` in the path (e.g., "docs/generated_models/example.4ml").

        - ONLY if the value of `previous_tool` is empty, respond ONLY with a `tool_call` named `"load_file"` with argument:
          `"filename"` = the file path from `[USER PROMPT]` (e.g., "docs/conflicts/MappingExample.4ml").
        - ONLY if `previous_tool` is `"load_file"`, respond ONLY with a tool_call named `"solve_file"` with argument:
          `"filename"` = the same file path from `[USER PROMPT]` (e.g., "docs/conflicts/MappingExample.4ml").
        - ONLY if `previous_tool` is `"solve_file"`, you will receive its output as a `tool_result`: `"content"`. 
          Look at content to find the line: "Started solve task with Id task_id".
          Then respond ONLY with a `tool_call` named `"extract_solution"` with argument: `"id"` = the "task_id" you found.
            - The `"id"` argument must be a string, not a number.

        For each step, respond ONLY with a JSON object in this **EXACT** format:
        {{
            "tool_calls": [
                {{
                    "name": "load_file" | "solve_file" | "extract_solution",
                    "arguments": {{ ... }}
                }}
            ]
        }}
        Do NOT include any explanation or text outside of this JSON.
        """
        system_token_count = count_tokens_approximately(self.SYSTEM_TOOL_PROMPT)
        print(f"[DEBUG] System prompt tokens estimate: {system_token_count}")
        self.FORMULA_REPAIR_PROMPT = """You are a FORMULA model repair agent. 
        Your goal is to resolve conflicts found by the `"extract_solution"` `tool_call`. 
        A conflict in FORMULA means that two or more constraints cannot be satisfied simultaneously. 

        To resolve these conflicts, you **MUST** thoroughly follow the rules below. Do **NOT** skip any steps or substep:
        1. Do **NOT** modify or remove any data in the `model` or `partial model`.
            - These blocks must be preserved **exactly** in the final output.
        2. You may **ONLY** act on the constraints listed in `[CONFLICT MESSAGES]`
            - Prefer removing the constraint(s) that are less aligned with the domains intended logic.
            - For each constraint, decide to either **keep** or **remove** it.
            - If you remove a conflicting constraint, you **MUST** follow Rule 3
            - You **MUST** preserve at least one constraint explicitly listed in `[CONFLICT MESSAGES]` exactly as it
              appears in the `[CURRENT MODEL]`. It must also appear unchanged in `conforms`.
        3. If you choose to remove a constraint, you **MUST**:
            - Remove it as a rule from the `domain` block.
            - Remove it from `conforms`.
            - Remove any reference to it in other constraints.
        4. Do **NOT** modify or remove any other constraints **not listed** in `[CONFLICT MESSAGES]`
        5. Before returning an updated model, you **MUST** thoroughly verify that:
            - The `domain` and `conforms` blocks reference the **SAME** set of constraints.
            - Every removed constraint does **NOT** appear in the `domain` block and in `conforms`.
            - At least one constraint from `[CONFLICT MESSAGES]` **MUST** be retained without any modifications
              and **exactly match** its definition and usage in the `[CURRENT MODEL]`.
        6. Your explanation **MUST** exactly match your final output. 
            - Do not claim to remove or keep any constraint unless these changes are accurately reflected in the `domain` and `conforms` blocks.

        [OUTPUT INSTRUCTIONS]
        1. Output a **single, complete** FORMULA model after you verified all rules have been followed.
        2. Do **NOT** output any reasoning, the current model, partial edits, or `[CONFLICT MESSAGES]` again.
        3. Include `[FINAL OUTPUT]` and a **concise** `[EXPLANATION]` describing the fix and how each rule was followed.
        4. Do **NOT** wrap the ouptput in backticks, markdown formatting, or code blocks of any kind.

        [EXAMPLE]
        [CURRENT MODEL]
        domain Battery {{
            Percentage ::= new (val: Integer).
            Temperature ::= new (fahrenheit: Real).

            isNegative :- p is Percentage, p.val <= 0.
            isPositive :- p is Percentage, p.val > 0.
            isOverheated :- t is Temperature, t.fahrenheit > 140.

            conforms isNegative, isPositive, no isOverheated.
        }}

        partial model pm of Battery {{
            percent is Percentage(x).
            temp is Temperature(y).
        }}

        [CONFLICT MESSAGES]
        Model not solvable. Unsat core terms below.
        Conflicts: Battery.isNegative 
        Conflicts: Battery.isPositive 

        [FINAL OUTPUT]
        domain Battery {{
            Percentage ::= new (val: Integer).
            Temperature ::= new (fahrenheit: Real).

            isPositive :- p is Percentage, p.val > 0.
            isOverheated :- t is Temperature, t.fahrenheit > 140.

            conforms isPositive, no isOverheated.
        }}

        partial model pm of Battery {{
            percent is Percentage(x).
            temp is Temperature(y).
        }}

        [EXPLANATION]
        - Rule 1: Preserved the `model` and `partial model` blocks exactly as given in the final output.
        - Rule 2: Acted only on constraints listed in `[CONFLICT MESSAGES]`. Preserved `isPositive` **exactly** as it appears in
                  [CURRENT MODEL], and included it unchanged in `conforms`. Chose to remove `isNegative` because
                  `isPositive` better reflects realistic battery behavior.
        - Rule 3: Removed `isNegative` as a rule in the `domain` block, `conforms`, and all other references.
        - Rule 4: Did not modify or remove the `isOverheated` constraint since it was not listed in `[CONFLICT MESSAGES]`
        - Rule 5: Verified that `domain` and `conforms` contain the same set of defined constraints (`isPositive`, `isOverheated`).
                  Confirmed that `isNegative` is fully removed, and at least one conflicting constraint (`isPositive`) was
                  retained without any modifications and matches its definitions and usage in `[CURRENT MODEL]` exactly. 
        - Rule 6: Did not include any reasoning for the model before `[FINAL OUTPUT]`.
                  The `[EXPLANATION]` is fully consistent with the `[FINAL OUTPUT]`. 
                  All described changes are accurately reflected in the `domain` and `conforms` blocks.
                  A single, complete model was produced. No `[CURRENT MODEL]`, `[CONFLICT MESSAGES]`, or partial models were included.
        """
        repair_token_count = count_tokens_approximately(self.FORMULA_REPAIR_PROMPT)
        print(f"[DEBUG] Repair prompt tokens estimate: {repair_token_count}")
    
    async def initialize_graph(self):
        mcp_tools = await self.mcp_client.get_tools()
        print("[DEBUG] Obtaining tools:")
        for tool in mcp_tools:
            print(f"[DEBUG] - {tool.name}: {tool.description}")

        # creates tool_calls for loading or solving a model, or extracting a task solution
        def tool_planner_node(state: AgentRepairState) -> AgentRepairState:
            print("\n[DEBUG] Entering tool_planner_node...")

            # read the model file contents from the provided input path
            input_text = state["input"]
            input_parts = input_text.split("generate a solution for", 1)
            model_path = input_parts[1].strip()
            if not state.get("latest_model"):
                try:
                    with open(model_path, "r") as f:
                        state["latest_model"] = f.read()
                        print(f"[DEBUG] Loaded model from {model_path}")
                except Exception as e:
                    print(f"[DEBUG] Failed to load model: {e}")

            # reset tool_call context if validating a freshly repaired model
            if model_path.startswith("docs/generated_models/") and not state.get("tool_reset"):
                last_tool_call = ""
                last_tool_result = ""
                state["tool_reset"] = True
                print("[DEBUG] Resetting tool context for fresh model validation.")
            # otherwise, extract the last tool_call (and result for the solve command)
            else:
                last_tool_call = ""
                last_tool_result = ""
                if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
                    last_tool_call = state["messages"][-1].name
                    print(f"[DEBUG] Last tool '{last_tool_call}'")
                    if last_tool_call == "solve_file":
                        last_tool_result = state["messages"][-1].content

            # construct human prompt
            human_prompt = "\n".join([
                "[USER PROMPT]",
                input_text,
                "[PREVIOUS TOOL CALL]",
                f"previous_tool = {last_tool_call}",
                "[PREVIOUS TOOL RESULT]",
                f"tool_result = {last_tool_result}",
            ])
            print(f"[DEBUG] Human prompt:\n{human_prompt}")
            human_token_count = count_tokens_approximately(human_prompt)
            print(f"[DEBUG] Human prompt tokens estimate: {human_token_count}")

            # construct full prompt
            prompt_parts = [
                ("system", self.SYSTEM_TOOL_PROMPT),
                ("human", "{input}")
            ]
            prompt = ChatPromptTemplate.from_messages(prompt_parts)
            
            # invoke llm
            chain = prompt | self.tool_llm
            result = chain.invoke({"input": human_prompt})
            print("[DEBUG] Raw LLM response object:", result)
            response_tokens = count_tokens_approximately(result.content)
            print(f"[DEBUG] LLM response tokens estimate: {response_tokens}")

            # convert any JSON string tool_calls outputted from the LLM in "content" into tool_calls JSON object
            debug_message = ""
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
                debug_message = f"[DEBUG] No valid JSON tool_call structure found: {e}"

            return {
                "input": input_text,
                "messages": state.get("messages", []) + [
                    AIMessage(
                        content=debug_message or result.content,
                        additional_kwargs=getattr(result, "additional_kwargs", {}),
                        tool_calls=tool_calls if tool_calls else []
                    )
                ],
                "latest_model": state["latest_model"],
                "tool_reset": state["tool_reset"]
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
        
        # loops to tool planner until an execute tool_call is made
        # routes to repair node if extraction produced a conflict in the model
        def route_after_tool(state: AgentRepairState) -> str:
            print("\n[DEBUG] Entering route_after_tool...")
            # checks to route to END
            if not state["messages"]:
                print("\n[DEBUG] No message found...")
                return END
            last_tool_message = state["messages"][-1]
            if not isinstance(last_tool_message, ToolMessage):
                print("\n[DEBUG] Last message found was not ToolMessage...")
                return END
            
            # determine where to route next based on the last tool_call made
            last_tool = last_tool_message.name
            if last_tool == "load_file" or last_tool== "solve_file":
                print(f"[DEBUG] {last_tool} called. Routing to 'tool_planner' node...")
                return "tool_planner"
            if last_tool == "extract_solution":
                result = last_tool_message.content
                if "Conflicts" in result:
                    print("[DEBUG] Conflicts found. Routing to 'repair_agent'...")
                    return "model_repair"
                else:
                    print("[DEBUG] No conflicts. Routing to END.")
                    return END
            return END    
        
        # resolves model conflicts using the repair LLM and saves the updated model to disk for revalidation
        def model_repair_node(state: AgentRepairState) -> AgentRepairState:
            print("\n[DEBUG] Entering model_repair_node...")

            # get the last ToolMessage (results from extract_solution tool)
            last_msg = state["messages"][-1]
            conflicts = ""
            if isinstance(last_msg, ToolMessage):
                conflicts = last_msg.content
                print("[DEBUG] Found conflicts in model:\n", conflicts)

            # construct human prompt
            human_prompt = "\n".join([
                "[CURRENT MODEL]",
                f"{state['latest_model']}",
                "",
                "[CONFLICT MESSAGES]",
                f"{conflicts}\n",
            ])
            print(f"[DEBUG] Human prompt:\n{human_prompt}")
            human_token_count = count_tokens_approximately(human_prompt)
            print(f"[DEBUG] Human prompt tokens estimate: {human_token_count}")

            # construct full prompt
            prompt_parts = [
                ("system", self.FORMULA_REPAIR_PROMPT),
                ("human", "{input}")
            ]
            prompt = ChatPromptTemplate.from_messages(prompt_parts)
            
            # invoke llm
            chain = prompt | self.repair_llm
            print("[DEBUG] Beginning LLM stream...")
            streamed_content = ""
            for chunk in chain.stream({"input": human_prompt}):
                print(chunk.content, end="", flush=True)
                streamed_content += chunk.content

            result = AIMessage(content=streamed_content)
            response_tokens = count_tokens_approximately(result.content)
            print(f"[DEBUG]\nLLM response tokens estimate: {response_tokens}")

            # extract the model content from the LLM response
            model_code = result.content.split("[EXPLANATION]")[0].replace("[FINAL OUTPUT]", "").strip()

            # write model to local file
            filename = f"{uuid.uuid4().hex}.4ml"
            output_dir = os.path.join("docs", "generated_models")
            os.makedirs(output_dir, exist_ok = True)
            model_path = os.path.join(output_dir, filename)
            with open(model_path, "w") as f:
                f.write(model_code)
            print(f"[DEBUG] Saved repaired FORMULA model to: {model_path}")
            
            return {
                "input": f"generate a solution for {model_path.replace("\\", "/")}",
                "messages": state["messages"],
                "latest_model": model_code,
                "tool_reset": False
            }

        graph = StateGraph(AgentRepairState)
        graph.add_node("tool_planner", tool_planner_node)
        graph.add_node("tools", ToolNode(mcp_tools))
        graph.add_node("model_repair", model_repair_node)

        graph.set_entry_point("tool_planner")
        graph.add_conditional_edges("tool_planner", should_call_tool)
        graph.add_conditional_edges("tools", route_after_tool)
        graph.add_edge("model_repair", "tool_planner")

        print("[DEBUG] Compiling LangGraph...")
        self.app = graph.compile()
        print("[DEBUG] Graph compiled successfully.")

    # deletes all generated model files from the local output directory
    def cleanup_generated_models(self):
        generated_dir = os.path.join("docs", "generated_models")
        if os.path.exists(generated_dir):
            os.chmod(generated_dir, stat.S_IWRITE)
            shutil.rmtree(generated_dir)
            print(f"[DEBUG] Deleted {generated_dir} and its files")

    # runs an interactive loop that sends user input through the repair graph until 'exit' is entered
    async def interactive_chat(self):
        print("Chat session started. Type 'exit' to quit.")
        state = {"input": "", "messages": [], "latest_model": "", "tool_reset": False}

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Ending chat session...")
                break

            print("[DEBUG] Invoking agent with input:", user_input)
            state["input"] = user_input
            result = await self.app.ainvoke(state)
            print("\n[DEBUG] Agent returned state:", result)
            last_message = result["messages"][-1]
            print(f"\nAgent: {last_message.content}\n")
        self.cleanup_generated_models()

async def main():
    print("[DEBUG] Initializing MCPRepair...")
    client = MCPRepair()

    print("\n[DEBUG] Initializing LangGraph agent...")
    await client.initialize_graph()

    print("\n[DEBUG] Starting interactive loop...")
    await client.interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())