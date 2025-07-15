import asyncio
import os
import json
import uuid
import shutil

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

class MCPRepair:
    def __init__(self, mcp_server_url="http://127.0.0.1:8000"):
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False,
            num_ctx=40000
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
        When the user asks to **provide a solution for** a model file, you must:
        - Create ONLY **one** tool_call per response.
        - NEVER create more than **one** tool_call in a single response, even if you know the next step.
        - Always create a new tool_call based on the previous_tool. 
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
        print(f"[DEBUG] System prompt tokens estimate: {system_token_count}")
        self.FORMULA_REPAIR_PROMPT = """You are a FORMULA model repair agent. 
        Your goal is to resolve conflicts found by the `extract_solution` tool call. 
        A conflict in FORMULA means that two or more constraints cannot be satisfied simultaneously. 

        To resolve these conflicts, you **MUST** thoroughly follow the rules below. Do **NOT** skip any steps or substep:
        1. Do **NOT** modify or remove any data in the `model` or `partial model`.
            - These blocks must be preserved exactly in the final output.
        2. You may **ONLY** act on the constraint(s) listed in [CONFLICT MESSAGES]
           - For each constraint, decide to either **keep** or **remove** it.
           - Prefer removing the constraint(s) that are less aligned with the domains intended logic.
           - If you remove a conflicting constraint, you **MUST** follow Rule 3
           - If you keep a conflicting constraint, it must remain exactly as defined in `domain` and be included in `conforms`.
           - You must keep at least one constraint explicitly listed in [CONFLICT MESSAGES]. 
        3. If you remove a constraint, you **MUST**:
           - Remove its definition from the `domain` block entirely. 
           - Remove it from `conforms`.
           - Remove any reference to it in other constraints.
        4. Do **NOT** modify or remove any other constraints not listed in [CONFLICT MESSAGES]
        5. Before returning an updated model, you **MUST** thoroughly check and verify that:
           - The `domain` and `conforms` blocks contain the **SAME** set of constraints.
           - Any removed constraint(s) does not appear in the final `domain` or `conforms`.
           - At least one constraint from [CONFLICT MESSAGES] is present, unchanged, and in both the `domain` and `conforms` blocks.
        6. Your reasoning and explanation must exactly match your final output. Do not claim to remove or keep any constraint unless 
           it is accurately reflected in the `domain` and `conforms` blocks.

        [OUTPUT INSTRUCTIONS]
        1. Output a **single, complete** FORMULA model after you verified all rules have been followed.
        2. Do **NOT** output the current model, partial edits, or updated conflict message.
        3. Include a **concise** explanation describing the fix and how each rule was followed.

        [EXAMPLE]
        Original input:
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

        [FINAL OUTPUT]:
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

        Explanation:
        - Rule 1: Preserved the `model` and `partial model` blocks exactly as given in the final output.
        - Rule 2: Acted only on constraints listed in [CONFLICT MESSAGES]. Chose to remove `isNegative` because
                  `isPositive` better reflects realistic battery behavior. Kept `isPositive` exactly as defined and included in `conforms`.
        - Rule 3: Fully removed `isNegative` from the `domain` block, `conforms`, and any other references.
        - Rule 4: Did not modify or remove the `isOverheated` constraint since it was not part of the conflict.
        - Rule 5: Checked and verified that `domain` and `conforms` contain the same set of defined constraints (`isPositive`, `isOverheated`),
                  `isNegative` is removed completely, and that at least one constraint from [CONFLICT MESSAGES] (`isPositive`) was kept.
        - Rule 6: The explanation is fully consistent with the [FINAL OUTPUT]. All described changes are accurately reflected in the `domain` and `conforms` blocks.
                  A single, complete model was produced. No [CURRENT MODEL], [CONFLICT MESSAGES], or partial models were included.
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

            # add model to state
            input_text = state["input"]
            input_parts = input_text.split("provide a solution for", 1)
            model_path = input_parts[1].strip()
            try:
                with open(model_path, "r") as f:
                    state["latest_model"] = f.read()
                    print(f"[DEBUG] Loaded model from {model_path}")
            except Exception as e:
                print(f"[DEBUG] Failed to load model: {e}")

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
                "[PREVIOUS TOOL CALL]",
                f"previous_tool = {last_tool_call}",
                "[PREVIOUS TOOL RESULT]",
                f"tool_result = {last_tool_result}\n",
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
            chain = prompt | self.llm
            result = chain.invoke({"input": human_prompt})
            print("[DEBUG] Raw LLM response object:", result)
            response_tokens = count_tokens_approximately(result.content)
            print(f"[DEBUG] LLM response tokens estimate: {response_tokens}")

            # convert any JSON string tool_calls outputted from the LLM in "content" into tool_calls JSON object
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
                ],
                "latest_model": state["latest_model"],
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
        
        # loops to tool planner until an execute tool call is made
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
            
            # determine where to route next based on the last tool call made
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
            chain = prompt | self.llm
            result = chain.invoke({"input": human_prompt})
            print("[DEBUG] Raw LLM response object:", result)
            response_tokens = count_tokens_approximately(result.content)
            print(f"[DEBUG] LLM response tokens estimate: {response_tokens}")

            # create unique filename for generated model
            filename = f"{uuid.uuid4().hex}.4ml"

            # write model to local file
            output_dir = os.path.join("docs", "generated_models")
            os.makedirs(output_dir, exist_ok = True)
            with open(os.path.join(output_dir, filename), "w") as f:
                f.write(result.content)
            print(f"[DEBUG] Saved generated FORMULA model as: {filename}")

            return {
                "input": "",
                "messages": state.get("messages", []) + [
                    AIMessage(
                        content=result.content,
                    )
                ],
                "latest_model": result.content,
            }

        graph = StateGraph(AgentRepairState)
        graph.add_node("tool_planner", tool_planner_node)
        graph.add_node("tools", ToolNode(mcp_tools))
        graph.add_node("model_repair", model_repair_node)

        graph.set_entry_point("tool_planner")
        graph.add_conditional_edges("tool_planner", should_call_tool)
        graph.add_conditional_edges("tools", route_after_tool)
        graph.add_edge("model_repair", END)

        print("[DEBUG] Compiling LangGraph...")
        self.app = graph.compile()
        print("[DEBUG] Graph compiled successfully.")

    def cleanup_generated_models(self):
        generated_dir = os.path.join("docs", "generated_models")
        if os.path.exists(generated_dir):
            shutil.rmtree(generated_dir)
            print(f"[DEBUG] Deleted {generated_dir} and its files")

    async def interactive_chat(self):
        print("Chat session started. Type 'exit' to quit.")
        state = {"input": "", "messages": [], "latest_model": ""}

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