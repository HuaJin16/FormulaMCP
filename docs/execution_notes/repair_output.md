# Execution Overview: Running the FORMULA Repair and Server

This document outlines the expected outputs and interactions between the repair and the backend server. While actual responses may vary due to LLM behavior, the examples and explanations below represent the intended flow of execution.

## formula_repair.py

This script initializes a local MCP repair agent that identifies and resolves **constraint conflicts** in FORMULA `.4ml` models. It uses a LangGraph-based workflow with a planning loop that sequentially executes FORMULA tool calls `(load_file, solve_file, and extract_solution)` before entering the repair phase if conflicts are detected.

### Example Behavior
1. Expected output after starting the MCP server `python formula_server.py`

```bash
🚀 Starting server...
[...] INFO      Starting MCP server 'formula_mcp' with transport 'sse' on
                http://127.0.0.1:8000/sse

                ...

INFO:           Started server process [4808]
INFO:           Waiting for application startup.
INFO:           Application startup complete.
INFO:           Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:           127.0.0.1:52768 - "GET /sse HTTP/1.1" 200 OK
INFO:           127.0.0.1:52769 - "POST /messages/?
                session_id=73d178a910984cec977f294daa332827 HTTP/1.1" 202 Accepted
```

2. Expected output after starting the MCP client `python formula_repair.py`

```bash
[DEBUG] Initializing MCPRepair...
Connecting to MCP server at http://127.0.0.1:8000...
[DEBUG] System prompt tokens estimate: 9275
[DEBUG] Repair prompt tokens estimate: 25945

[DEBUG] Initializing LangGraph agent...
[DEBUG] Obtaining tools:
[DEBUG] - load_file: Returns the output of Loading a FORMULA model
[DEBUG] - solve_file: Returns the output of Solving a FORMULA partial model
[DEBUG] - extract_solution: Returns the first solution from a solve task
[DEBUG] Compiling LangGraph...
[DEBUG] Graph compiled successfully.

[DEBUG] Starting interactive loop...
Chat session started. Type 'exit' to quit.
You:
```

- The repair first connects to the MCP server started in the previous step.
- It then prints the estimated token counts for both:
    - The system prompt, which instructs the LLM how to generate structured tool_call JSON objects for interacting with FORMULA tools.
    - The repair prompt, which instructs the LLM how to resolve constraint conflicts.
- Next, LangGraph retrieves and prints the list of available FORMULA tools from the MCP server `(load_file, solve_file, extract_solution)` and the interactive loop begins. 

3. Expected behavior after entering `"generate a solution for docs/conflicts/MappingExample.4ml"` into `formula_repair.py`

```bash
You: generate a solution for docs/conflicts/MappingExample.4ml
[DEBUG] Invoking agent with input: generate a solution for docs/conflicts/MappingExample.4ml

[DEBUG] Entering tool_planner_node...
[DEBUG] Loaded model from docs/conflicts/MappingExample.4ml
[DEBUG] Human prompt:
[USER PROMPT]
generate a solution for docs/conflicts/MappingExample.4ml
[PREVIOUS TOOL CALL]
previous_tool =
[PREVIOUS TOOL RESULT]
tool_result =
[DEBUG] Human prompt tokens estimate: 735
[DEBUG] Raw LLM response object: content='{\n    "tool_calls": [\n        {\n            
"name": "load_file",\n            "arguments": {\n                "filename": "docs/
conflicts/MappingExample.4ml"\n            }\n        }\n    ]\n}' additional_kwargs={} 
response_metadata={'model': 'llama3.1:latest', 'created_at': '2025-07-30T15:36:43.
9333628Z', 'done': True, 'done_reason': 'stop', 'total_duration': 8714275800, 
'load_duration': 100559300, 'prompt_eval_count': 488, 'prompt_eval_duration': 1773394000, 
'eval_count': 45, 'eval_duration': 6835439900, 'model_name': 'llama3.1:latest'} 
id='run--724ed982-2379-4277-9705-c0f756a9ad92-0' usage_metadata={'input_tokens': 488, 
'output_tokens': 45, 'total_tokens': 533}
[DEBUG] LLM response tokens estimate: 935
[DEBUG] Parsed tool_calls: [{'name': 'load_file', 'args': {'filename': 'docs/conflicts/
MappingExample.4ml'}, 'id': '551390ef-8c8a-47f1-882c-21073cc22972'}]

[DEBUG] Entering should_call_tool...
[DEBUG] Tool call found. Routing to 'tools' node.

[DEBUG] Entering route_after_tool...
[DEBUG] load_file called. Routing to 'tool_planner' node...

[DEBUG] Entering tool_planner_node...
[DEBUG] Last tool 'load_file'
[DEBUG] Human prompt:
[USER PROMPT]
generate a solution for docs/conflicts/MappingExample.4ml
[PREVIOUS TOOL CALL]
previous_tool = load_file
[PREVIOUS TOOL RESULT]
tool_result =
[DEBUG] Human prompt tokens estimate: 780
[DEBUG] Raw LLM response object: content='{\n    "tool_calls": [\n        {\n            
"name": "solve_file",\n            "arguments": {\n                "filename": "docs/
conflicts/MappingExample.4ml"\n            }\n        }\n    ]\n}' additional_kwargs={} 
response_metadata={'model': 'llama3.1:latest', 'created_at': '2025-07-30T15:36:55.
0283566Z', 'done': True, 'done_reason': 'stop', 'total_duration': 9539436600, 
'load_duration': 101107400, 'prompt_eval_count': 490, 'prompt_eval_duration': 1338536800, 
'eval_count': 45, 'eval_duration': 8086421200, 'model_name': 'llama3.1:latest'} 
id='run--3b079682-6209-467e-805b-df3d8e05eac2-0' usage_metadata={'input_tokens': 490, 
'output_tokens': 45, 'total_tokens': 535}
[DEBUG] LLM response tokens estimate: 940
[DEBUG] Parsed tool_calls: [{'name': 'solve_file', 'args': {'filename': 'docs/conflicts/
MappingExample.4ml'}, 'id': 'e6f02104-8e78-4e0a-8282-694f3901d8fa'}]

[DEBUG] Entering should_call_tool...
[DEBUG] Tool call found. Routing to 'tools' node.

[DEBUG] Entering route_after_tool...
[DEBUG] solve_file called. Routing to 'tool_planner' node...

[DEBUG] Entering tool_planner_node...
[DEBUG] Last tool 'solve_file'
[DEBUG] Human prompt:
[USER PROMPT]
generate a solution for docs/conflicts/MappingExample.4ml
[PREVIOUS TOOL CALL]
previous_tool = solve_file
[PREVIOUS TOOL RESULT]
tool_result = Parsing text took: 2
Visiting text took: 0
Started solve task with Id 0.
0.48s.
[DEBUG] Human prompt tokens estimate: 1200
[DEBUG] Raw LLM response object: content='{\n    "tool_calls": [\n        {\n            
"name": "extract_solution",\n            "arguments": {\n                "id": 
"0"\n            }\n        }\n    ]\n}' additional_kwargs={} response_metadata={'model': 
'llama3.1:latest', 'created_at': '2025-07-30T15:37:02.9196063Z', 'done': True, 
'done_reason': 'stop', 'total_duration': 6595800400, 'load_duration': 82259000, 
'prompt_eval_count': 518, 'prompt_eval_duration': 929260300, 'eval_count': 37, 
'eval_duration': 5570705800, 'model_name': 'llama3.1:latest'} 
id='run--c0b266d0-eb50-49d4-bd0b-d94ae3d61b07-0' usage_metadata={'input_tokens': 518, 
'output_tokens': 37, 'total_tokens': 555}
[DEBUG] LLM response tokens estimate: 780
[DEBUG] Parsed tool_calls: [{'name': 'extract_solution', 'args': {'id': '0'}, 'id': 
'20b6b20d-ca07-46e8-b78c-3ccd3ef046df'}]

[DEBUG] Entering should_call_tool...
[DEBUG] Tool call found. Routing to 'tools' node.

[DEBUG] Entering route_after_tool...
[DEBUG] Conflicts found. Routing to 'repair_agent'...

[DEBUG] Entering model_repair_node...
[DEBUG] Found conflicts in model:
 Model not solvable. Unsat core terms below.
Conflicts: Mapping.invalidUtilization
Conflicts: Mapping.validUtilization

0.01s.
[DEBUG] Human prompt:
[CURRENT MODEL]
domain Mapping
{
  Component ::= new (id: Integer, utilization: Real).
  Processor ::= new (id: Integer).
  Mapping   ::= new (c: Component, p: Processor).

  // The utilization must be > 0
  invalidUtilization :- c is Component, c.utilization <= 0.
  validUtilization :- c is Component, c.utilization > 0.

  badMapping :- p is Processor,
    s = sum(0.0, { c.utilization |
              c is Component, Mapping(c, p) }), s > 100.

  conforms no badMapping, no invalidUtilization, no validUtilization.
}

model m of Mapping
{
  c1 is Component(0, 10).
  c2 is Component(1, 90).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}

partial model pm of Mapping
{
  c1 is Component(0, x).
  c2 is Component(1, y).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}

[CONFLICT MESSAGES]
Model not solvable. Unsat core terms below.
Conflicts: Mapping.invalidUtilization
Conflicts: Mapping.validUtilization

0.01s.

[DEBUG] Human prompt tokens estimate: 4765
[DEBUG] Beginning LLM stream...
[FINAL OUTPUT]
domain Mapping
{
  Component ::= new (id: Integer, utilization: Real).
  Processor ::= new (id: Integer).
  Mapping   ::= new (c: Component, p: Processor).

  // The utilization must be > 0
  validUtilization :- c is Component, c.utilization > 0.
  badMapping :- p is Processor,
    s = sum(0.0, { c.utilization |
              c is Component, Mapping(c, p) }), s > 100.

  conforms no badMapping, validUtilization.
}

model m of Mapping
{
  c1 is Component(0, 10).
  c2 is Component(1, 90).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}

partial model pm of Mapping
{
  c1 is Component(0, x).
  c2 is Component(1, y).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}
[EXPLANATION]
- Rule 1: Preserved the `model` and `partial model` blocks exactly as given in the final 
output.
- Rule 2: Acted only on constraints listed in `[CONFLICT MESSAGES]`. Preserved 
`validUtilization` **exactly** as it appears in [CURRENT MODEL], and included it unchanged 
in `conforms`. Chose to remove `invalidUtilization` because `validUtilization` better 
reflects the intended logic of the model.
- Rule 3: Removed `invalidUtilization` as a rule in the `domain` block, `conforms`, and 
all other references.
- Rule 4: Did not modify or remove the `badMapping` constraint since it was not listed in `
[CONFLICT MESSAGES]`
- Rule 5: Verified that `domain` and `conforms` contain the same set of defined 
constraints (`validUtilization`, `badMapping`). Confirmed that `invalidUtilization` is 
fully removed, and at least one conflicting constraint (`validUtilization`) was retained 
without any modifications and matches its definitions and usage in [CURRENT MODEL] exactly.
- Rule 6: Did not include any reasoning for the model before `[FINAL OUTPUT]`. The `
[EXPLANATION]` is fully consistent with the `[FINAL OUTPUT]`. All described changes are 
accurately reflected in the `domain` and `conforms` blocks. A single, complete model was 
produced. No [CURRENT MODEL], `[CONFLICT MESSAGES]`, or partial models were included.
[DEBUG]
LLM response tokens estimate: 10195
[DEBUG] Saved reapired FORMULA model to: 
docs\generated_models\b84ffa609bfe4677864972986f64ae4e.4ml

[DEBUG] Entering tool_planner_node...
[DEBUG] Resetting tool context for fresh model validation.
[DEBUG] Human prompt:
[USER PROMPT]
generate a solution for docs/generated_models/b84ffa609bfe4677864972986f64ae4e.4ml
[PREVIOUS TOOL CALL]
previous_tool =
[PREVIOUS TOOL RESULT]
tool_result =
[DEBUG] Human prompt tokens estimate: 860
[DEBUG] Raw LLM response object: content='{\n    "tool_calls": [\n        {\n            
"name": "load_file",\n            "arguments": {\n                "filename": "docs/
generated_models/b84ffa609bfe4677864972986f64ae4e.4ml"\n            }\n        }\n    ]\n}
' additional_kwargs={} response_metadata={'model': 'llama3.1:latest', 'created_at': 
'2025-07-30T15:43:20.144053Z', 'done': True, 'done_reason': 'stop', 'total_duration': 
28985634100, 'load_duration': 17563619900, 'prompt_eval_count': 501, 
'prompt_eval_duration': 2616375800, 'eval_count': 58, 'eval_duration': 8781521400, 
'model_name': 'llama3.1:latest'} id='run--3d75e651-8c66-4ed1-9454-a8ac9a344941-0' 
usage_metadata={'input_tokens': 501, 'output_tokens': 58, 'total_tokens': 559}
[DEBUG] LLM response tokens estimate: 1060
[DEBUG] Parsed tool_calls: [{'name': 'load_file', 'args': {'filename': 'docs/
generated_models/b84ffa609bfe4677864972986f64ae4e.4ml'}, 'id': 
'599f6314-d5ab-498d-b3e0-4510bfd43b90'}]

[DEBUG] Entering should_call_tool...
[DEBUG] Tool call found. Routing to 'tools' node.

[DEBUG] Entering route_after_tool...
[DEBUG] load_file called. Routing to 'tool_planner' node...

[DEBUG] Entering tool_planner_node...
[DEBUG] Last tool 'load_file'
[DEBUG] Human prompt:
[USER PROMPT]
generate a solution for docs/generated_models/b84ffa609bfe4677864972986f64ae4e.4ml
[PREVIOUS TOOL CALL]
previous_tool = load_file
[PREVIOUS TOOL RESULT]
tool_result =
[DEBUG] Human prompt tokens estimate: 905
[DEBUG] Raw LLM response object: content='{\n    "tool_calls": [\n        {\n            
"name": "solve_file",\n            "arguments": {\n                "filename": "docs/
generated_models/b84ffa609bfe4677864972986f64ae4e.4ml"\n            }\n        }\n    ]\n}
' additional_kwargs={} response_metadata={'model': 'llama3.1:latest', 'created_at': 
'2025-07-30T15:43:33.5221067Z', 'done': True, 'done_reason': 'stop', 'total_duration': 
11847379900, 'load_duration': 61971200, 'prompt_eval_count': 503, 'prompt_eval_duration': 
1336006600, 'eval_count': 58, 'eval_duration': 10443368300, 'model_name': 'llama3.
1:latest'} id='run--4bc66925-33b6-4494-955a-7d3ef0b85e7c-0' usage_metadata=
{'input_tokens': 503, 'output_tokens': 58, 'total_tokens': 561}
[DEBUG] LLM response tokens estimate: 1065
[DEBUG] Parsed tool_calls: [{'name': 'solve_file', 'args': {'filename': 'docs/
generated_models/b84ffa609bfe4677864972986f64ae4e.4ml'}, 'id': 
'6547bd79-96f6-4c6f-b8a2-6c4cbaa1c895'}]

[DEBUG] Entering should_call_tool...
[DEBUG] Tool call found. Routing to 'tools' node.

[DEBUG] Entering route_after_tool...
[DEBUG] solve_file called. Routing to 'tool_planner' node...

[DEBUG] Entering tool_planner_node...
[DEBUG] Last tool 'solve_file'
[DEBUG] Human prompt:
[USER PROMPT]
generate a solution for docs/generated_models/b84ffa609bfe4677864972986f64ae4e.4ml
[PREVIOUS TOOL CALL]
previous_tool = solve_file
[PREVIOUS TOOL RESULT]
tool_result = Parsing text took: 2
Visiting text took: 0
Started solve task with Id 1.
0.21s.
[DEBUG] Human prompt tokens estimate: 1325
[DEBUG] Raw LLM response object: content='{\n    "tool_calls": [\n        {\n            
"name": "extract_solution",\n            "arguments": {\n                "id": 
"1"\n            }\n        }\n    ]\n}' additional_kwargs={} response_metadata={'model': 
'llama3.1:latest', 'created_at': '2025-07-30T15:43:41.5910083Z', 'done': True, 
'done_reason': 'stop', 'total_duration': 7122134500, 'load_duration': 47193900, 
'prompt_eval_count': 531, 'prompt_eval_duration': 1005272100, 'eval_count': 37, 
'eval_duration': 6065152600, 'model_name': 'llama3.1:latest'} 
id='run--eb57be06-88f0-4048-bfc6-7ef53efa1031-0' usage_metadata={'input_tokens': 531, 
'output_tokens': 37, 'total_tokens': 568}
[DEBUG] LLM response tokens estimate: 780
[DEBUG] Parsed tool_calls: [{'name': 'extract_solution', 'args': {'id': '1'}, 'id': 
'eb13ce9d-3b3e-401e-9e6a-b9d16045dd24'}]

[DEBUG] Entering should_call_tool...
[DEBUG] Tool call found. Routing to 'tools' node.

[DEBUG] Entering route_after_tool...
[DEBUG] No conflicts. Routing to END.

[DEBUG] Agent returned state: (omitted for brevity)

Agent: Solution number 0
Component(0, 1/4)
Component(1, 1/4)
Mapping(Component(0, 1/4), Processor(0))
Mapping(Component(1, 1/4), Processor(0))
Processor(0)


0.00s.

You:
```

#### Tool Call Planning and Execution Workflow
- The repair client starts in the `tool_planner` node, where it builds a prompt from the user’s input and checks for any previous tool usage. 
    - Since this is the initial step, both `[PREVIOUS TOOL CALL]` and `[PREVIOUS TOOL RESULT]` are empty.
    - The LLM responds with a `load_file` tool call using the model path from `[USER PROMPT]`.
- The workflow proceeds to `should_call_tool`, detects the tool call, and routes to the `tools` node.
    - After executing `load_file` via the MCP server, control passes to `route_after_tool`, which routes back to `tool_planner` with `[PREVIOUS TOOL CALL] = load_file`.
- A new prompt is constructed including `[PREVIOUS TOOL CALL] = load_file` and an empty `[PREVIOUS TOOL RESULT]`.
    - The LLM then responds with a `solve_file` tool call for the same model.
- The workflow routes again through `should_call_tool` and `tools`, executes `solve_file`, and returns to `tool_planner` via `route_after_tool`.
    - The tool result includes a task id, so `[PREVIOUS TOOL RESULT] = "Started solve task with Id 0."`.
    - The LLM is explicitly instructed to **parse the tool output**, extract the task id (`"0"`), and return an `extract_solution` tool call with it (`"id": "0"`).
- After executing `extract_solution`, the `route_after_tool` node inspects the result, detects **constraint conflicts**, and routes the workflow to `model_repair`.
```txt
Conflicts: Mapping.invalidUtilization  
Conflicts: Mapping.validUtilization
```

#### FORMULA Model Repair and Validation Loop
- In the `model_repair` node:
    - The `[CURRENT MODEL]` and `[CONFLICT MESSAGES]` are passed to the repair LLM, along with the `FORMULA_REPAIR_PROMPT`, to construct a new `.4ml` model that satisfies all constraints. 
    - The LLM returns a `[FINAL OUTPUT]` block containing the updated model and an `[EXPLANATION]` that justifies each change based on the defined repair rules.
    - The revised model is saved to a uniquely named file in `docs/generated_models/`.
- After saving the model, the client resets the tool context and re-enters the `tool_planner` node to validate the fixed model.
    - The workflow routes through the **Tool call planning and execution workflow** sequence using the new model file path.
    - After executing `extract_solution`, `route_after_tool` inspects the result, detects that there are no remaining conflicts, and routes the workflow to `END`.
- The agent then prints the valid solution extracted from the repaired model:
```txt
Agent: Solution number 0
Component(0, 1/4)
Component(1, 1/4)
Mapping(Component(0, 1/4), Processor(0))
Mapping(Component(1, 1/4), Processor(0))
Processor(0)
```

#### Output in `formula_server.py`

```bash
[TOOL DEBUG] load_file called with: docs/conflicts/MappingExample.4ml
domain Mapping
{
  Component ::= new (id: Integer, utilization: Real).
  Processor ::= new (id: Integer).
  Mapping   ::= new (c: Component, p: Processor).

  // The utilization must be > 0
  invalidUtilization :- c is Component, c.utilization <= 0.
  validUtilization :- c is Component, c.utilization > 0.

  badMapping :- p is Processor,
    s = sum(0.0, { c.utilization |
              c is Component, Mapping(c, p) }), s > 100.

  conforms no badMapping, no invalidUtilization, no validUtilization.
}

model m of Mapping
{
  c1 is Component(0, 10).
  c2 is Component(1, 90).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}

partial model pm of Mapping
{
  c1 is Component(0, x).
  c2 is Component(1, y).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}
INFO: (omitted for brevity)
[TOOL DEBUG] solve_file called with: docs/conflicts/MappingExample.4ml
INFO: (omitted for brevity)
[TOOL DEBUG] extract_solution called with task: 0

[TOOL DEBUG] load_file called with: docs/generated_models/b84ffa609bfe4677864972986f64ae4e.4ml
domain Mapping
{
  Component ::= new (id: Integer, utilization: Real).
  Processor ::= new (id: Integer).
  Mapping   ::= new (c: Component, p: Processor).

  // The utilization must be > 0
  validUtilization :- c is Component, c.utilization > 0.
  badMapping :- p is Processor,
    s = sum(0.0, { c.utilization |
              c is Component, Mapping(c, p) }), s > 100.

  conforms no badMapping, validUtilization.
}

model m of Mapping
{
  c1 is Component(0, 10).
  c2 is Component(1, 90).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}

partial model pm of Mapping
{
  c1 is Component(0, x).
  c2 is Component(1, y).
  p1 is Processor(0).
  Mapping(c1, p1).
  Mapping(c2, p1).
}
INFO: (omitted for brevity)
[TOOL DEBUG] solve_file called with: docs/generated_models/b84ffa609bfe4677864972986f64ae4e.4ml
INFO: (omitted for brevity)
[TOOL DEBUG] extract_solution called with task: 1
```

4. Executing "exit" in the terminal ends the session and deletes the `docs/generated_models` folder along with its contents.