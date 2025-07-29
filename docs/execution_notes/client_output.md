# Execution Overview: Running the FORMULA Client and Server

This document outlines the expected outputs and interactions between the client and the backend server. While actual responses may vary due to LLM behavior, the examples and explanations below represent the intended flow of execution.

## formula_client.py

This script initializes a local MCP client that accepts C code and natural language descriptions as input and uses a LangGraph-based agent to generate and iteratively refine corresponding FORMULA `.4ml` models. It integrates with MCP server tool calls to execute FORMULA's load command for model validation.

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

2. Expected output after starting the MCP client `python formula_client.py`

```bash
[DEBUG] Initializing MCPClient...
Connecting to MCP server at http://127.0.0.1:8000...
[DEBUG] SYSTEM PROMPT:

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

        [REFERENCE EXAMPLES]
        Use the example translation pairs below to learn **patterns** of abstraction and constraint formulation from C code and natural language
        into FORMULA models.

--- EXAMPLE: ArmStrong ---
### ArmStrongDescription.txt ###
Description: An Armstrong number of three digits is an integer such that the sum of the cubes of its digits is equal to the number itself.
### ArmStrongExampleC.txt ###
#include <stdio.h>
#include <math.h>

int main() {{
    int A, B, C;

    for (A = 1; A <= 9; A++) {{         // Hundreds place (cannot be 0)
        for (B = 0; B <= 9; B++) {{     // Tens place
            for (C = 0; C <= 9; C++) {{ // Units place
                int sumOfCubes = pow(A, 3) + pow(B, 3) + pow(C, 3);
                int valueFromDigits = 100 * A + 10 * B + C;

                if (sumOfCubes == valueFromDigits) {{
                    printf("%d%d%d\n", A, B, C);
                }}
            }}
        }}
    }}

    return 0;
}}
### ArmStrongExampleFormula.4ml ###
domain Armstrong {{
  Armstrong ::= new (a: Integer, b: Integer, c: Integer).

  goodValues :- gv is Armstrong, gv.a >= 1, gv.a <= 9, gv.b >= 0, gv.b <= 9, gv.c >= 0, gv.c <= 9.
  goodSolution :- gs is Armstrong, gs.a * 100 + gs.b * 10 + gs.c = gs.a * gs.a * gs.a + gs.b * gs.b * gs.b + gs.c * gs.c * gs.c.

  conforms goodValues, goodSolution.
}}

partial model pm of Armstrong {{
  Armstrong(a, b, c).
}}

[DEBUG] Formula_Gen prompt tokens estimate: 13785
[DEBUG] System prompt tokens estimate: 4005

[DEBUG] Initializing LangGraph agent...
[DEBUG] Obtaining tools:
[DEBUG] - load_file: Returns the output of Loading a FORMULA model
[DEBUG] - solve_file: Returns the output of Solving a FORMULA partial model
[DEBUG] - extract_solution: Returns the first solution from a solve task
[DEBUG] Compiling LangGraph...
[DEBUG] Graph compiled successfully.

[DEBUG] Starting interactive loop...
Chat session started. Type 'exit' to quit.
Paste your C code and natural language description. End with an "END" on a new line
```

- The client first connects to the MCP server started in the previous step.
- It then prints the system prompt that will guide the language model's behavior during translation.
    - This prompt includes a full example demonstrating how a natural language description and C code (for the Armstrong puzzle) are translated into a valid FORMULA model.
- After that, LangGraph retrieves the available FORMULA tools registered on the MCP server, which the client can use to validate and analyze generated models.

3. Expected behavior after pasting input from docs/inputs/SMM.txt into `formula_client.py`

```bash
You: <pasted contents of SMM.txt>
...

[DEBUG] Entering FormulaGen_node (generating formula model)...
[DEBUG] Human prompt:

            [C SOURCE CODE + NATURAL LANGUAGE DESCRIPTION]

            C Source Code:
#include <stdio.h>
#include <stdbool.h>

bool is_unique(int digits[], int len) {
    for (int i = 0; i < len; i++)
        for (int j = i + 1; j < len; j++)
            if (digits[i] == digits[j])
                return false;
    return true;
}

int main() {
    int S, E, N, D, M, O, R, Y;
    int digits[8];

    for (S = 1; S <= 9; S++)
    for (E = 0; E <= 9; E++)
    for (N = 0; N <= 9; N++)
    for (D = 0; D <= 9; D++)
    for (M = 1; M <= 9; M++)
    for (O = 0; O <= 9; O++)
    for (R = 0; R <= 9; R++)
    for (Y = 0; Y <= 9; Y++) {
        digits[0] = S; digits[1] = E; digits[2] = N; digits[3] = D;
        digits[4] = M; digits[5] = O; digits[6] = R; digits[7] = Y;

        if (!is_unique(digits, 8)) continue;

        int SEND = 1000*S + 100*E + 10*N + D;
        int MORE = 1000*M + 100*O + 10*R + E;
        int MONEY = 10000*M + 1000*O + 100*N + 10*E + Y;

        if (SEND + MORE == MONEY) {
            printf("  SEND = %d\n", SEND);
            printf("  MORE = %d\n", MORE);
            printf(" MONEY = %d\n", MONEY);
            printf("Mapping: S=%d E=%d N=%d D=%d M=%d O=%d R=%d Y=%d\n",
                   S,E,N,D,M,O,R,Y);
        }
    }

    return 0;
}

Natural Language Description:
The SEND + MORE = MONEY puzzle is a classic cryptarithmetic or verbal arithmetic puzzle in
which each letter represents a unique digit from 0 to 9. The goal is to assign digits to
letters such that the following equation holds true:

  SEND
+ MORE
--------
 MONEY

Each letter in the puzzle must be assigned exactly one digit, and the assignment must
follow a few key constraints. No two letters can represent the same digit, and all
occurrences of a given letter must use the same digit consistently throughout the
equation. Additionally, the leading letters—S in "SEND" and M in "MORE" and "MONEY"—cannot
be assigned the digit 0. The challenge is to find a digit assignment that satisfies the
equation SEND + MORE = MONEY while adhering to all of these rules.

END

            [PREVIOUS ATTEMPTS]

[DEBUG] Human prompt tokens estimate: 10665
[DEBUG] Raw LLM response object: content=(omitted for brevity)

[DEBUG] Entering model_loader_node...
[DEBUG] Model:(omitted for brevity)
[DEBUG] Saved generated FORMULA model as: 8cfa1204f3644780a981ef8c3e52fa9d.4ml
[DEBUG] Sending prompt to LLM: load docs/generated_models/8cfa1204f3644780a981ef8c3e52fa9d.4ml
[DEBUG] Raw LLM response object: content='{\n  "tool_calls": [\n    {\n      "name":
"load_file",\n      "arguments": {\n        "filename": "docs/generated_models/
8cfa1204f3644780a981ef8c3e52fa9d.4ml"\n      }\n    }\n  ]\n}' additional_kwargs={}
response_metadata={'model': 'llama3.1', 'created_at': '2025-07-29T15:38:41.0684413Z',
'done': True, 'done_reason': 'stop', 'total_duration': 23464816600, 'load_duration':
13369529200, 'prompt_eval_count': 212, 'prompt_eval_duration': 1513849900, 'eval_count':
63, 'eval_duration': 8564240600, 'model_name': 'llama3.1'} id='run--224a5807-bd43-4e65-a2c5-026c510d7fa4-0' 
usage_metadata={'input_tokens': 212,'output_tokens': 63, 'total_tokens': 275}
[DEBUG] JSON data successfully parsed
[DEBUG] Parsed tool_calls: [{'name': 'load_file', 'args': {'filename': 'docs/
generated_models/8cfa1204f3644780a981ef8c3e52fa9d.4ml'}, 'id':
'b9c0ca64-403a-4698-b19d-6c1b03ade3cb'}]

[DEBUG] Entering should_call_tool...
[DEBUG] Tool call found. Routing to 'tools' node.

[DEBUG] Entering load_result_node...
[DEBUG] Generated Model (omitted for brevity)
[DEBUG] Tool Result
(Compiled) 8cfa1204f3644780a981ef8c3e52fa9d
0.66s
[DEBUG] Model compiled successfully. Ending.

[DEBUG] Agent returned state: (omitted for brevity)

[Agent Output]:
-------- MODEL --------
(omitted for brevity)
-------- RESULT --------
(Compiled) 8cfa1204f3644780a981ef8c3e52fa9d.4ml
-----------------------

Chat session started. Type 'exit' to quit.
Paste your C code and natural language description. End with an "END" on a new line
```

- The client begins in the `FormulaGen` node, where it builds a prompt from the user's input (C source code and natural language description), along with any previous failed attempts. Each attempt includes the generated model and the associated error returned by the `"load_file"` tool. The LLM then generates a translated FORMULA `.4ml` model based on this prompt.
- It then enters the `model_loader` node, which:
    - Saves the generated model to a  uniquely named `.4ml` file in `docs/generated_models/`.
    - Constructs a new user prompt instructing the LLM to load the saved `.4ml` file by path.
    - Parses the LLM’s `"load_file"` tool call and appends it to the agent state for execution in the next node.
- The client then enters the `should_call_tool` node, which checks the latest message for a valid `"load_file"` tool call. If found, it routes to the tools node; otherwise, the workflow ends.
    - In the tools node, a prebuilt `ToolNode` automatically executes the requested MCP tool using the parsed arguments. The result is wrapped in a `ToolMessage` and appended to the agent state.
- After execution, the client enters the `load_result` node, which logs the tool’s response (e.g., `"(Compiled)"` or an error) alongside the generated model in models_results, and increments the iterations counter.
- Next, the `should_attempt_again` node evaluates the result:
    - If the response contains `"(Compiled)"`, the model is considered valid and execution ends.
    - If the response contains `"Error"` and the retry limit hasn't been reached, the client loops back to `FormulaGen` to attempt a fix.
    - Otherwise, the workflow terminates.

Output in `formula_server.py`
```bash
[TOOL DEBUG] load_file called with: docs/generated_models/8cfa1204f3644780a981ef8c3e52fa9d.4ml
domain SENDMore {
  SENDMore ::= new (s: Integer, e: Integer, n: Integer, d: Integer, m: Integer, o: Integer, r: Integer, y: Integer).

  digitsAreUnique :- ds is SENDMore, ds.s >= 1, ds.s <= 9, ds.e >= 0, ds.e <= 9, ds.n >= 0, ds.n <= 9, ds.d >= 0, ds.d <= 9, ds.m >= 1, ds.m <= 9, ds.o >= 0, ds.o <= 9, ds.r >= 0, ds.r <= 9, ds.y >= 0, ds.y <= 9, ds.s != ds.e, ds.s != ds.n, ds.s != ds.d, ds.s != ds.m, ds.s != ds.o, ds.s != ds.r, ds.s != ds.y, ds.e != ds.n, ds.e != ds.d, ds.e != ds.m, ds.e != ds.o, ds.e != ds.r, ds.e != ds.y, ds.n != ds.d, ds.n != ds.m, ds.n != ds.o, ds.n != ds.r, ds.n != ds.y, ds.d != ds.m, ds.d != ds.o, ds.d != ds.r, ds.d != ds.y, ds.m != ds.o, ds.m != ds.r, ds.m != ds.y, ds.o != ds.r, ds.o != ds.y, ds.r != ds.y.

  equationHolds :- eq is SENDMore, eq.s * 1000 + eq.e * 100 + eq.n * 10 + eq.d + eq.m * 1000 + eq.o * 100 + eq.r * 10 + eq.e = eq.m * 10000 + eq.o * 1000 + eq.n * 100 + eq.e * 10 + eq.y.

  conforms digitsAreUnique, equationHolds.
}

partial model pm of SENDMore {
  SENDMore(s, e, n, d, m, o, r, y).
}
```

4. Executing "exit" in the terminal ends the session and deletes the `docs/generated_models` folder along with its contents.
