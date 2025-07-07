import os
import re
import formulallm.formula as f
from fastmcp import FastMCP

mcp = FastMCP("formula_mcp")

@mcp.tool
def load_file(filename: str) -> str:
    """Returns the output of Loading a FORMULA model"""
    print(f"[TOOL DEBUG] load_file called with: {filename}")
    return f.load(filename)

@mcp.tool
def solve_file(filename: str) -> str:
    """Returns the output of Solving a FORMULA partial model"""
    print(f"[TOOL DEBUG] solve_file called with: {filename}")

    # load contents from disk
    abs_path = os.path.abspath(filename)
    with open(abs_path, 'r') as file:
        file_contents = file.read()

    # regular expression searching
    pattern = r"partial model\s(\w+)\sof\s(\w+)"
    match = re.search(pattern, file_contents)
    partial_model_name = match.group(1)
    goals = f"{match.group(2)}.conforms"

    return f.solve(partial_model_name, "1", goals)

@mcp.tool
def extract_solution(id: str) -> str:
    """Returns the first solution from a solve task"""
    print(f"[TOOL DEBUG] extract_solution called with task: {id}")
    return f.extract(id, "0", "0")

if __name__ == "__main__":
    print("🚀 Starting server...")
    mcp.run("sse")