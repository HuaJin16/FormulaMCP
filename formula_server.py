import formulallm.formula as f
from fastmcp import FastMCP

mcp = FastMCP("formula_mcp")

@mcp.tool
def load_file(filename: str) -> str:
    """Returns the output of Loading a FORMULA model"""
    print(f"[TOOL DEBUG] load_file called with: {filename}")
    return f.load(filename)

if __name__ == "__main__":
    print("🚀 Starting server...")
    mcp.run("sse")