# ollama-mcp – Locally Hosted MCP Client with Ollama + LangChain

### Create virtual environment

```bash
# Linux, macOS, and Windows (WSL)
$ python -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip

# Windows (Powershell)
$ py -3.12 -m venv venv
$ .\venv\Scripts\activate
$ python.exe -m pip install --upgrade pip
```

### Install Dependencies
```bash
$ pip install -r requirements.txt
```

### Ollama Setup
```bash
# Download and install Ollama from https://ollama.com/
# Then run the model locally:
$ ollama run llama3.2
```

### Getting Started
```bash
# Start the MCP server
python server.py --server_type=sse

# Run the client
python langchain_client.py
```

### Example Usage
#### Add a record:
```txt
# Prompt:
"Add John Doe 30 year old Engineer"
```
```sql
# The agent will generate and run:
INSERT INTO people (name, age, profession) VALUES ('John Doe', 30, 'Engineer')
```
#### View records:
```txt
# Prompt:
"Show all records"
```
```sql
# The agent will return a formatted table from:
SELECT * FROM people
```