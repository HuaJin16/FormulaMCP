# FormulaMCP - Local MCP Client using Ollama + LangGraph

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

### Install Formula
```bash
$ dotnet tool install --global VUISIS.Formula.<x64|ARM64> 
```

### Ollama Setup
```bash
# Download and install Ollama from https://ollama.com/
# Then run the model locally:
$ ollama run llama3.2
```

### Client Usage
```bash
# Start the MCP server
$ python formula_server.py
```
```bash
# Run the client
$ python formula_client.py
```
```txt
Copy the contents of docs/inputs/SMM.txt and paste them into the client
```

### Repair Usage
```bash
# Start the MCP server
$ python formula_server.py
```
```bash
# Run the repair
$ python formula_repair.py
```
```txt
You: "load docs/conflicts/MappingExample.4ml"
You: "solve docs/conflicts/MappingExample.4ml"
You: "extract <task_id>"
# Replace <task_id> with the Id returned from the solve command
```