# Console RAG Agent for Scientific Literature

This project provides a console-based Retrieval-Augmented Generation (RAG) agent for querying scientific literature on amyloidogenicity and protein aggregation. The agent uses a tool-based architecture, leveraging a large language model via the Groq API for reasoning and the Qwen-Agent framework for autonomous tool selection. The knowledge base is built on ChromaDB.

## Components

*   `build_kb.py`: A script to create a ChromaDB vector store from PDF documents located in the `sample_docs/` directory. This script must be run once to initialize the knowledge base.
*   `agent_tools.py`: Defines the set of tools the agent can use to interact with the knowledge base. These include functions for searching (`rag_search`), checking status (`kb_status`), and retrieving statistics (`kb_stats`). It also includes wrapper classes to make these functions compatible with the Qwen-Agent framework.
*   `agent_app.py`: The main application that runs the interactive console agent. It handles user queries, orchestrates the tool-calling loop with the LLM, and displays the final response.
*   `requirements.txt`: A list of all Python dependencies required for the project.

## Requirements

*   Python 3
*   A Groq API key
*   PDF documents for the knowledge base

The Python package dependencies are listed in the `requirements.txt` file.

## Setup

1.  **Install Dependencies**: Install the necessary Python packages.

    ```sh
    pip install -r requirements.txt
    ```

2.  **API Key**: Create a `.env` file in the project's root directory and add your Groq API key:

    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

3.  **Add Documents**: Place the PDF files you want to include in the knowledge base into a `sample_docs/` directory.

4.  **Build Knowledge Base**: Run the `build_kb.py` script to process the PDFs and create the ChromaDB vector store. This will create a `chroma_db/` directory.

    ```sh
    python3 build_kb.py
    ```

## Usage

After completing the setup, run the agent application:

```sh
python3 agent_app.py
```

The application will start an interactive console. You can ask questions related to the content of your documents, and the agent will use its tools to find relevant information and generate an answer.
