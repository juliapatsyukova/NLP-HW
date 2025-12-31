# RAG System for Scientific Literature

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions based on the content of uploaded PDF documents. It uses a Streamlit web interface, with a backend powered by TF-IDF for retrieval and the Mistral API for language model-based answer generation.

## Components

*   `app.py`: A Streamlit application that provides the user interface for file uploads and question-answering.
*   `rag_core.py`: The core module that handles the RAG pipeline. It performs PDF text extraction, document chunking, TF-IDF vectorization for retrieval, and calls the Mistral API to generate answers from retrieved context.
*   `requirements.txt`: A list of all Python dependencies required for the project.

## Requirements

The application requires Python 3 and the packages specified in `requirements.txt`. Key dependencies include:

*   `streamlit`
*   `pypdf`
*   `scikit-learn`
*   `requests`
*   `python-dotenv`

An API key from Mistral AI is also required.

## Installation

1.  Clone the repository.

2.  Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

3.  Create a `.env` file in the root directory of the project and add your Mistral API key and desired model:

    ```
    MISTRAL_API_KEY="your_api_key_here"
    MISTRAL_MODEL="mistral-small-latest"
    ```

## Usage

1.  Run the Streamlit application from the command line:

    ```sh
    streamlit run app.py
    ```

2.  Open the provided local URL in your web browser.

3.  Upload a PDF document using the sidebar.

4.  Click the "Process and Create Index" button to have the system load, chunk, and vectorize the document content.

5.  Once processing is complete, you can ask questions about the document in the main area and receive answers generated from the document's content.
