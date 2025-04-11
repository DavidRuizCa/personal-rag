# Personal RAG System – Retrieval-Augmented Generation with AI

This application implements a Retrieval-Augmented Generation (RAG) system using AI. It allows users to upload documents, process them into embeddings, and interact with an AI chatbot that retrieves relevant information from the uploaded documents to answer user queries.

## Features

1. **Document Upload and Processing**: Upload PDF documents, extract their content, split them into chunks, and generate embeddings.
2. **Vector Storage**: Store and manage document embeddings in Pinecone for efficient similarity search.
3. **AI Chatbot**: Interact with an AI chatbot that retrieves relevant document content to answer user queries.
4. **Document Management**: View, delete, and refresh the list of uploaded documents.

## How It Works

1. Upload PDF documents using the Gradio interface.
2. The app extracts text from the documents and generates embeddings using OpenAI's embedding model.
3. The embeddings are stored in Pinecone for efficient retrieval.
4. Interact with the AI chatbot to ask questions. The chatbot retrieves relevant content from the documents and generates concise answers.
5. Want to update your knowledge base? Simply delete the documents you no longer need and upload new ones.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DavidRuizCa/personal-rag.git
   ```
2. Navigate to the project folder:
   ```bash
   cd personal-rag
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   INDEX_NAME=your_index_name
   ```


## Usage

1. Run the application:
   ```bash
   python app.py
   ```
2. Open the Gradio interface in your browser.
3. Upload PDF documents and click "Process Documents" to generate embeddings.
4. Use the chatbot to ask questions and retrieve relevant information.

## Requirements

- Python 3.8 or higher
- Required Python libraries (install via `requirements.txt`):
  - `gradio`
  - `openai`
  - `pinecone-client`
  - `langchain`
  - `pypdf`
  - `loguru`
  - `python-dotenv`

## Folder Structure

```
Personal-RAG/
├── app.py                # Main application file
├── graph.py              # Defines the state graph for query handling
├── vector_store.py       # Functions for managing document embeddings in Pinecone
├── ingest.py             # Handles document ingestion and embedding generation
├── retriever.py          # Implements the retrieval tool for querying the vector store
├── config.py             # Configuration file for API keys and model settings
├── README.md             # Documentation for the app
├── requirements.txt      # List of dependencies
```

## Notes

- Ensure you have an active internet connection to process documents and interact with the chatbot.
- The app uses OpenAI's GPT model for generating responses. Make sure to set up your API key in the environment variables.
- The app uses Pinecone as a vector data base for storing and retrieving the document embeddings. Make sure to set up your API key and index in the environment variables.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

Developed by [David Ruiz Casares](https://www.linkedin.com/in/david-ruiz-casares/).
