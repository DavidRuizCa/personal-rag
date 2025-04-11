import uuid

import gradio as gr
from openai import OpenAI
from pinecone import Pinecone

from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore

from config import PINECONE_API_KEY, INDEX_NAME, EMBEDDINGS_MODEL, LLM_MODEL, OPENAI_API_KEY
from graph import build_graph, query_graph
from vector_store import delete_selected_document, get_documents_names
from ingest import process_loaded_documents

# Initialize Pinecone and OpenAI clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings_model = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
vector_store = PineconeVectorStore(index=index, embedding=embeddings_model)
llm = init_chat_model(model=LLM_MODEL, temperature=0.0)
openai = OpenAI(api_key=OPENAI_API_KEY)

# Configuration for the graph
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
graph = build_graph(vector_store, index, llm)

# Define helper functions
def make_query_fn(graph, config):
    """Creates a function to query the graph."""
    def query_fn(message, history):
        return query_graph(graph, config, message, history)
    return query_fn

def make_get_documents_names_fn(index):
    """Creates a function to fetch document names from the vector store."""
    def get_documents_names_fn():
        return get_documents_names(index)
    return get_documents_names_fn

def make_delete_and_refresh_fn(index):
    """Creates a function to delete selected documents and refresh the list."""
    def delete_and_refresh_list(selected_rows):
        return delete_selected_document(index, selected_rows), gr.update(interactive=False)
    return delete_and_refresh_list

def make_process_fn(openai, index):
    """Creates a function to process uploaded documents."""
    def process_loaded_documents_fn(pdf_files):
        return process_loaded_documents(openai, index, pdf_files)
    return process_loaded_documents_fn

def enable_delete(evt: gr.SelectData, docs):
    """Enables the delete button when a document is selected."""
    return gr.update(interactive=True), docs

# Define Gradio layout
with gr.Blocks() as demo:
    # Row for uploading and processing documents
    with gr.Row():
        with gr.Column():
            pdf_input = gr.Files(label="Upload your PDFs")
            process_button = gr.Button("Process Documents")
        with gr.Column():
            doc_list = gr.Dataframe(
                headers=["Loaded Documents"],
                datatype=["str"],
                row_count=(1, "dynamic"),
                interactive=True,
            )
            delete_button = gr.Button("Delete Document", interactive=False)

    # Row for Chatbot interaction
    with gr.Row():
        chatbot = gr.Chatbot(label="Chat with AI", height=400)
    with gr.Row():
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here and press Enter",
            lines=1,
        )
    send_btn = gr.Button("Send", variant="primary")

    # Set up interactions
    query = make_query_fn(graph, config)
    send_btn.click(query, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
    user_input.submit(query, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
    
    get_documents_names_fn = make_get_documents_names_fn(index)
    demo.load(get_documents_names_fn, outputs=doc_list)

    doc_list.select(enable_delete, inputs=[doc_list], outputs=[delete_button, doc_list])

    delete_and_refresh_list = make_delete_and_refresh_fn(index)
    delete_button.click(delete_and_refresh_list, inputs=[doc_list], outputs=[doc_list, delete_button])

    process_loaded_documents_fn = make_process_fn(openai, index)
    process_button.click(process_loaded_documents_fn, inputs=[pdf_input], outputs=doc_list, show_progress='full')

demo.launch(inbrowser=True)