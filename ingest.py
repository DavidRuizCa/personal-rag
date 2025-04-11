import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
from vector_store import refresh_documents_names
from loguru import logger
from config import EMBEDDINGS_MODEL

# Function to extract text from uploaded PDF files
def extract_text_from_pdfs(pdf_files):
    """Extracts text from a list of uploaded PDF files."""
    logger.info("Extracting text from PDF files...")
    doc_texts = {}
    
    for pdf_file in pdf_files:
        # Extract the file name without the path
        doc_name = os.path.basename(pdf_file.name)
        
        # Read the PDF and extract text from each page
        reader = pypdf.PdfReader(pdf_file.name)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        doc_texts[doc_name] = text  # Store text with the document name as the key

    return doc_texts

# Function to split text into smaller chunks for processing
def split_text(doc_texts):
    """Splits text into smaller chunks for processing."""
    logger.info("Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=50,
    )

    doc_chunks = {}
    for doc_name, text in doc_texts.items():
        chunks = splitter.split_text(text)
        doc_chunks[doc_name] = chunks

    return doc_chunks

# Function to generate embeddings for text chunks
def embed(openai, doc_chunks):
    """Generates embeddings for text chunks."""
    logger.info("Generating embeddings for text chunks...")
    all_chunks = [] # List to store all text chunks
    doc_map = [] # Map to track which document each chunk belongs to
    
    for doc_name, chunks in doc_chunks.items():
        all_chunks.extend(chunks)
        doc_map.extend([doc_name] * len(chunks))
    
    # Generate embeddings
    res = openai.embeddings.create(
        input=all_chunks,
        model=EMBEDDINGS_MODEL
    )
    
    # Organize embeddings by document
    doc_embeds = {doc_name: [] for doc_name in doc_chunks}
    for doc_name, embedding in zip(doc_map, res.data):
        doc_embeds[doc_name].append(embedding.embedding)

    return doc_embeds

# Function to process embeddings and upload them to Pinecone
def process_embeddings(index, doc_chunks, doc_embeds):    
    """Processes embeddings and uploads them to Pinecone."""
    logger.info("Processing and uploading embeddings to Pinecone...")
    batch_size = 200  # Number of vectors to upload in each batch

    for doc_name, chunks in doc_chunks.items():
        embeddings = doc_embeds[doc_name]
        vectors = []

        # Prepare vectors for Pinecone
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc_name}_{i}",
                "values": embedding,
                "metadata": {"document": doc_name, "text": chunk}
            })
        
        # Upload vectors to Pinecone in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=doc_name)
                logger.info(f"Uploaded batch {i // batch_size + 1} for document: {doc_name}")
            except Exception as e:
                logger.error(f"Error uploading batch {i // batch_size + 1} for {doc_name}: {e}")

# Function to process uploaded documents
def process_loaded_documents(openai, index, files):
    """Processes uploaded documents by extracting text, generating embeddings, and uploading to Pinecone."""
    logger.info("Processing uploaded documents...")

    doc_texts = extract_text_from_pdfs(files)
    doc_chunks = split_text(doc_texts)
    doc_embeds = embed(openai, doc_chunks)
    process_embeddings(index,doc_chunks, doc_embeds)
    return refresh_documents_names(index)