import os
from dotenv import load_dotenv

load_dotenv(override=True)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # Pinecone API key
INDEX_NAME = os.getenv('INDEX_NAME')  # Pinecone index name

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # OpenAI API key
LLM_MODEL= "gpt-4o-mini" # Model for the LLM
EMBEDDINGS_MODEL = "text-embedding-3-small" # Model for embeddings generator