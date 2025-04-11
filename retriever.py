from langchain_core.tools import tool
from vector_store import get_documents_names
from loguru import logger


def build_retrieve_tool(vector_store, index):
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):    
        """Use this tool when you need to answer questions based on document contents.
        It searches through a vector store and retrieves the most relevant document chunks related to the user's query."""

        logger.info(f"Retrieving documents for query: {query}")
        serialized = "\n\n"
        retrieved_docs = []
        try:
            for namespace in get_documents_names(index):
                docs = (vector_store.similarity_search(
                    query, 
                    k=2,
                    namespace=namespace[0] if isinstance(namespace, (list, tuple)) else namespace
                ))
                if docs:
                    retrieved_docs.extend(docs)
                    for doc in docs:
                        serialized += f"Source: {doc.metadata}\nContent: {doc.page_content}\n\n"
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return "An error occurred while retrieving documents.", []

        return serialized, retrieved_docs

    return retrieve
    