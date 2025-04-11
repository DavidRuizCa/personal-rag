import time
import pandas as pd


# Function to delete embeddings for a specific document
def delete_embeddings_for_document(index, doc_name):
    """Deletes all embeddings for a specific document using its namespace."""
    index.delete(namespace=doc_name)
    
    return f"All embeddings for document {doc_name} have been deleted."

# Function to retrieve document names from Pinecone
def get_documents_names(index):
    """Retrieves the list of document names (namespaces) from the Pinecone index."""
    namespaces = index.describe_index_stats()['namespaces'].keys()
    namespaces = list(namespaces)

    return [[name] for name in namespaces] if namespaces else [["No documents available"]]

# Function to refresh the list of document names
def refresh_documents_names(index):
    """Refreshes the list of document names by monitoring changes in the Pinecone index."""
    start_time = time.time()
    namespaces = index.describe_index_stats()['namespaces'].keys()
    n_docs = len(namespaces)

    while time.time() - start_time < 100:
        new_namespaces = index.describe_index_stats()['namespaces'].keys()
        if len(new_namespaces) != n_docs:
            return [[name] for name in new_namespaces] if new_namespaces else [["No documents available"]]
        time.sleep(1)

    return [["No documents available"]] if not namespaces else [[name] for name in namespaces]

# Function to delete a selected document
def delete_selected_document(index, selected_rows):
    """Deletes the selected document(s) from the Pinecone index."""
    if isinstance(selected_rows, pd.DataFrame):
        selected_rows = selected_rows.values.tolist()  # Convert DataFrame to list

    if isinstance(selected_rows, list) and len(selected_rows) > 0:
        selected_rows = selected_rows[0]  # Extract first element

    if not selected_rows or selected_rows == "No documents available":
        return get_documents_names(index)

    # Delete the documents
    for namespace in selected_rows:
        index.delete(delete_all=True, namespace=namespace)
    
    return get_documents_names(index)