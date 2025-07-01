from mcp.server.fastmcp import FastMCP
from tools9 import DatabaseOperations, fetch_recipients, send_email_function, extract_email_data_from_response

from tools9 import fetch_recipients, send_email_function, chunks_to_docs_wrapper
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores.oraclevs import OracleVS
import os
mcp = FastMCP("EmailAssistant")

global_vector_store = None

def set_vector_store(store: OracleVS):
    global global_vector_store
    global_vector_store = store


def get_vector_store():
    global global_vector_store   # ← Add this line
    if global_vector_store:
        return global_vector_store

    
    try:
        db_ops = DatabaseOperations()
        if not db_ops.connect():
            raise Exception("DB connect failed")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        global_vector_store = OracleVS(
        embedding_function=embeddings,
        client=db_ops.connection,
        table_name="MY_DEMO4",
        distance_strategy=DistanceStrategy.COSINE,
        )
        return global_vector_store
    except Exception as e:
        print("VectorStore init failed:", e)
        return None



@mcp.tool()
def lookup_recipients(name: str):
    return fetch_recipients(name)

# @mcp.tool()
# def prepare_and_send_email(to: str, subject: str, message: str):
#     return send_email_function({"to": to, "subject": subject, "message": message})


@mcp.tool()
def oracle_connect() -> str:
    """
    Checks and returns Oracle DB connection status.
    """
    try:
        db_ops = DatabaseOperations()
        if db_ops.connect():
            print("Oracle connection successful!")
            return db_ops.connection
        return None
    except Exception as e:
        print(f"Oracle connection failed: {str(e)}")
        return None    

@mcp.tool()
def extract_email_fields_from_response(response_text: str) -> dict:
    """
    Extracts email fields (to, subject, message) from an AI-generated response.

    Input:
    - response_text: A string containing the AI assistant's output.

    Output:
    - A dictionary with keys: "to", "subject", "message"
    """
    try:
        return extract_email_data_from_response(response_text)
    except Exception as e:
        return {"error": f"Failed to extract email data: {str(e)}"}


@mcp.tool()
def store_text_chunks(file_path: str) -> str:
    """Split text and store as embeddings in Oracle Vector Store"""
    try:
        db_ops = DatabaseOperations()
        
        if not db_ops.connect():
            return "❌ Oracle connection failed."

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(raw_text)
            file_name = os.path.basename(file_path)
            docs = [
                chunks_to_docs_wrapper({'id': f"{file_name}_{i}", 'link': f"{file_name} - Chunk {i}", 'text': chunk})
                for i, chunk in enumerate(chunks)
            ]


            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = OracleVS.from_documents(
                docs, embeddings, client=db_ops.connection,
                table_name="MY_DEMO4", distance_strategy=DistanceStrategy.COSINE)
            
            # OracleVS(
            #     embedding_function=embeddings,
            #     client=db_ops.connection,
            #     table_name="MY_DEMO4",
            #     distance_strategy=DistanceStrategy.COSINE,
            # )

            set_vector_store(vector_store)

            return f"✅ Stored {len(docs)} chunks from {file_name}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

@mcp.tool()
def rag_search(query: str) -> str:
    """
    Retrieve relevant information from user-uploaded documents stored in the Oracle Vector Store.

    Use this tool whenever a user asks a question that may be answered from the uploaded documents
    (e.g., HR policy files, contracts, technical manuals, PDF uploads, etc.).

    The tool performs a semantic similarity search over the embedded document chunks and returns
    the top 5 most relevant text snippets.

    Input:
    - A natural language question or topic from the user.

    Output:
    - A formatted string combining the most relevant document excerpts.

    Examples:
    - "What is the leave policy for new employees?"
    - "Summarize the refund terms in the uploaded contract"
    - "Find safety precautions mentioned in the manual"
    """
    try:
        # Load vector store (or access from persistent source if needed)
        vector_store = get_vector_store()
        if vector_store is None:
            return "❌ No documents have been indexed yet."

        docs = vector_store.similarity_search(query, k=5)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"❌ Error during document search: {str(e)}"


if __name__ == "__main__":
    print(" Starting MCP Agentic Server ...")
    mcp.run(transport="stdio")

