import os
import shutil
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local engine imports
from backend.rag_engine import (
    query_rag, 
    build_or_update_vector_store,
    DOCUMENTS_DIR,
    VECTOR_STORE_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-Based Legal Document Assistant API",
    description="Backend service for legal PDF parsing, FAISS indexing, and source-grounded question answering.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the confidentiality term in the NDA?")
    llm_provider: str = Field("openai", example="openai", description="Must be 'openai' or 'ollama'")
    model_name: str = Field("gpt-4o-mini", example="gpt-4o-mini")
    api_key: Optional[str] = Field(None, description="OpenAI API Key (optional if set in env)")
    top_k: int = Field(4, ge=1, le=10)
    temperature: float = Field(0.0, ge=0.0, le=2.0)

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    path: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Legal Document QA API is running.",
        "vector_store_configured": os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss"))
    }

@app.post("/upload", response_model=dict)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more legal PDF documents, save them locally,
    and update the FAISS vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    uploaded_files = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type for '{file.filename}'. Only PDF files are supported."
            )
        
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)
            logger.info(f"Successfully saved {file.filename}")
        except Exception as e:
            logger.error(f"Error saving {file.filename}: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to upload {file.filename}: {str(e)}"
            )

    # Rebuild the FAISS index to include the new document(s)
    try:
        logger.info("Rebuilding vector store index...")
        build_or_update_vector_store()
        logger.info("Vector store index rebuild completed.")
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Files uploaded, but failed to build search index: {str(e)}"
        )

    return {
        "message": f"Successfully uploaded and indexed {len(uploaded_files)} document(s).",
        "uploaded_files": uploaded_files
    }

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Search indexed legal documents and generate a grounded answer using the selected LLM.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # Check LLM provider support
    if request.llm_provider not in ["openai", "ollama"]:
        raise HTTPException(status_code=400, detail="llm_provider must be 'openai' or 'ollama'")

    try:
        logger.info(f"Processing query: '{request.query}' using {request.llm_provider}/{request.model_name}")
        result = query_rag(
            query=request.query,
            llm_provider=request.llm_provider,
            model_name=request.model_name,
            api_key=request.api_key,
            top_k=request.top_k,
            temperature=request.temperature
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except ValueError as ve:
        logger.error(f"Validation error in RAG query: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unhandled error during query execution: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    """
    List all uploaded PDF files and their basic metadata.
    """
    if not os.path.exists(DOCUMENTS_DIR):
        return []
    
    documents = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            try:
                stat_info = os.stat(file_path)
                documents.append(DocumentInfo(
                    filename=filename,
                    size_bytes=stat_info.st_size,
                    path=file_path
                ))
            except Exception as e:
                logger.error(f"Error accessing details of {filename}: {e}")
                
    return documents

@app.delete("/documents/{filename}", response_model=dict)
def delete_document(filename: str):
    """
    Delete an uploaded document and rebuild the vector store.
    """
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
        
    try:
        os.remove(file_path)
        logger.info(f"Deleted document {filename}")
        
        # Sync FAISS store
        build_or_update_vector_store()
        
        return {
            "message": f"Successfully deleted '{filename}' and updated the search index."
        }
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document or rebuild index: {str(e)}"
        )

@app.post("/clear", response_model=dict)
def clear_all():
    """
    Deletes all uploaded documents and wipes the FAISS vector database.
    """
    try:
        # Clear docs directory
        if os.path.exists(DOCUMENTS_DIR):
            for file in os.listdir(DOCUMENTS_DIR):
                file_path = os.path.join(DOCUMENTS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    
        # Clear vector store directory
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR)
            os.makedirs(VECTOR_STORE_DIR)
            
        logger.info("Cleared all documents and vector indices.")
        return {
            "message": "All documents and the search index have been cleared successfully."
        }
    except Exception as e:
        logger.error(f"Error during clearing operation: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")
