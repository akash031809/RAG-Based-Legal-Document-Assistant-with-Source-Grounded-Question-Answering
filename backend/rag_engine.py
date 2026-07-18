import os
import shutil
import logging
from typing import List, Dict, Any, Tuple
import pdfplumber
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configurations
DATA_DIR = os.getenv("DATA_DIR", "data")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "data/documents")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize directories
def initialize_dirs():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

initialize_dirs()

# Global Embeddings instance to avoid reloading multiple times
logger.info("Loading Sentence Transformers embeddings model (all-MiniLM-L6-v2)...")
try:
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    logger.info("Embeddings model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading embeddings model: {e}")
    embeddings_model = None


def extract_pdf_pages(file_path: str, filename: str) -> List[Document]:
    """
    Extracts text from each page of a PDF using pdfplumber,
    and returns a list of Document objects with page metadata.
    """
    documents = []
    logger.info(f"Extracting text from {file_path}")
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text()
            if text and text.strip():
                # Store page number and filename in metadata for grounding
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num
                    }
                )
                documents.append(doc)
    
    logger.info(f"Extracted {len(documents)} text-containing pages from {filename}")
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Chunks documents using RecursiveCharacterTextSplitter.
    Chunks are generated per-page to maintain precise page grounding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunked_docs = []
    for doc in documents:
        # Split each page's content individually
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata.copy()  # Inherit source and page
                )
            )
            
    logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} pages")
    return chunked_docs


def build_or_update_vector_store():
    """
    Rebuilds the entire FAISS vector store from all PDFs in the documents directory.
    This ensures no stale embeddings or deleted documents persist.
    """
    initialize_dirs()
    
    # 1. List all PDFs in documents dir
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        # If no documents, clear the vector store directory
        logger.info("No documents found. Clearing vector store index.")
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR)
            os.makedirs(VECTOR_STORE_DIR)
        return
        
    all_chunks = []
    for filename in pdf_files:
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        try:
            pages = extract_pdf_pages(file_path, filename)
            chunks = chunk_documents(pages)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            
    if not all_chunks:
        logger.warning("No text chunks extracted from any PDF documents.")
        return
        
    logger.info(f"Indexing {len(all_chunks)} chunks to FAISS vector store...")
    db = FAISS.from_documents(all_chunks, embeddings_model)
    db.save_local(VECTOR_STORE_DIR)
    logger.info("FAISS index saved successfully.")


def get_vector_store() -> FAISS:
    """
    Loads the local FAISS vector store. Returns None if it doesn't exist.
    """
    if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
        return None
    
    try:
        db = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None


def get_llm(llm_provider: str, model_name: str, api_key: str = None, temperature: float = 0.0):
    """
    Returns an initialized LLM chat model based on user selection.
    """
    if llm_provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key is missing. Please provide it in settings or config.")
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            openai_api_key=key
        )
    elif llm_provider == "ollama":
        # Streamlit or local defaults
        model = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def query_rag(
    query: str,
    llm_provider: str,
    model_name: str,
    api_key: str = None,
    top_k: int = 4,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Queries the RAG system:
    1. Retrieves relevant chunks from FAISS vector store.
    2. Constructs a prompt with the context.
    3. Feeds context and question to the LLM.
    4. Returns LLM answer along with grounded source citations.
    """
    db = get_vector_store()
    if not db:
        return {
            "answer": "No documents have been indexed yet. Please upload PDF files to get started.",
            "sources": []
        }
        
    # Retrieve top K documents with similarity scores (relevance scores)
    try:
        retrieved_docs_with_scores = db.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception as e:
        logger.error(f"Error searching vector store: {e}")
        return {
            "answer": f"Error performing search: {str(e)}",
            "sources": []
        }
        
    # Format retrieved sources
    sources = []
    context_chunks = []
    
    for idx, (doc, score) in enumerate(retrieved_docs_with_scores):
        filename = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 0)
        content = doc.page_content
        
        # Keep track of formatted context and metadata sources
        context_chunks.append(f"Source: {filename} (Page {page})\nContent: {content}")
        sources.append({
            "id": idx + 1,
            "filename": filename,
            "page": page,
            "content": content,
            "score": float(score)  # Convert numpy float to python float
        })
        
    context_str = "\n\n---\n\n".join(context_chunks)
    
    # 2. Define the LLM Prompt Template
    system_prompt = (
        "You are an expert legal assistant. Your task is to answer the user's question using ONLY the provided document context.\n\n"
        "Rules for your answer:\n"
        "1. Base your answer STRICTLY on the context provided below. Do not use any outside knowledge.\n"
        "2. If the context does not contain enough information to answer, state clearly: 'I cannot find the answer to this question in the uploaded documents.'\n"
        "3. Cite your sources directly in your response using the format '[Filename, Page X]' (for example, '[Service_Agreement.pdf, Page 2]') whenever you reference a fact or clause from a document.\n"
        "4. Keep your answer professional, clear, and well-structured.\n\n"
        "Context:\n"
        "{context}\n\n"
        "User Question: {question}\n"
        "Grounded Answer:"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # 3. Initialize LLM and Chain
    try:
        llm = get_llm(llm_provider, model_name, api_key, temperature)
        chain = prompt | llm | StrOutputParser()
        
        # 4. Generate answer
        answer = chain.invoke({
            "context": context_str,
            "question": query
        })
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        answer = f"Error generating answer from LLM ({llm_provider}): {str(e)}"
        
    return {
        "answer": answer,
        "sources": sources
    }
