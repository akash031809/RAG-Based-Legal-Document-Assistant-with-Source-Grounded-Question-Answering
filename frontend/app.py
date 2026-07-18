import os
import sys
import requests
import streamlit as nn
from dotenv import load_dotenv
from pathlib import Path

# Add root folder to sys.path to allow importing backend module in Streamlit Cloud or direct runs
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load env variables
load_dotenv()

# On Streamlit Cloud, load secrets from st.secrets (overrides .env)
try:
    if hasattr(nn, "secrets") and "OPENAI_API_KEY" in nn.secrets:
        os.environ["OPENAI_API_KEY"] = nn.secrets["OPENAI_API_KEY"]
except Exception:
    pass

BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
    BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Streamlit App Config
nn.set_page_config(
    page_title="LegalDoc.AI - RAG Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
nn.markdown("""
<style>
    /* Styling headers and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7F00FF, #E100FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #8892B0;
        margin-bottom: 2rem;
    }
    
    /* Document and Source Cards */
    .source-card {
        border-radius: 12px;
        padding: 16px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        border-color: rgba(127, 0, 255, 0.4);
        box-shadow: 0 6px 40px rgba(127, 0, 255, 0.1);
    }
    
    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-doc {
        background-color: rgba(127, 0, 255, 0.15);
        color: #C180FF;
        border: 1px solid rgba(127, 0, 255, 0.3);
    }
    
    .badge-page {
        background-color: rgba(0, 200, 83, 0.15);
        color: #00E676;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    
    .badge-score {
        background-color: rgba(255, 145, 0, 0.15);
        color: #FF9100;
        border: 1px solid rgba(255, 145, 0, 0.3);
    }
    
    .snippet-text {
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 10px;
        color: #E2E8F0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check backend health
def check_backend_connection() -> bool:
    try:
        response = requests.get(BACKEND_URL, timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Initialize session states
if "messages" not in nn.session_state:
    nn.session_state.messages = []
if "sources" not in nn.session_state:
    nn.session_state.sources = []
if "selected_sources" not in nn.session_state:
    nn.session_state.selected_sources = []

# Try importing RAG engine for direct local fallback
try:
    from backend.rag_engine import (
        query_rag,
        build_or_update_vector_store,
        DOCUMENTS_DIR,
        VECTOR_STORE_DIR
    )
    import shutil
    DIRECT_RAG_AVAILABLE = True
except ImportError:
    DIRECT_RAG_AVAILABLE = False

# Verify backend health
backend_active = check_backend_connection()
use_direct_mode = not backend_active and DIRECT_RAG_AVAILABLE

if not backend_active and not DIRECT_RAG_AVAILABLE:
    nn.error(f"⚠️ **Connection Error**: Cannot connect to the FastAPI backend at `{BACKEND_URL}`.")
    nn.info("Please start the backend server using the following command in your terminal:\n```bash\nuvicorn backend.main:app --reload --port 8000\n```")
    nn.stop()

# Load documents list
def fetch_documents():
    if use_direct_mode:
        if not os.path.exists(DOCUMENTS_DIR):
            return []
        documents = []
        for filename in os.listdir(DOCUMENTS_DIR):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(DOCUMENTS_DIR, filename)
                try:
                    stat_info = os.stat(file_path)
                    documents.append({
                        "filename": filename,
                        "size_bytes": stat_info.st_size,
                        "path": file_path
                    })
                except Exception:
                    pass
        return documents
    else:
        try:
            res = requests.get(f"{BACKEND_URL}/documents")
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            nn.sidebar.error(f"Error fetching docs: {e}")
        return []

# Sidebar Configuration
nn.sidebar.markdown("<h2 style='text-align: center; color: #C180FF; font-family: Playfair Display;'>⚙️ Configuration</h2>", unsafe_allow_html=True)

if use_direct_mode:
    nn.sidebar.success("⚡ Running in Local Direct Mode")
    nn.sidebar.caption("All parsing, embedding, and QA are running directly within the Streamlit process (no backend service needed).")
else:
    nn.sidebar.success("🔌 Connected to FastAPI Backend")
    nn.sidebar.caption(f"Connected to backend API at: {BACKEND_URL}")

# 1. LLM Settings
nn.sidebar.subheader("LLM Provider")
llm_provider_label = nn.sidebar.selectbox(
    "Choose LLM Provider",
    ["OpenAI GPT", "Local Ollama"],
    index=0
)
llm_provider = "openai" if llm_provider_label == "OpenAI GPT" else "ollama"

# Dynamic fields depending on provider
openai_api_key = None
if llm_provider == "openai":
    openai_api_key = nn.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-..."
    )
    model_name = nn.sidebar.selectbox(
        "Model Name",
        ["gpt-4o-mini", "gpt-4o", "o1-mini"],
        index=0
    )
else:
    model_name = nn.sidebar.text_input(
        "Ollama Model Name",
        value=os.getenv("OLLAMA_MODEL", "llama3"),
        placeholder="e.g. llama3, mistral, phi3"
    )

# Hyperparameters
nn.sidebar.subheader("Hyperparameters")
top_k = nn.sidebar.slider("Retrieve Top-K Chunks", min_value=1, max_value=8, value=4)
temperature = nn.sidebar.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Document Manager
nn.sidebar.markdown("---")
nn.sidebar.subheader("🗂️ Document Manager")

uploaded_docs = fetch_documents()

if uploaded_docs:
    nn.sidebar.caption(f"{len(uploaded_docs)} document(s) indexed")
    for doc in uploaded_docs:
        col_name, col_del = nn.sidebar.columns([4, 1])
        # Format file size
        size_kb = doc['size_bytes'] / 1024
        col_name.markdown(f"📄 **{doc['filename']}**<br><span style='font-size:0.8em;color:#8892B0;'>{size_kb:.1f} KB</span>", unsafe_allow_html=True)
        if col_del.button("🗑️", key=f"del_{doc['filename']}"):
            try:
                if use_direct_mode:
                    file_path = os.path.join(DOCUMENTS_DIR, doc['filename'])
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    build_or_update_vector_store()
                    nn.toast(f"Deleted {doc['filename']}")
                    nn.rerun()
                else:
                    res = requests.delete(f"{BACKEND_URL}/documents/{doc['filename']}")
                    if res.status_code == 200:
                        nn.toast(f"Deleted {doc['filename']}")
                        nn.rerun()
                    else:
                        nn.sidebar.error("Failed to delete.")
            except Exception as e:
                nn.sidebar.error(f"Error: {e}")
                
    nn.sidebar.markdown("<br>", unsafe_allow_html=True)
    if nn.sidebar.button("Wipe Database 🚨", use_container_width=True):
        try:
            if use_direct_mode:
                if os.path.exists(DOCUMENTS_DIR):
                    for file in os.listdir(DOCUMENTS_DIR):
                        file_path = os.path.join(DOCUMENTS_DIR, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                if os.path.exists(VECTOR_STORE_DIR):
                    shutil.rmtree(VECTOR_STORE_DIR)
                    os.makedirs(VECTOR_STORE_DIR)
                nn.toast("Database wiped successfully!")
                nn.rerun()
            else:
                res = requests.post(f"{BACKEND_URL}/clear")
                if res.status_code == 200:
                    nn.toast("Database wiped successfully!")
                    nn.rerun()
        except Exception as e:
            nn.sidebar.error(f"Error clearing: {e}")
else:
    nn.sidebar.info("No documents uploaded yet.")


# Main App Panel
nn.markdown("<h1 class='main-title'>⚖️ LegalDoc.AI</h1>", unsafe_allow_html=True)
nn.markdown("<p class='subtitle'>Source-Grounded Legal Document Assistant with RAG</p>", unsafe_allow_html=True)

# Document Ingestion Section
with nn.expander("📤 Upload Legal Documents", expanded=not uploaded_docs):
    uploaded_files = nn.file_uploader(
        "Upload legal contracts, NDAs, or agreements (PDFs only)",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if nn.button("Process & Index Documents", use_container_width=True):
        if not uploaded_files:
            nn.warning("Please select at least one PDF file first.")
        else:
            with nn.spinner("Extracting, chunking, and embedding PDFs locally using Sentence Transformers..."):
                try:
                    if use_direct_mode:
                        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
                        for f in uploaded_files:
                            file_path = os.path.join(DOCUMENTS_DIR, f.name)
                            with open(file_path, "wb") as buffer:
                                buffer.write(f.getbuffer())
                        build_or_update_vector_store()
                        nn.success(f"Successfully uploaded and indexed {len(uploaded_files)} document(s) locally.")
                        nn.rerun()
                    else:
                        files_payload = []
                        for f in uploaded_files:
                            files_payload.append(("files", (f.name, f.read(), "application/pdf")))
                        res = requests.post(f"{BACKEND_URL}/upload", files=files_payload)
                        if res.status_code == 200:
                            nn.success(res.json().get("message", "Ingested successfully."))
                            nn.rerun()
                        else:
                            nn.error(f"Upload failed: {res.text}")
                except Exception as e:
                    nn.error(f"Error: {e}")

# Layout Columns for Q&A and Source Inspector
col_qa, col_sources = nn.columns([5, 3])

with col_qa:
    nn.subheader("💬 Legal Chatbot")
    
    # Display message history
    for msg in nn.session_state.messages:
        with nn.chat_message(msg["role"]):
            nn.write(msg["content"])
            
    # Input field
    if user_query := nn.chat_input("Ask a question about your legal documents..."):
        # Display user message
        with nn.chat_message("user"):
            nn.write(user_query)
        nn.session_state.messages.append({"role": "user", "content": user_query})
        
        # Prepare API query
        query_data = {
            "query": user_query,
            "llm_provider": llm_provider,
            "model_name": model_name,
            "api_key": openai_api_key if llm_provider == "openai" else None,
            "top_k": top_k,
            "temperature": temperature
        }
        
        # Query API
        with nn.chat_message("assistant"):
            with nn.spinner("Analyzing document context and generating source-grounded answer..."):
                try:
                    if use_direct_mode:
                        result = query_rag(
                            query=user_query,
                            llm_provider=llm_provider,
                            model_name=model_name,
                            api_key=openai_api_key,
                            top_k=top_k,
                            temperature=temperature
                        )
                        answer = result["answer"]
                        sources = result["sources"]
                        
                        # Display answer
                        nn.write(answer)
                        nn.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Store sources in session state for rendering on the right panel
                        nn.session_state.selected_sources = sources
                        nn.rerun()
                    else:
                        res = requests.post(f"{BACKEND_URL}/query", json=query_data)
                        if res.status_code == 200:
                            res_json = res.json()
                            answer = res_json["answer"]
                            sources = res_json["sources"]
                            
                            # Display answer
                            nn.write(answer)
                            nn.session_state.messages.append({"role": "assistant", "content": answer})
                            
                            # Store sources in session state for rendering on the right panel
                            nn.session_state.selected_sources = sources
                            nn.rerun()
                        else:
                            error_detail = res.json().get("detail", res.text)
                            nn.error(f"Error querying backend: {error_detail}")
                except Exception as e:
                    nn.error(f"Error processing query: {e}")

# Sources Inspector (Right side panel)
with col_sources:
    nn.subheader("🔍 Grounding Sources")
    nn.caption("Review the exact PDF document snippets retrieved to ground the assistant's answer.")
    
    sources_to_show = nn.session_state.selected_sources
    
    if sources_to_show:
        for src in sources_to_show:
            # Score percent representation
            score_pct = src['score'] * 100
            
            card_html = f"""
            <div class="source-card">
                <div>
                    <span class="badge badge-doc">📄 {src['filename']}</span>
                    <span class="badge badge-page">Page {src['page']}</span>
                    <span class="badge badge-score">Relevance: {score_pct:.1f}%</span>
                </div>
                <div class="snippet-text">
                    "{src['content']}"
                </div>
            </div>
            """
            nn.markdown(card_html, unsafe_allow_html=True)
    else:
        nn.info("No sources retrieved yet. Submit a query to see matching document snippets here.")
