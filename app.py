import os
import streamlit as st
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from deep_translator import GoogleTranslator
from langdetect import detect
from langchain_google_vertexai import ChatVertexAI
from google.cloud import aiplatform
from dotenv import load_dotenv
import requests
import arxiv

# ------------------------------
# 🔤 Supported Languages
# ------------------------------
SUPPORTED_LANGUAGES = {
    'en': 'English', 
    'fr': 'French', 
    'es': 'Spanish', 
    'de': 'German', 
    'zh-cn': 'Chinese', 
    'ar': 'Arabic'
}

# ------------------------------
# 🔑 Load API Key from .env file
# ------------------------------
load_dotenv()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("google_api_key")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# ------------------------------
# 🚀 Initialize Google Vertex AI
# ------------------------------
aiplatform.init(project="nih-cl-cm500-cingabir-7f5a", location="us-central1")

# ------------------------------
# 🤖 Initialize Vertex AI Chat Models
# ------------------------------
llm_gemini_pro = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0.7)
llm_gemini_flash = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0.7)

# ------------------------------
# 🗄️ Initialize ChromaDB
# ------------------------------
db_path = "./chromadb_store"
try:
    client = chromadb.PersistentClient(path=db_path)
    collection_minilm = client.get_or_create_collection("research_papers_minilm")
    collection_mpnet = client.get_or_create_collection("research_papers_mpnet")
except Exception as e:
    st.error(f"Error loading ChromaDB: {e}")
    st.stop()

# ------------------------------
# 📂 Ensure database has papers
# ------------------------------
all_papers_minilm = collection_minilm.get()
all_papers_mpnet = collection_mpnet.get()
if len(all_papers_minilm['ids']) < 1 and len(all_papers_mpnet['ids']) < 1:
    st.warning("ChromaDB contains no research papers. Please run data_processing.py first!")
    st.stop()

# ------------------------------
# 📘 Load Embedding Models
# ------------------------------
embedding_model_minilm = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model_mpnet = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MPNet-base-v2")

# ------------------------------
# 🌐 Available Research Domains
# ------------------------------
available_domains = ["AI", "Education", "Computer Vision", "Agriculture", "Healthcare"]

# ------------------------------
# 🖥️ Streamlit App UI
# ------------------------------
st.title("Research Assistant with RAG & Google Vertex AI")

# ------------------------------
# 🧩 Sidebar for Model Selection
# ------------------------------
st.sidebar.header("Model Selection")
embedding_choice = st.sidebar.radio("Select an embedding model:", ["MiniLM", "MPNet"])
selected_embedding = embedding_model_minilm if embedding_choice == "MiniLM" else embedding_model_mpnet

llm_choice = st.sidebar.radio("Select an LLM:", ["Gemini 1.5 Pro", "Gemini 1.5 Flash"])
selected_llm = llm_gemini_pro if llm_choice == "Gemini 1.5 Pro" else llm_gemini_flash

# ------------------------------
# 📝 User Input
# ------------------------------
query = st.text_area("Enter your research question in any language:")
selected_domain = st.selectbox("Select a research domain:", available_domains)
recommended_papers = st.checkbox(":page_facing_up: Include research paper recommendations")
# debug_mode = st.checkbox("Enable debug mode")

# ------------------------------
# 🌐 Language Detection & Validation
# ------------------------------
def detect_and_validate_language(query):
    detected_lang = detect(query)
    if detected_lang not in SUPPORTED_LANGUAGES:
        st.warning(f"⚠️ The detected language ('{detected_lang}') is not supported. Defaulting to English.")
        return "en"
    return detected_lang

# ------------------------------
# 🔍 Retrieve Passages from ChromaDB
# ------------------------------
def retrieve_passages_from_rag(query, domain, embedding_model, top_n=5):
    """Retrieves relevant research passages from ChromaDB for the selected domain."""
    collection = collection_minilm if embedding_model == embedding_model_minilm else collection_mpnet
    query_embedding = embedding_model.embed_query(query)
    search_results = collection.query(query_embeddings=[query_embedding], n_results=top_n)
    
    passages = [
        doc for doc, meta in zip(search_results["documents"][0], search_results["metadatas"][0])
        if meta.get("domain", "").lower() == domain.lower()
    ][:top_n]
    
    return passages

# ------------------------------
# 🔎 Fetch Online Research Papers
# ------------------------------
def fetch_research_papers(query):
    paper_titles, paper_urls = [], []
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        for result in client.results(search):
            paper_titles.append(result.title)
            paper_urls.append(result.pdf_url)
    except Exception as e:
        print(f"Error fetching from arXiv: {e}")
    return paper_titles, paper_urls

# ------------------------------
# 🤖 Generate LLM Response
# ------------------------------
def generate_llm_response(query, passages, selected_llm, detected_lang):
    context = "\n".join(passages[:3]) if passages else "No relevant research found."
    prompt = f"""You are an AI assistant using Retrieval-Augmented Generation (RAG).
    You support the following user languages: {', '.join(SUPPORTED_LANGUAGES.values())}.
    
    **Query:** {query}

    **Retrieved Context:** 
    {context}

    **User's original language:** {SUPPORTED_LANGUAGES.get(detected_lang, 'English')}

    **Provide a detailed and factual response in one paragraph. IMPORTANT: Your response should be in the same language as the user's original query ({SUPPORTED_LANGUAGES.get(detected_lang, 'English')}).**
    """
    response = selected_llm.invoke(prompt).content if selected_llm else "No response generated."
    return response

# ------------------------------
# 🚀 Execute Query
# ------------------------------
if st.button("Query"):
    detected_lang = detect_and_validate_language(query)
    translated_query = GoogleTranslator(source="auto", target="en").translate(query)

    # Retrieve passages using the selected embedding model
    passages = retrieve_passages_from_rag(translated_query, selected_domain, selected_embedding)
    response = generate_llm_response(query, passages, selected_llm, detected_lang)
    st.write(response)

    # Fetch recommended research papers
    if recommended_papers:
        titles, urls = fetch_research_papers(translated_query)
        if titles:
            st.write("### 📄 Recommended Research Papers:")
            for title, url in zip(titles, urls):
                st.write(f"📄 **{title}**")
                st.write(f"🔗 [Read Paper]({url})")