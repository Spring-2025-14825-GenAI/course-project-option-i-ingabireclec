# # import os
# # import streamlit as st
# # import chromadb
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from deep_translator import GoogleTranslator
# # from langdetect import detect
# # from langchain_google_vertexai import ChatVertexAI
# # from google.cloud import aiplatform
# # from dotenv import load_dotenv

# # # Load API key from .env file
# # dotenv_path = "D:/CMU/Year2/Spring2025/genAI/Assignment/hw-2-llm-application-development-ingabireclec/.env"
# # load_dotenv(dotenv_path)

# # # Get API Key and Credentials from environment
# # GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# # # Initialize Google Vertex AI
# # aiplatform.init(project="nih-cl-cm500-cingabir-7f5a", location="us-central1")

# # # Initialize Vertex AI Chat Model (Gemini)
# # llm = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0.7)

# # # Initialize ChromaDB
# # db_path = "./chromadb_store"
# # client = chromadb.PersistentClient(path=db_path)
# # collection = client.get_or_create_collection("research_papers")
# # # Debugging: Inspect ChromaDB contents
# # all_papers = collection.get()
# # st.write("### ChromaDB Contents (Debugging):")
# # st.write(all_papers)

# # if not all_papers['ids']:
# #     st.error("ChromaDB is empty!  Please populate it with data.")
# # else:
# #     st.success("ChromaDB has data. Check metadata for 'title' and 'url'.")

# # # Load embedding model
# # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # # Supported languages
# # SUPPORTED_LANGUAGES = {
# #     'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German', 'zh-cn': 'Chinese', 'ar': 'Arabic'
# # }

# # # Streamlit UI
# # st.title("Multilingual Research Paper Assistant with Google Vertex AI")

# # query = st.text_area("Enter your query in any language:")
# # research_domain = st.selectbox(
# #     "Select a research domain:", 
# #     ["AI", "Health care", "Agriculture", "Climate change", "Cyber security"]
# # )
# # recommended_papers = st.checkbox("Include research paper recommendations")

# # def retrieve_papers(query, domain, top_n=3):
# #     """ Retrieves relevant research papers from ChromaDB based on query & domain. """
# #     query_embedding = embedding_model.embed_query(query)
# #     search_results = collection.query(query_embeddings=[query_embedding], n_results=top_n)

# #     if not search_results["documents"]:  # If no results found
# #         return [], []

# #     paper_titles = []
# #     paper_urls = []

# #     for sublist, meta_list in zip(search_results["documents"], search_results["metadatas"]):
# #         for meta in meta_list:
# #             if meta.get("domain") == domain:
# #                 paper_titles.append(meta.get("title", "Unknown Title"))  # Default to "Unknown Title" if missing
# #                 paper_urls.append(meta.get("url", "#"))  # Default to "#" if URL is missing

# #     return paper_titles, paper_urls


# # def generate_vertex_response(query, retrieved_titles):
# #     """ Generates a response using Google Vertex AI (Gemini) with retrieved research context. """
    
# #     # Combine retrieved knowledge (titles only, no paper translation)
# #     context = "\n".join(retrieved_titles)[:1000]  # Limit to 1000 characters
# #     prompt = f"Based on the following research paper titles, provide a detailed answer to: {query}\n\n{context}\n\nAnswer:"
    
# #     response = llm.invoke(prompt)

# #     return response.content if response else "❌ No response generated."

# # if query:
# #     # Detect and translate query
# #     detected_lang = detect(query)

# #     if detected_lang not in SUPPORTED_LANGUAGES:
# #         st.write(f"❌ Unsupported language detected: {detected_lang}")
# #     else:
# #         translated_query = GoogleTranslator(source="auto", target="en").translate(query)
# #         st.write(f"📝 Detected Language: {SUPPORTED_LANGUAGES[detected_lang]}")
# #         st.write(f"🔍 Searching in {research_domain}...")

# #         # Retrieve research papers (ONLY title & URL)
# #         titles, urls = retrieve_papers(translated_query, research_domain)

# #         if recommended_papers:
# #             st.write("### 📚 Recommended Research Papers:")
# #             for title, url in zip(titles, urls):
# #                 st.write(f"📄 **{title}**")
# #                 st.write(f"🔗 [Read Paper]({url})")

# #         # Generate response using retrieved knowledge and Google Vertex AI (Gemini)
# #         vertex_response = generate_vertex_response(translated_query, titles)

# #         if vertex_response:
# #             # Translate response back to match the user's query language
# #             final_response = GoogleTranslator(source="en", target=detected_lang).translate(vertex_response)
# #             st.success(final_response)
# #         else:
# #             st.error("⚠️ Google Vertex AI did not return a response.")

# import os
# import streamlit as st
# import chromadb
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from deep_translator import GoogleTranslator
# from langdetect import detect
# from langchain_google_vertexai import ChatVertexAI
# from google.cloud import aiplatform
# from dotenv import load_dotenv

# # Load API key from .env file
# dotenv_path = "D:/CMU/Year2/Spring2025/genAI/Assignment/hw-2-llm-application-development-ingabireclec/.env"
# load_dotenv(dotenv_path)

# # Get API Key and Credentials from environment
# GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# # Initialize Google Vertex AI
# aiplatform.init(project="nih-cl-cm500-cingabir-7f5a", location="us-central1")

# # Initialize Vertex AI Chat Model (Gemini)
# llm = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0.7)

# # Initialize ChromaDB
# db_path = "./chromadb_store"
# client = chromadb.PersistentClient(path=db_path)
# collection = client.get_or_create_collection("research_papers")

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# st.title("Multilingual Research Paper Assistant with Google Vertex AI")

# query = st.text_area("Enter your query in any language:")
# research_domain = st.selectbox(
#     "Select a research domain:", 
#     ["AI", "Biology", "Economics", "Health Care"]
# )
# recommended_papers = st.checkbox("Include research paper recommendations")

# def retrieve_papers(query, domain, top_n=3):
#     """Retrieves relevant research papers from ChromaDB based on query & domain."""
#     query_embedding = embedding_model.embed_query(query)
#     search_results = collection.query(query_embeddings=[query_embedding], n_results=top_n)

#     paper_titles, paper_urls, passages = [], [], []

#     for sublist, meta_list in zip(search_results["documents"], search_results["metadatas"]):
#         for doc, meta in zip(sublist, meta_list):
#             if meta.get("domain") == domain:
#                 paper_titles.append(meta.get("title", "Unknown Title"))
#                 paper_urls.append(meta.get("url", "#"))
#                 passages.append(doc)

#     return paper_titles, paper_urls, passages

# def generate_vertex_response(query, retrieved_passages):
#     """Generates a response using Google Vertex AI with retrieved research context."""
#     context = "\n".join(retrieved_passages)[:2000]
#     prompt = f"Based on the following research, provide a detailed answer to: {query}\n\n{context}\n\nAnswer:"
#     response = llm.invoke(prompt)
#     return response.content if response else "❌ No response generated."

# if query:
#     detected_lang = detect(query)
#     translated_query = GoogleTranslator(source="auto", target="en").translate(query)

#     titles, urls, passages = retrieve_papers(translated_query, research_domain)

#     if recommended_papers:
#         st.write("### 📚 Recommended Research Papers:")
#         for title, url in zip(titles, urls):
#             st.write(f"📄 **{title}**")
#             st.write(f"🔗 [Read Paper]({url})")

#     vertex_response = generate_vertex_response(translated_query, passages)
#     final_response = GoogleTranslator(source="en", target=detected_lang).translate(vertex_response)
#     st.success(final_response)

#vvvvv333333333333333
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

# Load API key from .env file
load_dotenv()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("google_api_key")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Initialize Google Vertex AI
aiplatform.init(project="nih-cl-cm500-cingabir-7f5a", location="us-central1")

# Initialize Vertex AI Chat Model (Gemini) and an alternative LLM
llm_gemini = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0.7)
llm_alternative = ChatVertexAI(model_name="gemini-1.5-pro", temperature=0.7)

# Initialize ChromaDB
db_path = "./chromadb_store"
try:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("research_papers")
except Exception as e:
    st.error(f"Error loading ChromaDB: {e}")
    st.stop()

# Ensure database contains at least 20 papers
all_papers = collection.get()
if len(all_papers['ids']) < 20:
    st.warning("ChromaDB contains less than 20 research papers. Consider adding more data.")

# Load two embedding models for comparison
embedding_model_1 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model_2 = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

st.title("Multilingual Research Paper Assistant with Google Vertex AI & FlowiseAI")

query = st.text_area("Enter your query in any language:")
research_domain = st.selectbox("Select a research domain:", ["AI", "Biology", "Economics", "Health Care"])
recommended_papers = st.checkbox("Include research paper recommendations")

embedding_choice = st.radio("Choose an embedding model:", ["MiniLM", "MPNet"])
llm_choice = st.radio("Choose an LLM:", ["Gemini 1.5 Pro", "Gemini 1.5 Alternative"])

def retrieve_papers(query, domain, embedding_model, top_n=3):
    """Retrieves relevant research papers from ChromaDB based on query & domain."""
    query_embedding = embedding_model.embed_query(query)
    search_results = collection.query(query_embeddings=[query_embedding], n_results=top_n)

    paper_titles, paper_urls, passages = [], [], []
    for sublist, meta_list in zip(search_results["documents"], search_results["metadatas"]):
        for doc, meta in zip(sublist, meta_list):
            if meta.get("domain") == domain:
                paper_titles.append(meta.get("title", "Unknown Title"))
                paper_urls.append(meta.get("url", "#"))
                passages.append(doc)
    return paper_titles, paper_urls, passages

def generate_vertex_response(query, retrieved_passages, llm_model):
    """Generates a response using the selected LLM."""
    context = "\n".join(retrieved_passages)[:2000]
    prompt = f"Based on the following research, provide a detailed answer to: {query}\n\n{context}\n\nAnswer:"
    response = llm_model.invoke(prompt)
    return response.content if response else "❌ No response generated."

if st.button("Submit Query"):
    detected_lang = detect(query)
    translated_query = GoogleTranslator(source="auto", target="en").translate(query)
    selected_embedding = embedding_model_1 if embedding_choice == "MiniLM" else embedding_model_2
    selected_llm = llm_gemini if llm_choice == "Gemini 1.5 Pro" else llm_alternative

    titles, urls, passages = retrieve_papers(translated_query, research_domain, selected_embedding)

    if recommended_papers:
        st.write("### 📚 Recommended Research Papers:")
        for title, url in zip(titles, urls):
            st.write(f"📄 **{title}**")
            st.write(f"🔗 [Read Paper]({url})")

    vertex_response = generate_vertex_response(translated_query, passages, selected_llm)
    final_response = GoogleTranslator(source="en", target=detected_lang).translate(vertex_response)
    st.success(final_response)

