import streamlit as st
# from googletrans import Translator
# from langdetect import detect
# import chromadb
# from langdetect import detect

# import the google vertex AI
# from google.cloud import aiplatform

# supported languages
Supported_languages = {
    'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German', 'zh-cn': 'Chinese', 'ar': 'Arabic'
}

# translator = Translator()
# client = chromadb.PersistentClient(path="./chromadb_store")
# collection = client.get_or_create_collection("research_papers")

# use google vertex AI like openai
# llm = aiplatform.gapic.LanguageModelServiceClient()

st.title("Multilingual Research Paper Assistant with RAG")
query = st.text_area("Enter your query in any language:")

research_domain = st.selectbox("Select a research domain:", ["AI", "Health care", "Finance", "Education", "Agriculture", "Climate change", "Cyber security"])
print(research_domain)
recommended_papers = st.checkbox("Include research paper recommendations")
if recommended_papers:
    st.write("You selected to include research paper recommendations.")

if query:
    # detected_lang = detect(query)

    # if detected_lang not in Supported_languages:
    #     st.write(f"unsupported language: {detected_lang}")

    # else:
    #     translated_query = translator.translate(query, src=detected_lang, dest='en').text
    #     st.write(f"Detected Language: {Supported_languages[detected_lang]}")
    #     st.write(f"Searching for :{translated_query} in {research_domain}")

    #     # Retrieve related research papers using ChromaDB
    #     search_results = collection.query(query_texts=[translated_query], n_results=3)

    #     response = llm.complete(prompt=f"Provide a research based paper answer for:{translated_query} in {research_domain}", max_tokens=200)
    #     translated_response = translator.translate(response, src='en', dest=detected_lang).text
    #     st.success(translated_response)

    #     if recommended_papers:
    #         st.write("Recommended Research Papers:")
    #         for paper in search_results:
    #             st.write(f"Title: {paper['title']}")
    #             st.write(f"Abstract: {paper['abstract']}")
    #             st.write(f"URL: {paper['url']}")
    #             st.write("----")
    st.write("Query received. Processing is currently disabled.")