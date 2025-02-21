This documentation provides an analysis of the Streamlit-based research assistant application. The system uses Retrieval-Augmented Generation (RAG) with ChromaDB/Milvus as the vector database, integrates Language Models (LLMs) for response generation, and is developed using FlowiseAI with Langchain. The application compares embedding models (MiniLM vs MPNet) and LLMs (Gemini Pro and Gemini Plus)
### 1. Team Introduction
Clemence Ingabire (cingabir)
Marie Cynthia Abijuru Kamikazi (mabijuru)
### 2. Limitation and assumptions
Limitations
- We used just 20 papers for our vector store which limited the amount of external knowledge our LLMs prvided, unless a user ask exactly what is in domain Knowledge database the model will always give the response from whatever data it is trained on
- FlowiseAI visual design translates efficiently to production code
Assumptions
- RAG will always outperform just LLM response, we found out that this will depend on how good your prompts are.
- FlowiseAI visual design will translates almost the same in production code
### 3.Test and observation
#### Text Splitter Testing
We tried modifying our text splitter chunk sizes from 300 tokens  to 1200 tokens. With narrow splits, responses contained more precise contextual relevant from the text but LLM often missed broader context. We were surprised that introduced more irrelevant information. We found out the optimal balance was around 800 tokens.
#### Language Testing
We tried querying in German about "Architekturen neuronaler Netzwerke" and compared results to English queries. It looked like German responses showed lower relevance despite having similar papers in our corpus.This documentation provides an analysis of the Streamlit-based research assistant application. The system uses Retrieval-Augmented Generation (RAG) with ChromaDB/Milvus as the vector database, integrates Language Models (LLMs) for response generation, and is developed using FlowiseAI with Langchain. The application compares embedding models (MiniLM vs MPNet) and LLMs (Gemini Pro and Gemini Plus)
### 1. Team Introduction
Clemence Ingabire (cingabir)
Marie Cynthia Abijuru Kamikazi (mabijuru)
### 2. Limitation and assumptions
Limitations
- We used just 20 papers for our vector store which limited the amount of external knowledge our LLMs prvided, unless a user ask exactly what is in domain Knowledge database the model will always give the response from whatever data it is trained on
- FlowiseAI visual design translates efficiently to production code
Assumptions
- RAG will always outperform just LLM response, we found out that this will depend on how good your prompts are.
- FlowiseAI visual design will translates almost the same in production code
### 3.Test and observation
#### Text Splitter Testing
We tried modifying our text splitter chunk sizes from 300 tokens  to 1200 tokens. With narrow splits, responses contained more precise contextual relevant from the text but LLM often missed broader context. We were surprised that introduced more irrelevant information. We found out the optimal balance was around 800 tokens.
#### Language Testing
We tried querying in German about "Architekturen neuronaler Netzwerke" and compared results to English queries. It looked like German responses showed lower relevance despite having similar papers in our corpus.

#### Video Link: https://cmuafricaexchange.slack.com/archives/D08E9CDH7A7/p1740113675342279
