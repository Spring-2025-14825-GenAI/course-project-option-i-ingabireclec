# import os
# import arxiv
# import chromadb
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from pypdf import PdfReader

# # Initialize ChromaDB
# db_path = "./chromadb_store"
# client = chromadb.PersistentClient(path=db_path)
# collection = client.get_or_create_collection("research_papers")

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def download_papers(query, num_papers=5, save_folder="papers"):
#     """ Downloads research papers from arXiv. """
#     os.makedirs(save_folder, exist_ok=True)
#     search = arxiv.Search(query=query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)

#     paper_metadata = []
#     for result in search.results():
#         pdf_filename = f"{result.entry_id.split('/')[-1]}.pdf"
#         pdf_path = os.path.join(save_folder, pdf_filename)

#         try:
#             result.download_pdf(dirpath=save_folder, filename=pdf_filename)
#             print(f"✅ Downloaded: {pdf_filename}")

#             paper_metadata.append({
#                 "title": result.title,
#                 "url": result.pdf_url,
#                 "pdf_path": pdf_path
#             })
#         except Exception as e:
#             print(f"❌ Failed to download {result.title}: {e}")

#     return paper_metadata

# def extract_text(paper_metadata):
#     """ Extracts text from PDFs. """
#     paper_texts = []
#     for paper in paper_metadata:
#         reader = PdfReader(paper["pdf_path"])
#         text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
#         paper_texts.append({"text": text, "metadata": paper})

#     return paper_texts

# def embed_papers_for_domains(domains, num_papers=5):
#     """ Downloads, extracts, and embeds research papers for multiple domains. """
#     for domain in domains:
#         print(f"\n🔍 Processing domain: {domain}")
#         paper_metadata = download_papers(query=domain, num_papers=num_papers)
#         paper_texts = extract_text(paper_metadata)

#         for i, paper in enumerate(paper_texts):
#             embedding = embedding_model.embed_query(paper["text"])
#             collection.add(
#                 ids=[f"{domain}_paper_{i}"],
#                 documents=[paper["text"]],
#                 metadatas=[{
#                     "title": paper["metadata"]["title"],
#                     "url": paper["metadata"].get("url", "N/A"),
#                     "domain": domain
#                 }]
#             )

#         print(f"✅ Embedded {len(paper_texts)} papers for domain: {domain}")

# # Define research domains
# domains = ["Artificial intelligence", "cybersecurity" ,"Health Care", "Education", "Climate Change"]

# # Run embedding
# embed_papers_for_domains(domains, num_papers=5)

import os
import arxiv
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize ChromaDB
db_path = "./chromadb_store"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("research_papers")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define research domains with mapping to app domains
domain_mapping = {
    "Artificial intelligence and machine learning": "AI",
    "Healthcare informatics and medical technologies": "Health care",
    "Sustainable agriculture and food systems": "Agriculture",
    "Climate change mitigation and adaptation": "Climate change",
    "Cybersecurity and network defense": "Cyber security"
}

def download_papers(query, num_papers=4, save_folder="papers"):
    """ Downloads research papers from arXiv. """
    os.makedirs(save_folder, exist_ok=True)
    search = arxiv.Search(query=query, max_results=num_papers+2, sort_by=arxiv.SortCriterion.Relevance)

    paper_metadata = []
    for result in search.results():
        if len(paper_metadata) >= num_papers:
            break
            
        pdf_filename = f"{result.entry_id.split('/')[-1]}.pdf"
        pdf_path = os.path.join(save_folder, pdf_filename)

        try:
            result.download_pdf(dirpath=save_folder, filename=pdf_filename)
            print(f"✅ Downloaded: {pdf_filename}")

            paper_metadata.append({
                "title": result.title,
                "url": result.pdf_url,
                "authors": ", ".join([author.name for author in result.authors]),
                "published": str(result.published),
                "pdf_path": pdf_path
            })
        except Exception as e:
            print(f"❌ Failed to download {result.title}: {e}")

    return paper_metadata

def extract_and_chunk_text(paper_metadata):
    """ Extracts text from PDFs and chunks it for better retrieval. """
    paper_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for paper in paper_metadata:
        try:
            reader = PdfReader(paper["pdf_path"])
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            # Try to extract abstract
            abstract_start = text.lower().find("abstract")
            abstract_end = text.lower().find("introduction")
            if abstract_start > 0 and abstract_end > abstract_start:
                abstract = text[abstract_start:abstract_end]
            else:
                # If can't find proper abstract boundaries, take first part
                abstract = text[:1500]
            
            # Chunk the text
            chunks = text_splitter.split_text(text)
            
            # Add each chunk
            for i, chunk in enumerate(chunks):
                paper_chunks.append({
                    "text": chunk,
                    "metadata": {
                        **paper,
                        "chunk_id": i,
                        "abstract": abstract
                    }
                })
        except Exception as e:
            print(f"Error processing {paper['title']}: {e}")
    
    return paper_chunks

def embed_papers_for_domains():
    """ Downloads, extracts, and embeds research papers for multiple domains. """
    total_papers = 0
    
    # Check if collection already has papers
    existing = collection.get()
    if len(existing["ids"]) > 0:
        print(f"⚠️ Collection already contains {len(existing['ids'])} documents.")
        clear = input("Do you want to clear the existing collection? (y/n): ")
        if clear.lower() == "y":
            collection.delete(ids=existing["ids"])
            print("✅ Collection cleared.")
        else:
            print("⚠️ Skipping paper processing. Collection already contains data.")
            return
    
    for search_query, app_domain in domain_mapping.items():
        print(f"\n🔍 Processing domain: {app_domain} (Search query: {search_query})")
        paper_metadata = download_papers(query=search_query, num_papers=4)
        paper_chunks = extract_and_chunk_text(paper_metadata)

        # Track unique papers for this domain
        domain_papers = set()
        
        for i, chunk in enumerate(paper_chunks):
            paper_title = chunk["metadata"]["title"]
            
            # Generate unique ID for each chunk
            chunk_id = f"{app_domain}_{paper_title[:20].replace(' ', '_')}_{chunk['metadata']['chunk_id']}"
            
            # Add to domain papers tracking
            domain_papers.add(paper_title)
            
            try:
                # Create embedding for the chunk
                embedding = embedding_model.embed_query(chunk["text"])
                
                # Add to collection
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[{
                        "title": chunk["metadata"]["title"],
                        "url": chunk["metadata"]["url"],
                        "authors": chunk["metadata"]["authors"],
                        "published": chunk["metadata"]["published"],
                        "domain": app_domain,
                        "chunk_id": chunk["metadata"]["chunk_id"],
                        "abstract": chunk["metadata"]["abstract"][:500]  # Truncate long abstracts
                    }]
                )
            except Exception as e:
                print(f"❌ Error embedding chunk {i} from {paper_title}: {e}")

        print(f"✅ Embedded {len(domain_papers)} papers ({len(paper_chunks)} chunks) for domain: {app_domain}")
        total_papers += len(domain_papers)
    
    # Verify the database contents
    all_docs = collection.get()
    print(f"\n📊 Database Statistics:")
    print(f"Total documents: {len(all_docs['ids'])}")
    
    # Count documents per domain
    domain_counts = {}
    for metadata in all_docs["metadatas"]:
        domain = metadata.get("domain", "Unknown")
        if domain not in domain_counts:
            domain_counts[domain] = 0
        domain_counts[domain] += 1
    
    for domain, count in domain_counts.items():
        print(f"  - {domain}: {count} chunks")
    
    print(f"\n✅ Successfully processed {total_papers} papers across {len(domain_mapping)} domains")

if __name__ == "__main__":
    print("🚀 Starting Research Paper Processing")
    embed_papers_for_domains()
    print("✅ Processing complete!")