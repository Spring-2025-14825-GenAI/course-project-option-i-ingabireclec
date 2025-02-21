import os
import arxiv
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize ChromaDB

db_path = "./chromadb_store"
client = chromadb.PersistentClient(path=db_path)

# Create separate collections for different embedding models
collection_minilm = client.get_or_create_collection("research_papers_minilm")
collection_mpnet = client.get_or_create_collection("research_papers_mpnet")

# Load embedding models
embedding_model_minilm = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model_mpnet = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MPNet-base-v2")

# Define research domains with mapping to app domains
research_domains = ["AI", "Education", "Computer Vision", "Agriculture", "Healthcare"]

def download_papers(query, domain, num_papers=4, save_folder="papers"):
    """Downloads research papers from arXiv and assigns them to a specific research domain."""
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
            print(f"Downloaded: {pdf_filename}")

            paper_metadata.append({
                "title": result.title,
                "url": result.pdf_url,
                "authors": ", ".join([author.name for author in result.authors]),
                "published": str(result.published),
                "pdf_path": pdf_path,
                "domain": domain  # Assign the paper to the selected domain
            })
        except Exception as e:
            print(f"Failed to download {result.title}: {e}")

    return paper_metadata

def extract_and_chunk_text(paper_metadata):
    """Extracts text from PDFs, chunks it, and retains domain information for retrieval."""
    
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
                abstract = text[:1500]  # Fallback if no clear abstract section
            
            # Chunk the text
            chunks = text_splitter.split_text(text)
            
            # Assign each chunk its metadata, including domain
            for i, chunk in enumerate(chunks):
                paper_chunks.append({
                    "text": chunk,
                    "metadata": {
                        **paper,  # Inherit all original metadata (title, authors, etc.)
                        "chunk_id": i,
                        "abstract": abstract,
                        "domain": paper["domain"]  # Ensure the domain is retained
                    }
                })
        except Exception as e:
            print(f"Error processing {paper['title']}: {e}")
    
    return paper_chunks

def embed_papers_for_domains():
    """Downloads, processes, and embeds research papers for the predefined domains."""

    total_papers = 0
    existing_minilm = collection_minilm.get()
    existing_mpnet = collection_mpnet.get()

    if len(existing_minilm["ids"]) > 0 and len(existing_mpnet["ids"]) > 0:
        print(f"ChromaDB already contains {len(existing_minilm['ids'])} MiniLM and {len(existing_mpnet['ids'])} MPNet documents. Skipping download.")
        return  

    for domain in research_domains:
        print(f"\n Downloading research papers for domain: {domain}...")

        # Pass the correct domain to the download function
        paper_metadata = download_papers(query=domain, domain=domain, num_papers=4)
        paper_chunks = extract_and_chunk_text(paper_metadata)

        for i, chunk in enumerate(paper_chunks):
            paper_title = chunk["metadata"]["title"]
            chunk_id = f"{domain}_{paper_title[:20].replace(' ', '_')}_{chunk['metadata']['chunk_id']}"

            try:
                # Generate embeddings for both models
                embedding_minilm = embedding_model_minilm.embed_query(chunk["text"])
                embedding_mpnet = embedding_model_mpnet.embed_query(chunk["text"])

                # Add to MiniLM collection
                collection_minilm.add(
                    ids=[chunk_id],
                    embeddings=[embedding_minilm],
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]]
                )

                # Add to MPNet collection
                collection_mpnet.add(
                    ids=[chunk_id],
                    embeddings=[embedding_mpnet],
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]]
                )
                
                print(f"Embedded chunk {i} from {paper_title} under domain: {domain}")

            except Exception as e:
                print(f"Error embedding chunk {i} from {paper_title}: {e}")

        print(f"Embedded {len(paper_chunks)} chunks for domain: {domain}")
        total_papers += len(paper_chunks)

    print(f"\n**Final Database Statistics:** Total MiniLM documents = {len(collection_minilm.get()['ids'])}, Total MPNet documents = {len(collection_mpnet.get()['ids'])}")

if __name__ == "__main__":
    print("Starting Research Paper Processing")
    embed_papers_for_domains()
    print("Processing complete!")

