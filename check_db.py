import chromadb

# Initialize ChromaDB
db_path = "./chromadb_store"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("research_papers")

# Check stored papers
papers = collection.get()

print("🔍 Checking stored papers...")
print(f"Total papers stored: {len(papers['ids'])}")

# Print first few stored papers
for i, meta in enumerate(papers["metadatas"][:20]):  # Show first 5 papers
    print(f"\n📄 Paper {i+1}: {meta['title']}")
    print(f"🔗 URL: {meta['url']}")
