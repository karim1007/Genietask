from rag import RAGManager

# rag_manager = RAGManager()


# sources = [
#     "https://en.wikipedia.org/wiki/Noel_Price_(footballer)",
#     "report.pdf",
#     "karim.txt"
# ]
# rag_manager.process_sources(sources)



# rag_manager.save_index("index")


rag_manager = RAGManager()
rag_manager.load_index("index")

results = rag_manager.similarity_search("executive summary", k=2)

# Process the results
for chunk, metadata, score in results:
    print(f"Relevance score: {score}")
    print(f"Source: {metadata.get('source', 'Unknown')}")
    print(f"Content: {chunk[:150]}...")  # Show first 150 characters
    print("-" * 50)