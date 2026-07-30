"""Embeddings is a way of converting text/words into long list of number called as vectors. To store this vectors/numbers we use chromadb.
For math part, converting words to numbers we are using Hugging face embeddings. Hugging face embeddings is completely free and runs locally on our computer. No API Keys no extra cost. Chromadb live in our local system RAM. This  means your Database is deleted. You have to reprocess your pdf every single time you want to ask a question. To avoid this we use persistent directory. It tells chroma db to save the data in hard drive.
"""

import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
import os
warnings.filterwarnings("ignore")

pdf_path = "data/aiml_concepts_master_guide.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

print(f"\nCreated {len(chunks)} chunks from {len(pages)} pages.\n")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# #sample of embedding text to numbers
# sample_text = "Hello World"
# print(f"Embedding dimensions: {len(sample_text)}")
# print(f"First 5 values: {len(sample_text[:5])}")
# print(f"Embedding vector: {embeddings.embed_query(sample_text)}")

persistent_directory = "./chroma_db"
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persistent_directory,collection_name="pdf_collection")

# print(f"\nVector store created with {len(vector_store)} vectors.\n")?
print(f"Stored {len(chunks)} chunks in the vector store/chromadb.\n")
print(f"Data saved to: {os.path.abspath(persistent_directory)}\n")

query = "What is AI?"
results = vector_store.similarity_search(query, k=3)

for doc in results:
    print(f"Page Number: {doc.metadata['page']}")
    print(f"Content: {doc.page_content[:200]}\n") 
