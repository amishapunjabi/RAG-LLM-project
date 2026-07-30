from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings("ignore")
pdf_path = "data/aiml_concepts_master_guide.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# print(f"Total Pages: {len(pages)}")
# print(f"Page 1 Content: {pages[0].page_content[:500]}")  # Print first 500 characters of page 1
# print(f"\n Metadata of Page 1: {pages[0].metadata}")

total_characters = sum(len(page.page_content) for page in pages)
# print(f"\nTotal Characters in PDF: {total_characters}")
estimated_tokens = total_characters / 4  # Rough estimate: 1 token ≈ 4 characters
# print(f"Estimated Tokens in PDF: {estimated_tokens}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len,is_separator_regex=False)
chunks = text_splitter.split_documents(pages)
# print(f"\nTotal Chunks Created after splitting: {len(chunks)}")

# print(f"\nContent of First Chunk: {chunks[0].page_content[:500]}")  # Print first 500 characters of first chunk

if len(chunks) > 1:
    print(f" End part of chunk 1: {chunks[0].page_content[-100:]}")  # Print last 500 characters of first chunk
    print(f" start part of Chunk 2: {chunks[1].page_content[:100]}")  # Print first 500 characters of second chunk   
