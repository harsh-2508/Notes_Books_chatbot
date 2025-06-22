!pip install langchain openai faiss-cpu tiktoken PyMuPDF

!pip install sentence-transformers

from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]

import fitz  

doc = fitz.open(filename)
text = ""
for page in doc:
    text += page.get_text()

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def ask_question(query, top_k=1):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0]]
    return "\n---\n".join(results)

while True:
    query = input("\nAsk something about your book (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = ask_question(query)
    print("Answer:", answer)

