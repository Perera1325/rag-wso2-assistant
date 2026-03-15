from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

db = FAISS.load_local("vectorstore", embeddings)

def retrieve_docs(query):
    docs = db.similarity_search(query, k=3)
    return docs