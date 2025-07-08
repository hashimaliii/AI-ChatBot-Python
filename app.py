from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

# Setup embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Check if index exists, if not create one
if os.path.exists("faiss_index/index.faiss"):
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    dummy_doc = [Document(page_content="placeholder text")]
    db = FAISS.from_documents(dummy_doc, embeddings)
    db.save_local("faiss_index")

# Text splitter
splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)

# LLM (Ollama)
llm = Ollama(model="llama2")

# QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), chain_type="stuff")

# Function to add new data in real-time
def add_new_data(text):
    docs = splitter.create_documents([text])
    db.add_documents(docs)
    db.save_local("faiss_index")
    print(f"✅ Added {len(docs)} document chunks.")

# Add real data
add_new_data("LangChain is an open-source Python framework for building applications powered by large language models.")

# Query system
response = qa.invoke({"query": "What is LangChain?"})
print("Answer:", response)
