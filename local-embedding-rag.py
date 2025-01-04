##pip install python-dotenv langchain chromadb tiktoken sentence-transformers

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain, RetrievalQA
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 0: 加载环境变量
load_dotenv()
API_URL = os.getenv("DOUBAO_API_URL", "https://ark.cn-beijing.volces.com/api/v3")
API_KEY = os.getenv("DOUBAO_API_KEY")

# Step 1: 加载和预处理文档
def load_documents():
    loader = TextLoader("csm_course_outline.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    print(split_docs)
    return split_docs

# Step 2: 创建嵌入和向量数据库
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings(api_key=API_KEY, model="ep-20250104171017-p8sfd", base_url=API_URL)
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="local_db/")
    vectorstore.persist()
    return vectorstore

# Step 3: 构建代理
def build_agent(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0, api_key=API_KEY, model="ep-20250103223903-7kzhd", base_url=API_URL)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain
    
# Step 4: 主函数，运行代理
def main():
    # 加载或构建向量数据库
    try:
        vectorstore = Chroma(persist_directory="local_db/")
    except Exception:
        documents = load_documents()
        vectorstore = create_vectorstore(documents)
    print(vectorstore)
    agent = build_agent(vectorstore)

    print("AI Agent is ready. Type your questions (type 'exit' to quit).")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        result = agent(query)
        print("AI:", result["result"])
        print("Sources:", [doc.metadata["source"] for doc in result["source_documents"]])


if __name__ == "__main__":
    main()