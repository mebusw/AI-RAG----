##pip install python-dotenv langchain chromadb tiktoken sentence-transformers

import os
from dotenv import load_dotenv
import ollama
import base64
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain, RetrievalQA
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
# logging.basicConfig(level=logging.DEBUG)

# Step 0: 加载环境变量
load_dotenv()
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://ark.cn-beijing.volces.com/api/v3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Step 1.1 从PDF文档提起文本
# 先将文档转换为图像，然后，使用Tesseract识别图像中的文本。Tesseract是HP在1985年开发的主要OCR系统，目前由谷歌维护。
def handle_PDFs():
    import pdf2image #1.17.0
    doc_img = pdf2image.convert_from_path("data/doc_nvidia.pdf", dpi=300)

    import pytesseract #0.3.10
    doc_txt = []
    for page in doc_img:
        text = pytesseract.image_to_string(page)
        doc_txt.append(text)

# Step 1.2: 加载和预处理文本
def load_documents():
    loader = TextLoader("csm_course_outline.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    ## 可选地，使用目录为段落添加标签
    def add_metadata(doc_txt=[]):
        title_map = { 
            "4-12" : "Business", 
            "13-33" : "Risk Factors",
            "34-44" : "Financials",
            "45-46" : "Directors",
            "47-83" : "Data" 
        }
        lst_docs, lst_ids, lst_metadata = [], [], []
        for n, page in enumerate(doc_txt):
            try:
                ## 获取标题
                title = [v for k,v in title_map.items() if n in range(int(k.split("-")[0]), int(k.split("-")[1])+1)][0]
                ## 清理页面
                page = page.replace("Table of Contents","")
                ## 获取段落
                for i,p in enumerate(page.split('\n\n')):
                    if len(p.strip())>5: 
                        lst_docs.append(p.strip())
                        lst_ids.append(str(n)+"_"+str(i))
                        lst_metadata.append({"title":title})
            except:
                continue
    
    ## 可选地，元数据增强可以显著提高文档检索效率。例如，可以使用Phi3将每个段落总结为几个关键词。
    def keyword_generator(p, top=3):
        prompt = "summarize the following paragraph in 3 keywords separated by , : "
        res = ollama.generate(model="phi3", prompt=prompt)["response"]
        return res.replace("\n", " ").strip()

    
    return split_docs

# Step 1.3: 加载和预处理图像图表
# 本函数是调用ollama某个模型来描述图像或图表
def load_images():
    def encode_image(path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')

    img = encode_image("/path/to/image.jpg")
    prompt = "describe the image"
    res = ollama.generate(model="llava", prompt=prompt, images=[img])["response"] ##注意这里的提示词参数是prompt，不是messages
    print(res)

    img = encode_image("/path/to/diagram.jpg")
    prompt = "Describe the image in detail. Be specific about graphs, such as bar plots, line graphs, etc."
    res = ollama.generate(model="llava", prompt=prompt, images=[img])["response"] ##注意这里的提示词参数是prompt，不是messages
    print(res)



# Step 2: 加载或创建向量数据库
def load_or_create_vectorstore():
    embeddings = OpenAIEmbeddings(
                    api_key=OPENAI_API_KEY, 
                    model="ep-20250104171017-p8sfd", 
                    base_url=OPENAI_API_URL, 
                    tiktoken_enabled=False
                )

    if os.path.exists("local_db"):
        print("Loading existing vector database...")
        return Chroma(persist_directory="local_db/", embedding_function=embeddings)
    else:
        print("Creating new vector database...")
        documents = load_documents()
        return Chroma.from_documents(documents, embeddings, persist_directory="local_db/")

# Step 3: 构建代理
def build_agent(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="ep-20250103223903-7kzhd", base_url=OPENAI_API_URL)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain
    
# Step 4: 主函数，运行代理
def main():
    vectorstore = load_or_create_vectorstore()
    agent = build_agent(vectorstore)

    while True:
        print("\nAI Agent is ready. Type your questions (type 'exit' to quit).")
        query = input(">>> You: ")
        if query.lower() == "exit":
            break
        
        result = agent.invoke({"query": query})
        print(">>> AI: ", result["result"])
        print("Sources:", [doc.metadata["source"] for doc in result["source_documents"]])


if __name__ == "__main__":
    main()
