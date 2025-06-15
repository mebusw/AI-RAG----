##pip install python-dotenv langchain chromadb tiktoken sentence-transformers

import os
from dotenv import load_dotenv
import ollama
import base64
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from typing import List
# logging.basicConfig(level=logging.DEBUG)

# Step 0: 加载环境变量
load_dotenv()
API_URL = os.getenv("OPENAI_API_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMB_MODEL = os.getenv("EMB_MODEL")

print(f"{API_URL}, {API_KEY}, {CHAT_MODEL}, {EMB_MODEL}")

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
class CustomOpenAICompatibleEmbeddings(Embeddings):
    """
    Custom Embeddings class to interact directly with an OpenAI-compatible API,
    bypassing Langchain's default model validation.
    """
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Ensure the input is a list of strings, as expected by OpenAI client
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        print(f"Embedding doc response: {response}")
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        # Ensure the input is a list containing the single string
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

def load_or_create_vectorstore():
    embeddings = CustomOpenAICompatibleEmbeddings(
                    api_key=API_KEY, 
                    model=EMB_MODEL, 
                    base_url=API_URL, 
                )

    if os.path.exists("local_db"):
        print("Loading existing vector database...")
        return Chroma(persist_directory="local_db/", embedding_function=embeddings)
    else:
        print("Creating new vector database...")
        documents = load_documents()
        return Chroma.from_documents(documents, embeddings, persist_directory="local_db/")

from typing import Generator
from queue import Queue

class StreamingCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)

    def __init__(self):
        self._generator = None
        self.queue = Queue()

    def start_streaming(self) -> Generator[str, None, None]:
        """
        启动一个生成器，用于消费 on_llm_new_token 中的数据。
        """
        if self._generator is not None:
            try:
                while True:
                    token = self.queue.get(block=True) 
                    if token is None:  # 检查是否结束
                        break
                    yield token
            except StopIteration:
                # 生成器已被关闭
                pass

    def set_generator(self, generator: Generator[str, None, None]):
        """
        设置一个生成器实例。
        """
        self._generator = generator

handler = StreamingCallbackHandler()


# Step 3: 构建代理
def build_agent(vectorstore, streaming=False):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, api_key=API_KEY, model=CHAT_MODEL, base_url=API_URL,
                streaming=streaming,
                callbacks=[handler],)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain
    
# Step 4: 主函数，运行代理
def main():

    def _main_loop():
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
    _main_loop()


# Step 4A: Streamlit是构建快速Web应用程序最常用的Python库，因为它通过其流式功能简化了NLP应用程序的开发。首先，定义布局：我的屏幕应有一个侧边栏，用户可以在其中查看聊天历史记录。
import streamlit as st
def buildUI():
    ## 布局
    st.title("Write your questions")
    st.sidebar.title("Chat History")

    app = st.session_state
    if 'messages' not in app:
        app['messages'] = [{"role": "assistant", "content": "I'm ready to retrieve information"}]
    if 'history' not in app:
        app['history'] = []
    if 'full_response' not in app:
        app['full_response'] = ''

    ## 保持消息在聊天中
    for msg in app["messages"]:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

    ## 聊天
    if txt := st.chat_input():
        ### 用户写入
        app["messages"].append({"role": "user", "content": txt})
        st.chat_message("user", avatar="🧑").write(txt)
        app["full_response"] = ""
    
        # ### AI 使用聊天流式响应
        # with st.chat_message("assistant", avatar="🤖"):
        #     ai.respond(app["messages"], use_knowledge=True)
        #     for chunk in handler._generator:
        #         print(f"chunk={chunk}")
        #         if chunk is not None:
        #             st.write(chunk) #FIXME 流式下，st.write没办法不换行
        #             app["full_response"] += str(chunk)

        ### AI 非流式响应
        with st.chat_message("assistant", avatar="🤖"):
            chunk = ai.respond(app["messages"], use_knowledge=True)
            app["full_response"] += chunk["result"]
            st.write(app["full_response"])
            print("Sources:", [doc.metadata["source"] for doc in chunk["source_documents"]])


        ### 显示历史记录
        app["messages"].append({"role": "assistant", "content": app["full_response"]})
        app['history'].append("🧑: " + txt)
        app['history'].append("🤖: " + app["full_response"])
        st.sidebar.markdown("<br/>".join(app['history']) + "<br/><br/>", unsafe_allow_html=True)


class AI:
    def __init__(self):
        self.vectorstore = load_or_create_vectorstore()
        self.agent = build_agent(self.vectorstore, streaming=False)
        # 创建生成器并启动
        generator = handler.start_streaming()
        handler.set_generator(generator)


    def respond(self, lst_messages, use_knowledge=False):
        if use_knowledge:
            prompt = "Give the most accurate answer using your knowledge to user's query.\n'{query}':"
        else:
            prompt = "Give the most accurate answer without external knowledge to user's query.\n'{query}':"

        res = self.agent.invoke({"query": prompt + lst_messages[-1]["content"]})
        return res


if __name__ == "__main__":
    ai = AI()
    buildUI()
    # main()
