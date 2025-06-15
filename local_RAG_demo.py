# A demo from https://mp.weixin.qq.com/s/zeMICV-9URcUdBfDb5_Yeg, with Doubao API

import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 0: 加载环境变量
load_dotenv()
API_URL = os.getenv("OPENAI_API_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMB_MODEL = os.getenv("EMB_MODEL")

print(f"{API_URL}, {API_KEY}, {CHAT_MODEL}, {EMB_MODEL}")

client = OpenAI(
    api_key = API_KEY,
    base_url = API_URL,
)

# Step 2.1 预处理文本段落：
def read_text_file(file_name):
    """
    按行读取当前目录下的一个文本文件。

    :param file_name: 文本文件名
    :return: 包含文件内容的列表，每个元素为文件的一行
    """
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)
    
    if not os.path.exists(file_path):
        print(f"文件 {file_name} 不存在于目录 {current_directory}")
        return []
    
    text_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.readlines()
    
    return text_data

file_name = "csm_course_outline.txt"  # 预处理过的文本
doc_txt = read_text_file(file_name)
# for line in doc_txt:
#     print(line.strip())


## 使用目录为段落添加标签
title_map = { 
    "1-10" : "Course title", 
    "11-28" : "Company Introduction", 
    "29-40" : "Course overview",
    "41-48" : "Benefits to participants",
    "49-58" : "Advantages of the course",
    "59-75" : "Key Learning Objectives",
    "76-81" : "Target participants",
    "82-364" : "Agenda", 
    "365-366" : "Attachements and Quotation", 
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

for doc, doc_id, metadata in zip(lst_docs, lst_ids, lst_metadata):
    print(f"Document: {doc}, ID: {doc_id}, Metadata: {metadata}")

# Step 2.2 预处理表格:
table = """Table:
Year Ended
Jan 28,2024 Jan 29,2023 Change($ in millions, except per share data)
Revenue $ 60,922 $ 26,974 Up 126%
Gross margin 72.7 % 56.9 % Up 15.8 pts
Operating expenses $ 11,329 $ 11,132 Up 2%
Operating income $ 32,972 $ 4,224 Up 681%
Net income $ 29,76日 $4,368 Up 581%
Net income per diluted share $ 11.93 $ 1.74 Up 586%"""
print("Table: \n", table)

# completion = client.chat.completions.create(
#     model = "ep-20250103223903-7kzhd",  # your model endpoint ID
#     messages = [
#         {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
#         {"role": "user", "content": f"Summarize the following table: {table}"},
#     ],
# )
# print("\nSummary : \n", completion.choices[0].message.content)


# Step 3 向量数据库

import chromadb  # 0.5.0
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# 创建一个持久化的数据库实例
db = chromadb.PersistentClient()

# 创建一个默认嵌入函数实例
embedding_function = OpenAIEmbeddingFunction(
    api_key=API_KEY,  # 替换为您的 OpenAI API 密钥
    api_base=API_URL,
    model_name=EMB_MODEL  # 默认模型，可以根据需要更改
)

# 获取或创建一个名为 "nvidia" 的集合
collection_name = "nvidia"
collection = db.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)


# 将文档、ID和元数据添加到集合中
collection.add(
    documents=lst_docs,
    ids=lst_ids,
    metadatas=lst_metadata,
    images=None,
    embeddings=None
)

print(f"查看集合中的一个样本：")
print(collection.peek(1))


print(f"接下来，尝试查询一些信息：")

query = "课程优势是什么?"
res_db = collection.query(query_texts=[query])["documents"]
print(res_db)
context = ' '.join(res_db[0][0:100]).replace("\n", " ")
print(context)
