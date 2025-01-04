# A demo from https://mp.weixin.qq.com/s/zeMICV-9URcUdBfDb5_Yeg, with Doubao API

import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: 加载环境变量
load_dotenv()
API_URL = os.getenv("DOUBAO_API_URL", "https://ark.cn-beijing.volces.com/api/v3")
API_KEY = os.getenv("DOUBAO_API_KEY")

client = OpenAI(
    api_key = API_KEY,
    base_url = API_URL,
)

# Non-streaming:
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

completion = client.chat.completions.create(
    model = "ep-20250103223903-7kzhd",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": f"Summarize the following table: {table}"},
    ],
)
print("\nSummary : \n", completion.choices[0].message.content)

