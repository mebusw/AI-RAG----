import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: 加载环境变量
load_dotenv()
API_URL = os.getenv("OPENAI_API_URL", "https://ark.cn-beijing.volces.com/api/v3")
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = API_KEY,
    base_url = API_URL,
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model = "ep-20250103223903-7kzhd",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    model = "ep-20250103223903-7kzhd",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
    stream=True
)

for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()