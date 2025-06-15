import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.embeddings import Embeddings

# Step 0: 加载环境变量
load_dotenv()
API_URL = os.getenv("OPENAI_API_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMB_MODEL = os.getenv("EMB_MODEL")

client = OpenAI(
    api_key = API_KEY,
    base_url = API_URL,
)

# Non-streaming:
def test_non_streaming():
    try:
        print("----- standard request -----")
        completion = client.chat.completions.create(
            model = CHAT_MODEL,  # your model endpoint ID
            messages = [
                {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
                {"role": "user", "content": "常见的十字花科植物有哪些？"},
            ],
        )
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")
test_non_streaming()


# Streaming:
def test_streaming():
    try:
        print("----- streaming request -----")
        stream = client.chat.completions.create(
            model = CHAT_MODEL,  # your model endpoint ID
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
    except Exception as e:
        print(f"An error occurred: {e}")
test_streaming()


# Embedding:
def test_embedding():
    print("----- embedding request -----")
    try:
        completion = client.embeddings.create(
            model=EMB_MODEL,  # your model endpoint ID
            input="hi",
        )
        print(f"Successfully embedded. Vector length: {len(completion.data[0].embedding)}")
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
test_embedding()

# Embedding with Langchain
def test_embedding_with_langchain():
    print("----- embedding request with Langchain-----")
    try:
        embeddings = OpenAIEmbeddings(model=EMB_MODEL, 
                                      api_key=API_KEY, 
                                      base_url=API_URL,
                                      tiktoken_enabled=False)
        test_text = "hi"
        print(f"Attempting to embed: '{test_text}' (Type: {type(test_text)})")
        embedded_vector = embeddings.embed_query(test_text)
        print(f"Successfully embedded. Vector length: {len(embedded_vector)}")
    except Exception as e:
        print(f"An error occurred during embedding: {e}")

test_embedding_with_langchain()

# Embedding with Custom OpenAI-Compatible Class
class CustomOpenAICompatibleEmbeddings(Embeddings):
    """
    Custom Embeddings class to interact directly with an OpenAI-compatible API,
    bypassing Langchain's default model validation.
    """
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Ensure the input is a list of strings, as expected by OpenAI client
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        # Ensure the input is a list containing the single string
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

def test_custom_embedding_with_langchain():
    print("----- Custom embedding request with Langchain-----")
    try:
        embeddings = CustomOpenAICompatibleEmbeddings(
            api_key=API_KEY,
            base_url=API_URL,
            model_name=EMB_MODEL # Use the exact model name your service expects
        )
        test_text = "hi"
        print(f"Attempting to embed: '{test_text}' (Type: {type(test_text)})")
        embedded_vector = embeddings.embed_query(test_text)
        print(f"Successfully embedded. Vector length: {len(embedded_vector)}")
    except Exception as e:
        print(f"An error occurred during custom embedding: {e}")

# Once you've confirmed the direct call works, you can run this:
test_custom_embedding_with_langchain()