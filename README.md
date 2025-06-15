# AI Bot with RAG

It call OpenAI compitable chat/embedding APIs such as Doubao API with LangChain wrapped.

## Setup

1. register LLM and put the API_KEY and base_url of API into `.env` file
2. create a virtual env of Python 3.12 (Can't be higher version, since httpx error)
```bash
pyenv install 3.12
pyenv local 3.12
pyenv virtualenv py312-ai
pyenv activate py312-ai
python -m pip install --upgrade pip setuptools
```
3. install dependences
```bash
pip install -r requirements.txt
```
4. start application
```bash
streamlit run local-embedding-rag.py

```

## 注意
- 免费的LLM可能会限流，导致异常，可以改为充值的LLM
- 更换embbeding模型后，必须删掉原来的向量数据库文件