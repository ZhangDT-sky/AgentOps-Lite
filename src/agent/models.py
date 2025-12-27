from datetime import datetime
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.agent import prompt
from src.agent.prompt import PLANNER_PROMPT

# LLM 配置
qwen_api_key = os.environ.get("QWEN_API_KEY")
qwen_base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")  # 设默认值

# Embedding 模型配置
embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
embedding_base_url = os.environ.get("EMBEDDING_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1")

llm = ChatOpenAI(
    api_key=qwen_api_key,
    model="qwen-plus",
    temperature=0,  # 控制输出的随机性
    streaming=True,  # 关闭流式输出以避免异步问题
    base_url=qwen_base_url
)

llm_turbo = ChatOpenAI(
    api_key=qwen_api_key,
    model="qwen-turbo-latest",
    temperature=0,  # 控制输出的随机性
    streaming=True,  # 关闭流式输出以避免异步问题
    base_url=qwen_base_url
)

# Embedding 模型实例
embedding = OpenAIEmbeddings(
    api_key=embedding_api_key,
    model="text-embedding-v4",
    base_url=embedding_base_url,
    chunk_size=100,  # 设置 chunk_size 以确保批量处理
)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt.PRIMARY_ASSISTANT_PROMPT
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt.PLANNER_PROMPT,
        ),
        ("placeholder", "{messages}"),
    ]
)

retrieval_decision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt.RETRIEVAL_DECISION
        ),
        ("placeholder", "{messages}"),
    ]
)

assistant_runnable = primary_assistant_prompt | llm

planner_runnable = planner_prompt | llm

retrieval_decision_runnable = retrieval_decision_prompt | llm_turbo

