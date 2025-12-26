from datetime import datetime
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.agent import prompt
from src.agent.prompt import PLANNER_PROMPT

qwen_api_key = os.environ.get("QWEN_API_KEY")
qwen_base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")  # 设默认值

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


assistant_runnable = primary_assistant_prompt | llm

planner_runnable = planner_prompt | llm
