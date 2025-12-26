from langchain_core.messages import HumanMessage

from src.agent.models import assistant_runnable
from src.agent.state import State


class IntentRouterNode:
    """意图识别节点：根据用户输入分类意图，写回到 State.intent"""

    def run(self, state: State) -> dict:
        # 将当前用户输入包装成对话消息传给 LLM
        messages = [HumanMessage(content=state.user_query)]
        # prompt 中占位的是 {messages}
        response = assistant_runnable.invoke({"messages": messages})

        # LangChain ChatOpenAI 返回 ChatMessage / AIMessage
        intent_text = getattr(response, "content", str(response)).strip()

        # 更新状态中的意图字段
        state.intent = intent_text

        # LangGraph 节点返回一个对状态的增量更新
        return {"intent": intent_text}
