from langchain_core.messages import HumanMessage

from src.agent.models import assistant_runnable
from src.agent.state import State


class IntentRouterNode:
    """意图识别节点：根据用户输入分类意图，写回到 State.intent"""

    ALLOWED_INTENTS = {
        "qa","task","analysis"
    }

    def run(self, state: State) -> dict:
        try:
            # 将当前用户输入包装成对话消息传给 LLM
            messages = [HumanMessage(content=state.user_query)]
            response = assistant_runnable.invoke({"messages": messages})
            # LangChain ChatOpenAI 返回 ChatMessage / AIMessage
            intent_text = getattr(response, "content", str(response)).strip()

        except Exception as e:
            state.memory["intent_error"] = str(e)
            return {"intent":"qa"}

        # 兜底裁剪
        intent_text = intent_text.split()[0]
        intent_text = intent_text.replace('"', '').replace("'", "")
        if intent_text not in self.ALLOWED_INTENTS:
            intent_text = "qa"

        # 状态增量更新
        return {"intent": intent_text}
