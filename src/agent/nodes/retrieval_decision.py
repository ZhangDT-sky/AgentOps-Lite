from langchain_core.messages import HumanMessage

from src.agent.models import retrieval_decision_runnable
from src.agent.state import State


class RetrievalDecisionNode:
    """
    判断是否需要触发 RAG 检索
    """
    def run(self, state: State) -> dict:
        try:
            user_message = (
                f"用户查询：{state.user_query}\n"
                f"意图：{state.intent}\n"
                f"计划：{state.plan}\n\n"
                f"请判断是否需要进行知识库检索（RAG）。\n"
                f"只回答 true 或 false。"
            )

            messages = [HumanMessage(content=user_message)]
            response = retrieval_decision_runnable.invoke({"messages": messages})

            raw_result = getattr(response, "content", str(response)).strip().lower()

            need_retrieval = self._parse_bool(raw_result)

        except Exception as e:
            # 失败时采用保守策略：仍然检索
            state.memory["retrieval_decision_error"] = str(e)
            need_retrieval = True

        state.need_retrieval = need_retrieval
        return {"need_retrieval": need_retrieval}

    def _parse_bool(self, text: str) -> bool:
        """
        解析 LLM 返回的布尔语义
        """
        if text in {"true", "yes", "是", "需要"}:
            return True
        if text in {"false", "no", "否", "不需要"}:
            return False

        # 不确定时，保守处理
        return True
