from src.agent.state import State


class DraftAnswerNode:
    def run(self, state: State):
        try:
            if not state.plan:
                state.draft_answer = (
                    f"无法生成执行计划，直接回答用户问题：{state.user_query}"
                )
                return state
            if state.intent == "qa":
                state.draft_answer = self._generate_qa_answer(state)

            elif state.intent == "task":
                state.draft_answer = self._generate_task_answer(state)

            elif state.intent == "analysis":
                state.draft_answer = self._generate_analysis_answer(state)
            else:
                state.draft_answer = (
                    f"收到用户请求：{state.user_query}\n"
                    f"已生成初步计划：{state.plan}"
                )

        except Exception as e:
            # 任何异常都不要抛出，交给 Critic 决策
            state.draft_answer = None
            state.memory["draft_error"] = str(e)

        return state

    def _generate_qa_answer(self, state) -> str:
        return (
            f"用户问题：{state.user_query}\n\n"
            f"基于当前信息给出的初步回答如下：\n"
            f"- {state.plan[0]}"
        )

    def _generate_task_answer(self, state) -> str:
        steps = "\n".join(
            [f"{idx + 1}. {step}" for idx, step in enumerate(state.plan)]
        )
        return (
            f"针对任务请求，已生成如下执行方案：\n\n"
            f"{steps}\n\n"
            f"（当前为草稿结果，尚未执行具体工具）"
        )

    def _generate_analysis_answer(self, state) -> str:
        steps = "\n".join(
            [f"- {step}" for step in state.plan]
        )
        return (
            f"针对分析请求，生成初步分析思路如下：\n\n"
            f"{steps}\n\n"
            f"后续可基于数据或工具进一步完善分析结论。"
        )