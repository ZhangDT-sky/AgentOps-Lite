from src.agent.state import State


class CriticNode:
    """
    Critic Node：执行结果质量控制与流程裁决
    - 评估当前 Agent 执行结果是否可接受
    - 决定流程走向：accept / retry / fail
    """

    def run(self,state: State) -> dict:
        # 检查是否生成了草稿答案
        if not state.draft_answer or not state.draft_answer.strip():
            return self._handle_retry(
                state,
                reason="答案为空"
            )

        # 检查工具调用
        failed_tools = [
            call for call in state.tool_calls
            if not call.success
        ]
        if failed_tools:
            failed_names = [call.name for call in failed_tools]
            return self._handle_retry(
                state,
                reason=f"工具调用失败：{', '.join(failed_names)}"
            )

        # 基础质量检查
        quality_check = self._check_quality(state)
        if not quality_check["passed"]:
            return self._handle_retry(
                state,
                reason=quality_check["critic_reason"]
            )

        # 通过所有检查，接受结果
        state.final_answer = state.draft_answer
        return {
            "critic_decision": "accept",
            "critic_reason": "全部检查通过",
            "final_answer": state.draft_answer
        }




    def _handle_retry(self, state, reason):
        """
        统一处理重试/失败逻辑
        """
        state.retries += 1
        if state.retries > state.max_retries:
            # 超过最大重试次数则判定失败
            failure_reason = (
                f"智能体执行在重试 {state.max_retries} 次后失败。"
                f"最后失败原因：{reason}。"
                f"追踪ID：{state.trace_id}"
            )
            state.final_answer = failure_reason
            return {
                "critic_decision": "fail",  # 保持约定值不变，避免影响上下游逻辑
                "critic_reason": f"超出最大重试次数：{reason}",
                "final_answer": failure_reason
            }
        return {
            "critic_decision": "retry",  # 保持约定值不变
            "critic_reason": reason,
            "final_answer": None
        }

    def _check_quality(self, state: State) -> dict:
        """
        基础质量检查
        """
        draft = state.draft_answer.strip()

        if len(draft) < 10:
            return {
                "passed": False,
                "critic_reason": f"答案过短（{len(draft)} 字符）"
            }
        # 根据意图类型做针对性检查

        if state.intent == "analysis":
            # 分析类意图应包含分析性内容
            analysis_keywords = ["分析", "趋势", "结果", "数据", "计算", "统计"]
            if not any(keyword in draft for keyword in analysis_keywords):
                # 如果计划中有分析步骤，但答案中无分析性内容，判定不通过
                if state.plan and any("分析" in step for step in state.plan):
                    return {
                        "passed": False,
                        "critic_reason": "分析类意图但答案缺少分析性内容"
                    }
        if state.intent == "task":
            # 任务类意图应说明任务执行情况
            task_keywords = ["完成", "已执行", "成功", "调用", "执行"]
            if not any(keyword in draft for keyword in task_keywords):
                # 如果计划中有工具调用步骤，但答案中无执行说明，判定不通过
                if state.plan and any("调用" in step or "工具" in step for step in state.plan):
                    if state.tool_calls:  # 存在工具调用记录
                        return {
                            "passed": False,
                            "critic_reason": "任务类意图但答案缺少执行状态说明"
                        }

        # 如果计划要求检索，但检索文档为空
        if state.plan:
            has_retrieval_step = any(
                "检索" in step or "知识库" in step or "RAG" in step  # RAG为行业通用缩写，保留
                for step in state.plan
            )
            if has_retrieval_step and not state.retrieved_docs:
                return {
                    "passed": False,
                    "critic_reason": "计划要求检索操作但检索文档为空"
                }

        return {"passed": True, "critic_reason": "质量检查通过"}