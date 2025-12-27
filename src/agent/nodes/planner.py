import re
from typing import List

from src.agent.models import planner_runnable
from src.agent.state import State


class PlannerNode:
    MAX_STEPS = 8
    MIN_STEP_LENGTH = 5

    def run(self, state: State) -> dict:
        if not state.intent:
            state.memory["planner_error"] = "missing intent"
            return {"plan": [f"直接回答用户问题：{state.user_query}"]}
        try:
            plan_text = self._generate_plan(state.user_query,state.intent)
            plan_steps = self._parse_plan(plan_text)
        except Exception as e:
            state.memory["planner_error"] = str(e)
            return {"plan":[state.user_query]}

        plan_steps = self._sanitize_plan(plan_steps, state)
        return {"plan":plan_steps}


    def _generate_plan(self,user_query,intent:str)->str:
        response = planner_runnable.invoke(
            {"intent": intent, "user_query": user_query},
        )
        return getattr(response, "content", str(response)).strip()

    def _parse_plan(self, plan_text: str) -> List[str]:
        """
        支持：
        - 1. xxx
        - 1) xxx
        - Step 1: xxx
        - 无编号文本（fallback）
        """
        steps: List[str] = []

        patterns = [
            r'^\d+[.)]\s*(.+)$',
            r'^step\s*\d+[:：]?\s*(.+)$',
            r'^[-•]\s*(.+)$',
        ]

        # 按行拆分计划文本，逐行解析
        for line in plan_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 标记当前行是否匹配到结构化步骤
            matched = False
            # 遍历正则表达式，按优先级匹配
            for pattern in patterns:
                # re.IGNORECASE：忽略大小写（匹配 Step/step/STEP）
                m = re.match(pattern, line, re.IGNORECASE)
                # 如果匹配成功
                if m:
                    # 提取捕获组的内容（步骤文本），加入步骤列表
                    steps.append(m.group(1))
                    # 标记为已匹配，跳出正则循环（避免重复匹配）
                    matched = True
                    break

            # 如果当前行未匹配到结构化步骤，且步骤列表非空（已有解析的步骤）
            if not matched and steps:
                # 将当前行内容追加到最后一个步骤的末尾（处理跨行步骤）
                steps[-1] += " " + line

        return steps


    def _sanitize_plan(self, steps: List[str], state: State) -> List[str]:
        cleaned = []

        for step in steps:
            step = step.strip()
            if len(step) < self.MIN_STEP_LENGTH:
                continue
            cleaned.append(step)

        # 回退到初始问题
        if not cleaned:
            state.memory["planner_warning"] = "步骤过少。回退"
            return [state.user_query]
        # 步骤约束
        if len(cleaned) > self.MAX_STEPS:
            cleaned = cleaned[:self.MAX_STEPS]
            state.memory["planner_warning"] = "步骤过多。截断"

        return cleaned
