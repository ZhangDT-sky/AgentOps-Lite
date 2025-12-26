from typing import List

from src.agent.models import planner_runnable
from src.agent.state import State


class PlannerNode:
    def run(self, state: State) -> dict:
        if not state.intent:
            raise ValueError("No intent")

        plan_text = self._generate_plan(state.user_query,state.intent)
        plan_steps = self._parse_plan(plan_text)
        state.plan=plan_steps
        return {"plan":plan_steps}


    def _generate_plan(self,user_query,intent:str)->str:
        response = planner_runnable.invoke(
            {"intent": intent, "user_query": user_query},
        )
        return getattr(response, "content", str(response)).strip()

    def _parse_plan(self, plan_text: str) -> List[str]:
        """解析计划文本为步骤列表"""
        # 提取编号步骤：1. xxx  2. xxx
        import re
        steps = []
        # 匹配 "1. 步骤描述" 或 "1) 步骤描述" 格式
        pattern = r'^\d+[.)]\s*(.+)$'
        for line in plan_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                steps.append(match.group(1))
            elif steps:  # 如果已有步骤，可能是续行
                steps[-1] += " " + line

        return steps if steps else [plan_text]  # 如果没有编号，返回整个文本作为一步