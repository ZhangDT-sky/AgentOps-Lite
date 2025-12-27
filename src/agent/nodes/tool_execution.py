
import re
import json
from typing import List, Dict, Any, Optional
from src.agent.state import State, ToolCall
from src.tools.tool_registry import tool_registry

class ToolExecution:
    """
    工具执行节点：Agent 级工具调用执行器
    """
    def __init__(self):
        self.tool_keywords = ["调用", "使用", "执行", "工具"]
        self.tool_name_patterns = [
            r'调用(\w+)工具',
            r'使用(\w+)工具',
            r'执行(\w+)工具',
            r'调用(\w+)',
        ]
        # 中文工具名到英文工具名的映射
        self.tool_name_mapping = {
            "天气": "get_weather",
            "天气查询": "get_weather",
            "查询天气": "get_weather",
            "邮件": "send_email",
            "发送邮件": "send_email",
            "邮件发送": "send_email",
            "数据库": "query_database",
            "查询数据库": "query_database",
            "数据库查询": "query_database",
            "计算": "calculator",
            "计算器": "calculator",
        }

    def run(self,state: State):
        """工具调用"""
        if not state.plan:
            return {"tool_calls":[]}
        # 提取需要执行的工具调用
        tool_calls_to_execute = self._extract_tool_calls(state)

        if not tool_calls_to_execute:
            # 没有工具调用，返回空列表
            return {"tool_calls": []}

        # 执行工具调用
        executed_calls = []
        for tool_call_info in tool_calls_to_execute:
            tool_call = self._execute_tool(tool_call_info, state)
            executed_calls.append(tool_call)

        # 合并到现有 tool_calls（避免重复执行）
        existing_tool_names = {call.name for call in state.tool_calls}
        new_calls = [
            call for call in executed_calls
            if call.name not in existing_tool_names
        ]

        updated_tool_calls = state.tool_calls + new_calls

        return {"tool_calls": updated_tool_calls}

    def _extract_tool_calls(self, state):

        tool_calls = []
        for step in state.plan:
            if not any(keyword in step for keyword in self.tool_keywords):
                continue

            # 尝试提取工具名称
            tool_name = self._extract_tool_name(step)
            if not tool_name:
                continue

            # 提取参数（从步骤文本中解析）
            params = self._extract_params(step, tool_name)

            # 更新工具调用记录
            tool_calls.append({
                "tool_name": tool_name,
                "params": params,
                "step": step
            })

        return tool_calls

    def _extract_tool_name(self, step):
        """从步骤文本中提取工具名称"""
        # 正则
        for pattern in self.tool_name_patterns:
            match = re.search(pattern,step, re.IGNORECASE)
            if match:
                name = match.group(1).lower()
                # 转换为标准工具名称
                return self._normalize_tool_name(name)

        for chinese_name,english_name in self.tool_name_mapping.items():
            if chinese_name in step:
                return english_name

        return None


    def _normalize_tool_name(self, name: str) -> str:
        """标准化工具名称"""
        # 中文工具名映射
        if name in self.tool_name_mapping:
            return self.tool_name_mapping[name]

        # 包含匹配
        for key, value in self.tool_name_mapping.items():
            if key in name or name in key:
                return value

        # 转换为下划线命名
        return name.replace(" ", "_").replace("-", "_")

    def _extract_params(self, step, tool_name):
        """
          从步骤文本中提取工具调用参数

          支持格式：
          - "调用天气查询工具，城市=北京"
          - "调用邮件发送工具，收件人=张三，内容=xxx"
          - "调用工具，参数：{'city': '北京'}"
          """
        params = {}

        # key=value or key:value
        kv_pattern = r'(\w+)[=：]\s*([^，,]+)'

        matches = re.findall(kv_pattern, step)
        for key, value in matches:
            normal_key = self._normalize_param_name(key, tool_name)
            params[normal_key] = value.strip().strip('"').strip("'")

    def _normalize_param_name(self, param_name: str, tool_name: str) -> str:
        """标准化参数名称（中英文映射）"""
        param_mapping = {
            "城市": "city",
            "城市名": "city",
            "日期": "date",
            "收件人": "to",
            "主题": "subject",
            "内容": "content",
            "抄送": "cc",
            "表名": "table",
            "表": "table",
            "条件": "conditions",
            "限制": "limit",
            "操作": "operation",
            "运算": "operation",
            "第一个数": "a",
            "第二个数": "b",
        }

        if param_name in param_mapping:
            return param_mapping[param_name]

        # 如果已经是英文，直接返回
        return param_name

    def _execute_tool(
            self,
            tool_call_info: Dict[str, Any],
            state: State
    ) -> ToolCall:
        """
        执行单个工具调用（使用 ToolRegistry.execute）

        Returns:
            ToolCall: 工具调用结果
        """
        tool_name = tool_call_info["tool_name"]
        params = tool_call_info["params"]
        step_text = tool_call_info["step_text"]

        tool_call = ToolCall(
            name=tool_name,
            input_params=params,
            success=False,
            output=None,
            error=None
        )

        try:
            # 使用 ToolRegistry.execute 执行工具
            result = tool_registry.execute(tool_name, state, params)

            # 记录成功结果
            tool_call.success = True
            tool_call.output = str(result) if result is not None else "执行成功"

        except ValueError as e:
            # 参数校验失败或工具未注册
            error_msg = str(e)
            tool_call.success = False
            tool_call.error = error_msg

            # 记录到 memory（用于调试）
            state.memory[f"tool_{tool_name}_error"] = error_msg
            state.memory["tool_execution_errors"] = state.memory.get(
                "tool_execution_errors", []
            ) + [f"{tool_name}: {error_msg}"]

        except Exception as e:
            # 工具执行异常
            error_msg = str(e)
            tool_call.success = False
            tool_call.error = error_msg

            # 记录到 memory
            state.memory[f"tool_{tool_name}_error"] = error_msg
            state.memory["tool_execution_errors"] = state.memory.get(
                "tool_execution_errors", []
            ) + [f"{tool_name}: {error_msg}"]

        return tool_call
