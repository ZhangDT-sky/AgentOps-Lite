"""
工具系统模块

导入此模块会自动注册所有示例工具
"""
from src.tools.tool_registry import tool_registry

# 导入示例工具（触发注册）
from src.tools import example_tool  # noqa: F401

__all__ = ["tool_registry"]

