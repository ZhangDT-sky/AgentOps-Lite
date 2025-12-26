from typing import Optional, List
from pydantic import BaseModel, Field
import uuid

class ToolCall(BaseModel):
    name: str                     # 工具名称
    input_params: dict            # 调用参数
    output: Optional[str] = None  # 工具输出
    success: bool = True          # 调用是否成功
    error: Optional[str] = None   # 异常信息

class State(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # 用户输入
    user_query: str
    # 意图与计划
    intent: Optional[str] = None         # IntentRouter 输出
    plan: Optional[List[str]] = None     # Planner 输出计划步骤
    # RAG / 检索结果
    retrieved_docs: Optional[List[str]] = None
    # 工具调用记录
    tool_calls: List[ToolCall] = []
    # 草稿答案
    draft_answer: Optional[str] = None
    # 重试控制
    retries: int = 0
    max_retries: int = 2
    # 最终答案
    final_answer: Optional[str] = None

    # 可扩展字段（如日志、上下文缓存等）
    memory: Optional[dict] = Field(default_factory=dict)
    context: Optional[List[str]] = Field(default_factory=list)

    # Critic 决策信息（用于路由）
    critic_decision: Optional[str] = None  # "accept" | "retry" | "fail"
    critic_reason: Optional[str] = None    # 决策原因

    class Config:
        arbitrary_types_allowed = True