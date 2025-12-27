from enum import Enum
from typing import Optional, List

from pydantic import BaseModel

class ExecutionStatus(str, Enum):
    """执行状态枚举（执行层面，不涉及业务语义）"""
    SUCCESS = "success"  # 成功完成（Graph 正常结束）
    FAILED = "failed"  # 执行失败（异常）
    TERMINATED = "terminated"  # 被终止（超时/超限）
    TIMEOUT = "timeout"  # 超时
    INTERRUPTED = "interrupted"  # 被中断（边界控制触发）

class ExecutionConfig(BaseModel):
    """执行配置（边界控制）"""
    max_steps: int = 20                    # 最大 Graph loop 次数（通过 LangGraph 配置）
    max_execution_time: float = 300.0      # 最大执行时间（秒）
    interrupt_after: Optional[List[str]] = None  # 在哪些节点后中断（用于边界检查）
    fail_fast: bool = False                # 是否快速失败
    enable_trace: bool = True              # 是否启用执行追踪
    return_final_state: bool = True        # 是否在结果中包含完整 State