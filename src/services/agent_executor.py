"""
AgentExecutor: AgentOps-Lite 的执行控制中枢
核心职责：
1. State 生命周期管理（初始化、传递、返回）
2. Graph 驱动与执行（调用 LangGraph API）
3. 执行边界控制（通过 LangGraph 配置，而非事后检查）
4. 统一异常治理（捕获、分类、记录）
5. 执行结果封装（从 State 提取，不假设业务字段）
设计原则：
- Executor 是"调度器"，不包含业务逻辑
- Executor 不直接读取业务语义字段（如 critic_decision）
- 边界控制通过 LangGraph 配置实现，而非事后检查
- State 是唯一事实源，Trace 只记录执行层面信息
- Result 从 State 提取，但不假设字段存在
- 所有执行过程通过 LoggingService 记录
"""
import uuid
import time
import traceback
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel
from src.agent.state import State
from src.agent.graph import graph
from src.services.logging_service import (
    get_logging_service,
    EventType
)

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

@dataclass
class ExecutionTrace:
    """
    执行追踪记录
    职责边界：
    - Trace: 记录"如何执行"（步数、耗时、节点顺序、异常）
    - State: 记录"执行了什么"（业务数据）
   - LoggingService: 记录"过程中的离散事实"
    """
    execution_id: str  # 执行唯一标识符，用于关联所有执行相关日志和事件
    request_id: Optional[str] = None  # 外部请求 ID（如 HTTP 请求 ID），用于跨系统追踪
    user_id: Optional[str] = None  # 用户 ID，用于用户级别的执行追踪和统计
    start_time: datetime = field(default_factory=datetime.now)  # 执行开始时间戳
    end_time: Optional[datetime] = None  # 执行结束时间戳（执行完成或终止时设置）
    duration: float = 0.0  # 执行总耗时（秒），由 end_time - start_time 计算得出
    status: ExecutionStatus = ExecutionStatus.SUCCESS  # 执行状态（success/failed/terminated/timeout），执行层面状态，不涉及业务语义
    termination_reason: Optional[str] = None  # 终止原因描述（如超时、超过最大步数等），仅在 status 为 terminated 时有效
    steps_count: int = 0  # Graph 执行步数（loop 次数），用于统计和性能分析
    node_execution_order: List[str] = field(default_factory=list)  # 节点执行顺序列表，记录 Graph 中节点的执行顺序，用于调试和回放
    errors: List[Dict[str, Any]] = field(default_factory=list)  # 执行过程中的异常记录列表，每个元素包含异常类型、消息、堆栈等信息

class ExecutionResult(BaseModel):
    """
    执行结果（结构化返回）
    
    职责：提供对外接口，屏蔽 Agent 内部复杂性，从 State 中提取信息但不假设字段存在
    """
    execution_id: str  # 执行唯一标识符，与 ExecutionTrace 中的 execution_id 一致
    status: ExecutionStatus  # 执行状态（success/failed/terminated/timeout），执行层面状态
    termination_reason: Optional[str] = None  # 终止原因描述，仅在 status 为 terminated 时有效
    steps_count: int = 0  # Graph 执行步数，从 ExecutionTrace 中提取
    execution_time: float = 0.0  # 执行总耗时（秒），从 ExecutionTrace 中提取
    answer: Optional[str] = None  # 最终答案文本，从 state.final_answer 安全提取（如果存在）
    intent: Optional[str] = None  # 识别的用户意图（qa/task/analysis），从 state.intent 安全提取
    plan: Optional[List[str]] = None  # 生成的执行计划步骤列表，从 state.plan 安全提取
    used_tools: List[str] = field(default_factory=list)  # 使用的工具名称列表，从 state.tool_calls 中提取工具名称
    triggered_rag: bool = False  # 是否触发了 RAG 检索，通过检查 state.retrieved_docs 是否存在且非空判断
    retrieved_docs_count: int = 0  # 检索到的文档数量，从 state.retrieved_docs 的长度提取
    retries_count: int = 0  # 重试次数，从 state.retries 安全提取
    error_message: Optional[str] = None  # 错误消息（如果执行失败），从异常对象中提取
    error_type: Optional[str] = None  # 错误类型（异常类名），如 TimeoutError、ValueError 等
    final_state: Optional[Dict[str, Any]] = None  # 完整 State 的字典快照（可选），用于调试和深度分析，仅在 enable_trace=True 时包含


class AgentExecutor:
    """
    Agent 执行器：连接 Agent 内核与真实系统环境的执行与治理层

    核心原则：
    1. Executor 不包含业务逻辑
    2. Executor 不直接读取业务语义字段
    3. 边界控制通过 LangGraph 配置实现
    4. State 是唯一事实源
    5. 所有执行过程通过 LoggingService 记录
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.graph = graph
        self.logging_service = get_logging_service() if self.config.enable_logging else None

    def execute(
            self,
            user_query: str,
            request_id: Optional[str] = None,
            user_id: Optional[str] = None,
            initial_state: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """同步执行 Agent（阻塞式）"""
        execution_id = str(uuid.uuid4())
        trace = ExecutionTrace(
            execution_id=execution_id,
            request_id=request_id,
            user_id=user_id
        )

        start_time = time.time()

        # 记录执行开始
        if self.logging_service:
            self.logging_service.log_event(
                execution_id=execution_id,
                event_type=EventType.EXECUTION_STARTED,
                source="executor",
                request_id=request_id,
                user_id=user_id,
                payload={"user_query": user_query}
            )

        try:
            # 1. 初始化 State
            state = self._initialize_state(user_query, initial_state)

            # 2. 构建 LangGraph 配置
            graph_config = self._build_graph_config(execution_id, trace)

            # 3. 执行 Graph（带超时控制）
            final_state = self._execute_with_timeout(
                lambda: self.graph.invoke(
                    {"user_query": state.user_query},
                    config=graph_config
                ),
                trace
            )

            # 4. 更新追踪信息
            trace.end_time = datetime.now()
            trace.duration = time.time() - start_time
            trace.status = ExecutionStatus.SUCCESS

            # 5. 记录执行完成
            if self.logging_service:
                self.logging_service.log_event(
                    execution_id=execution_id,
                    event_type=EventType.EXECUTION_COMPLETED,
                    source="executor",
                    payload={"duration": trace.duration, "steps_count": trace.steps_count}
                )
                # 记录最终状态快照
                self.logging_service.log_state_snapshot(
                    execution_id=execution_id,
                    state=final_state,
                    source="executor"
                )

            # 6. 构建执行结果
            result = self._build_result(
                execution_id=execution_id,
                state=final_state,
                trace=trace,
                execution_time=time.time() - start_time
            )

            return result

        except TimeoutError as e:
            return self._handle_timeout(
                execution_id=execution_id,
                trace=trace,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return self._handle_execution_error(
                execution_id=execution_id,
                error=e,
                trace=trace,
                execution_time=time.time() - start_time
            )

    async def execute_async(
            self,
            user_query: str,
            request_id: Optional[str] = None,
            user_id: Optional[str] = None,
            initial_state: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """异步执行 Agent"""
        execution_id = str(uuid.uuid4())
        trace = ExecutionTrace(
            execution_id=execution_id,
            request_id=request_id,
            user_id=user_id
        )

        start_time = time.time()

        # 记录执行开始
        if self.logging_service:
            self.logging_service.log_event(
                execution_id=execution_id,
                event_type=EventType.EXECUTION_STARTED,
                source="executor",
                request_id=request_id,
                user_id=user_id,
                payload={"user_query": user_query}
            )

        try:
            state = self._initialize_state(user_query, initial_state)
            graph_config = self._build_graph_config(execution_id, trace)

            final_state = await self._execute_async_with_timeout(
                self.graph.ainvoke(
                    {"user_query": state.user_query},
                    config=graph_config
                ),
                trace
            )

            trace.end_time = datetime.now()
            trace.duration = time.time() - start_time
            trace.status = ExecutionStatus.SUCCESS

            if self.logging_service:
                self.logging_service.log_event(
                    execution_id=execution_id,
                    event_type=EventType.EXECUTION_COMPLETED,
                    source="executor",
                    payload={"duration": trace.duration}
                )
                self.logging_service.log_state_snapshot(
                    execution_id=execution_id,
                    state=final_state,
                    source="executor"
                )

            result = self._build_result(
                execution_id=execution_id,
                state=final_state,
                trace=trace,
                execution_time=time.time() - start_time
            )

            return result

        except TimeoutError as e:
            return self._handle_timeout(
                execution_id=execution_id,
                trace=trace,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return self._handle_execution_error(
                execution_id=execution_id,
                error=e,
                trace=trace,
                execution_time=time.time() - start_time
            )

    async def stream(
            self,
            user_query: str,
            request_id: Optional[str] = None,
            user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式执行 Agent（异步生成器）"""
        execution_id = str(uuid.uuid4())
        state = self._initialize_state(user_query)

        graph_config = {
            "configurable": {
                "execution_id": execution_id,
                "recursion_limit": self.config.max_steps
            }
        }

        # 记录执行开始
        if self.logging_service:
            self.logging_service.log_event(
                execution_id=execution_id,
                event_type=EventType.EXECUTION_STARTED,
                source="executor",
                request_id=request_id,
                user_id=user_id,
                payload={"user_query": user_query}
            )

        try:
            async for state_snapshot in self.graph.astream(
                    {"user_query": user_query},
                    config=graph_config
            ):
                if state_snapshot:
                    latest_node = list(state_snapshot.keys())[-1]
                    node_output = state_snapshot[latest_node]

                    # 记录节点执行
                    if self.logging_service:
                        self.logging_service.log_event(
                            execution_id=execution_id,
                            event_type=EventType.NODE_EXECUTION_COMPLETED,
                            source="graph",
                            node_name=latest_node,
                            payload={"output": node_output.model_dump() if hasattr(node_output, 'model_dump') else str(
                                node_output)}
                        )

                    yield {
                        "type": "node_execution",
                        "execution_id": execution_id,
                        "node": latest_node,
                        "state_snapshot": node_output.model_dump() if hasattr(node_output,
                                                                              'model_dump') else node_output
                    }

            final_state = await self.graph.ainvoke(
                {"user_query": user_query},
                config=graph_config
            )

            if self.logging_service:
                self.logging_service.log_event(
                    execution_id=execution_id,
                    event_type=EventType.EXECUTION_COMPLETED,
                    source="executor"
                )

            yield {
                "type": "final",
                "execution_id": execution_id,
                "result": self._extract_answer_safe(final_state),
                "status": ExecutionStatus.SUCCESS.value
            }

        except Exception as e:
            if self.logging_service:
                self.logging_service.log_exception(
                    execution_id=execution_id,
                    exception=e,
                    source="executor"
                )

            yield {
                "type": "error",
                "execution_id": execution_id,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def _initialize_state(
            self,
            user_query: str,
            initial_state: Optional[Dict[str, Any]] = None
    ) -> State:
        """初始化 State"""
        if initial_state:
            return State(**initial_state)
        else:
            return State(
                user_query=user_query,
                max_retries=self.config.max_steps
            )

    def _build_graph_config(
            self,
            execution_id: str,
            trace: ExecutionTrace
    ) -> Dict[str, Any]:
        """构建 LangGraph 配置（真正的边界控制）"""
        config = {
            "configurable": {
                "execution_id": execution_id,
                "recursion_limit": self.config.max_steps
            }
        }

        if self.config.interrupt_after:
            config["configurable"]["interrupt_after"] = self.config.interrupt_after

        return config

    def _execute_with_timeout(
            self,
            func,
            trace: ExecutionTrace
    ) -> State:
        """带超时控制的执行"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"执行超时: {self.config.max_execution_time}秒")

        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.max_execution_time))

        try:
            result = func()
            return result
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

    async def _execute_async_with_timeout(
            self,
            coro,
            trace: ExecutionTrace
    ) -> State:
        """异步带超时控制的执行"""
        import asyncio

        try:
            return await asyncio.wait_for(
                coro,
                timeout=self.config.max_execution_time
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"执行超时: {self.config.max_execution_time}秒")

    def _build_result(
            self,
            execution_id: str,
            state: State,
            trace: ExecutionTrace,
            execution_time: float
    ) -> ExecutionResult:
        """构建执行结果（从 State 提取，不假设字段存在）"""
        answer = getattr(state, 'final_answer', None)
        intent = getattr(state, 'intent', None)
        plan = getattr(state, 'plan', None)
        retries = getattr(state, 'retries', 0)

        used_tools = []
        if hasattr(state, 'tool_calls') and state.tool_calls:
            used_tools = [call.name for call in state.tool_calls]

        triggered_rag = False
        retrieved_docs_count = 0
        if hasattr(state, 'retrieved_docs') and state.retrieved_docs:
            triggered_rag = True
            retrieved_docs_count = len(state.retrieved_docs)

        status = ExecutionStatus.SUCCESS
        if trace.termination_reason:
            status = ExecutionStatus.TERMINATED

        return ExecutionResult(
            execution_id=execution_id,
            status=status,
            termination_reason=trace.termination_reason,
            steps_count=trace.steps_count,
            execution_time=execution_time,
            answer=answer,
            intent=intent,
            plan=plan,
            used_tools=used_tools,
            triggered_rag=triggered_rag,
            retrieved_docs_count=retrieved_docs_count,
            retries_count=retries,
            final_state=state.model_dump() if self.config.return_final_state else None
        )

    def _extract_answer_safe(self, state: State) -> Optional[str]:
        """安全提取答案（不假设字段存在）"""
        return getattr(state, 'final_answer', None)

    def _handle_timeout(
            self,
            execution_id: str,
            trace: ExecutionTrace,
            execution_time: float
    ) -> ExecutionResult:
        """处理超时"""
        trace.end_time = datetime.now()
        trace.duration = execution_time
        trace.status = ExecutionStatus.TIMEOUT
        trace.termination_reason = f"执行超时: {self.config.max_execution_time}秒"

        if self.logging_service:
            self.logging_service.log_event(
                execution_id=execution_id,
                event_type=EventType.TERMINATION_TIMEOUT,
                source="executor",
                payload={"max_execution_time": self.config.max_execution_time}
            )

        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.TIMEOUT,
            termination_reason=trace.termination_reason,
            execution_time=execution_time,
            error_message=trace.termination_reason,
            error_type="TimeoutError"
        )

    def _handle_execution_error(
            self,
            execution_id: str,
            error: Exception,
            trace: ExecutionTrace,
            execution_time: float
    ) -> ExecutionResult:
        """处理执行错误"""
        trace.end_time = datetime.now()
        trace.duration = execution_time
        trace.status = ExecutionStatus.FAILED
        trace.errors.append({
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        })

        if self.logging_service:
            self.logging_service.log_exception(
                execution_id=execution_id,
                exception=error,
                source="executor"
            )

        status = ExecutionStatus.TIMEOUT if isinstance(error, TimeoutError) else ExecutionStatus.FAILED

        return ExecutionResult(
            execution_id=execution_id,
            status=status,
            termination_reason=str(error),
            execution_time=execution_time,
            error_message=str(error),
            error_type=type(error).__name__
        )


_default_executor: Optional[AgentExecutor] = None


def get_executor(config: Optional[ExecutionConfig] = None) -> AgentExecutor:
    """获取 AgentExecutor 实例（单例模式）"""
    global _default_executor

    if _default_executor is None or config is not None:
        _default_executor = AgentExecutor(config=config)

    return _default_executor