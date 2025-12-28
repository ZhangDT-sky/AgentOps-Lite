"""
LoggingService: AgentOps-Lite 执行侧日志与追踪的统一出口层

核心定位：
- 执行侧日志与追踪的统一出口层（不是通用 logging 封装）
- 为 AgentExecutor、Graph、Tool、Node 提供一致、可结构化、可下沉的观测通道
- 不参与执行决策，不影响 State，不做错误恢复，只忠实记录"发生了什么"

设计原则：
1. 结构化事件（Event）为基本单位，而非字符串日志
2. execution_id 为一等公民
3. 语义化事件类型（非通用 info/warning/error）
4. State 快照/差量处理，不直接持有 State
5. 异常完整记录
6. 非阻塞写入（异步、批量、缓冲、降级）
7. 存储抽象（adapter/sink 模式）
8. 失败不影响主流程（自动降级）
"""
import json
import traceback
import asyncio
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from queue import Queue
from threading import Thread
import logging

from pydantic import BaseModel


class EventType(str, Enum):
    """
    语义化事件类型（稳定、可预测，用于分析、统计、告警与回放）

    分层：
    - 执行生命周期事件
    - Graph 驱动事件
    - Node 执行事件
    - Tool 调用事件
    - 状态变更事件
    - 异常事件
    - 终止事件
    """
    # 执行生命周期事件
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_TERMINATED = "execution.terminated"

    # Graph 驱动事件
    GRAPH_LOOP_STARTED = "graph.loop_started"
    GRAPH_LOOP_COMPLETED = "graph.loop_completed"
    GRAPH_ROUTE_DECISION = "graph.route_decision"

    # Node 执行事件
    NODE_EXECUTION_STARTED = "node.execution_started"
    NODE_EXECUTION_COMPLETED = "node.execution_completed"
    NODE_EXECUTION_FAILED = "node.execution_failed"

    # Tool 调用事件
    TOOL_CALL_STARTED = "tool.call_started"
    TOOL_CALL_COMPLETED = "tool.call_completed"
    TOOL_CALL_FAILED = "tool.call_failed"

    # 状态变更事件
    STATE_SNAPSHOT = "state.snapshot"  # 完整快照
    STATE_DELTA = "state.delta"  # 变更摘要

    # 异常事件
    EXCEPTION_RAISED = "exception.raised"
    EXCEPTION_HANDLED = "exception.handled"

    # 终止事件
    TERMINATION_TIMEOUT = "termination.timeout"
    TERMINATION_MAX_STEPS = "termination.max_steps"
    TERMINATION_MAX_RETRIES = "termination.max_retries"
    TERMINATION_MAX_TOOL_CALLS = "termination.max_tool_calls"


@dataclass
class LogEvent:
    """
    结构化事件模型（稳定、可序列化、可索引）

    核心字段（必须）：
    - execution_id: 执行 ID（一等公民）
    - event_type: 事件类型（语义化）
    - timestamp: 时间戳
    - source: 事件来源（executor/graph/node/tool）
    - payload: 事件载荷（结构化数据）

    扩展字段（可选）：
    - request_id: 请求 ID
    - user_id: 用户 ID
    - node_name: 节点名称（如果是节点事件）
    - tool_name: 工具名称（如果是工具事件）
    - error_info: 错误信息（如果是异常事件）
    """
    execution_id: str
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"  # executor/graph/node/tool
    payload: Dict[str, Any] = field(default_factory=dict)

    # 扩展字段
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    node_name: Optional[str] = None
    tool_name: Optional[str] = None
    error_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（可序列化）"""
        data = asdict(self)
        # 处理 datetime
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        # 处理 EventType
        if isinstance(data.get('event_type'), EventType):
            data['event_type'] = data['event_type'].value
        return data

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class LogSink:
    """
    日志存储适配器接口（抽象）

    职责：
    - 接收 LogEvent 并写入存储
    - 处理写入失败（不抛出异常）
    - 支持同步/异步写入
    """

    def write(self, event: LogEvent) -> bool:
        """
        写入事件（同步）

        Args:
            event: 日志事件

        Returns:
            bool: 是否成功（失败不抛异常）
        """
        try:
            self._write_impl(event)
            return True
        except Exception as e:
            # 降级：记录到 stderr，但不影响主流程
            logging.error(f"LogSink.write failed: {e}", exc_info=True)
            return False

    async def write_async(self, event: LogEvent) -> bool:
        """
        写入事件（异步）

        Args:
            event: 日志事件

        Returns:
            bool: 是否成功
        """
        try:
            await self._write_async_impl(event)
            return True
        except Exception as e:
            logging.error(f"LogSink.write_async failed: {e}", exc_info=True)
            return False

    def _write_impl(self, event: LogEvent):
        """子类实现：同步写入逻辑"""
        raise NotImplementedError

    async def _write_async_impl(self, event: LogEvent):
        """子类实现：异步写入逻辑"""
        raise NotImplementedError

    def flush(self):
        """刷新缓冲区（同步）"""
        pass

    async def flush_async(self):
        """刷新缓冲区（异步）"""
        pass


class ConsoleLogSink(LogSink):
    """控制台输出适配器（用于开发/调试）"""

    def _write_impl(self, event: LogEvent):
        print(f"[{event.timestamp.isoformat()}] {event.event_type.value} | {event.source} | {event.execution_id}")
        if event.payload:
            print(f"  Payload: {json.dumps(event.payload, ensure_ascii=False, indent=2)}")

    async def _write_async_impl(self, event: LogEvent):
        self._write_impl(event)


class FileLogSink(LogSink):
    """文件输出适配器（JSONL 格式）"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = None

    def _write_impl(self, event: LogEvent):
        if self._file is None:
            self._file = open(self.filepath, 'a', encoding='utf-8')
        self._file.write(event.to_json() + '\n')
        self._file.flush()

    async def _write_async_impl(self, event: LogEvent):
        # 异步写入文件（使用线程池）
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_impl, event)

    def flush(self):
        if self._file:
            self._file.flush()

    def __del__(self):
        if self._file:
            self._file.close()


class LoggingService:
    """
    日志服务：执行侧日志与追踪的统一出口层

    核心能力：
    1. 结构化事件记录
    2. execution_id 关联
    3. 非阻塞写入（异步、批量、缓冲）
    4. 存储抽象（多 sink 支持）
    5. 失败降级（不影响主流程）
    """

    def __init__(
            self,
            sinks: Optional[List[LogSink]] = None,
            enable_async: bool = True,
            buffer_size: int = 100,
            flush_interval: float = 1.0
    ):
        """
        初始化日志服务

        Args:
            sinks: 存储适配器列表
            enable_async: 是否启用异步写入
            buffer_size: 缓冲区大小
            flush_interval: 刷新间隔（秒）
        """
        self.sinks = sinks or [ConsoleLogSink()]
        self.enable_async = enable_async
        self.buffer_size = buffer_size

        # 事件缓冲区
        self._event_queue: Queue = Queue(maxsize=buffer_size * 2)
        self._worker_thread: Optional[Thread] = None
        self._running = False

        if enable_async:
            self._start_worker()

    def _start_worker(self):
        """启动后台工作线程"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._running = True
            self._worker_thread = Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self):
        """后台工作循环（批量写入）"""
        batch = []
        last_flush = time.time()

        while self._running:
            try:
                # 从队列获取事件（非阻塞）
                try:
                    event = self._event_queue.get(timeout=0.1)
                    batch.append(event)
                except:
                    pass

                # 批量写入或定时刷新
                current_time = time.time()
                should_flush = (
                        len(batch) >= self.buffer_size or
                        (batch and current_time - last_flush >= self.flush_interval)
                )

                if should_flush:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = current_time

            except Exception as e:
                # 降级：记录错误但不中断
                logging.error(f"LoggingService worker error: {e}", exc_info=True)
                time.sleep(0.1)

    def _flush_batch(self, batch: List[LogEvent]):
        """批量刷新事件"""
        for event in batch:
            for sink in self.sinks:
                sink.write(event)

    def log_event(
            self,
            execution_id: str,
            event_type: EventType,
            source: str,
            payload: Optional[Dict[str, Any]] = None,
            request_id: Optional[str] = None,
            user_id: Optional[str] = None,
            node_name: Optional[str] = None,
            tool_name: Optional[str] = None,
            error_info: Optional[Dict[str, Any]] = None
    ):
        """
        记录事件（同步接口，内部可能异步处理）

        Args:
            execution_id: 执行 ID（必须）
            event_type: 事件类型
            source: 事件来源
            payload: 事件载荷
            request_id: 请求 ID
            user_id: 用户 ID
            node_name: 节点名称
            tool_name: 工具名称
            error_info: 错误信息
        """
        # 构建事件
        event = LogEvent(
            execution_id=execution_id,
            event_type=event_type,
            source=source,
            payload=payload or {},
            request_id=request_id,
            user_id=user_id,
            node_name=node_name,
            tool_name=tool_name,
            error_info=error_info
        )

        # 写入（异步或同步）
        if self.enable_async:
            try:
                self._event_queue.put_nowait(event)
            except:
                # 降级：队列满时同步写入
                self._write_sync(event)
        else:
            self._write_sync(event)

    def _write_sync(self, event: LogEvent):
        """同步写入（降级策略）"""
        for sink in self.sinks:
            sink.write(event)

    def log_state_snapshot(
            self,
            execution_id: str,
            state: Any,
            source: str = "executor"
    ):
        """
        记录 State 完整快照

        Args:
            execution_id: 执行 ID
            state: State 对象（不直接持有，只序列化）
            source: 事件来源
        """
        try:
            # 尝试序列化 State
            if hasattr(state, 'model_dump'):
                state_dict = state.model_dump()
            elif hasattr(state, 'dict'):
                state_dict = state.dict()
            elif isinstance(state, dict):
                state_dict = state
            else:
                state_dict = {"_serialization_error": "无法序列化 State"}

            self.log_event(
                execution_id=execution_id,
                event_type=EventType.STATE_SNAPSHOT,
                source=source,
                payload={"state": state_dict}
            )
        except Exception as e:
            # 降级：记录序列化失败，但不影响主流程
            self.log_event(
                execution_id=execution_id,
                event_type=EventType.EXCEPTION_RAISED,
                source="logging_service",
                error_info={
                    "type": "StateSerializationError",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )

    def log_state_delta(
            self,
            execution_id: str,
            delta: Dict[str, Any],
            source: str = "executor"
    ):
        """
        记录 State 变更摘要

        Args:
            execution_id: 执行 ID
            delta: 状态变更（字典）
            source: 事件来源
        """
        self.log_event(
            execution_id=execution_id,
            event_type=EventType.STATE_DELTA,
            source=source,
            payload={"delta": delta}
        )

    def log_exception(
            self,
            execution_id: str,
            exception: Exception,
            source: str,
            context: Optional[Dict[str, Any]] = None
    ):
        """
        记录异常（完整上下文）

        Args:
            execution_id: 执行 ID
            exception: 异常对象
            source: 事件来源
            context: 额外上下文
        """
        error_info = {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }

        self.log_event(
            execution_id=execution_id,
            event_type=EventType.EXCEPTION_RAISED,
            source=source,
            error_info=error_info
        )

    def flush(self):
        """刷新缓冲区（同步）"""
        # 处理队列中剩余事件
        batch = []
        while not self._event_queue.empty():
            try:
                batch.append(self._event_queue.get_nowait())
            except:
                break

        if batch:
            self._flush_batch(batch)

        # 刷新所有 sink
        for sink in self.sinks:
            sink.flush()

    async def flush_async(self):
        """刷新缓冲区（异步）"""
        # 处理队列中剩余事件
        batch = []
        while not self._event_queue.empty():
            try:
                batch.append(self._event_queue.get_nowait())
            except:
                break

        if batch:
            for event in batch:
                for sink in self.sinks:
                    await sink.write_async(event)

        # 刷新所有 sink
        for sink in self.sinks:
            await sink.flush_async()

    def shutdown(self):
        """关闭服务（刷新缓冲区）"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        self.flush()


# 全局日志服务实例（单例模式）
_default_logging_service: Optional[LoggingService] = None


def get_logging_service() -> LoggingService:
    """
    获取日志服务实例（单例模式）
    """
    global _default_logging_service

    if _default_logging_service is None:
        _default_logging_service = LoggingService()

    return _default_logging_service


def configure_logging_service(
        sinks: Optional[List[LogSink]] = None,
        enable_async: bool = True,
        buffer_size: int = 100,
        flush_interval: float = 1.0
) -> LoggingService:
    """
    配置日志服务（全局）

    Args:
        sinks: 存储适配器列表
        enable_async: 是否启用异步写入
        buffer_size: 缓冲区大小
        flush_interval: 刷新间隔

    Returns:
        LoggingService: 配置后的日志服务实例
    """
    global _default_logging_service

    _default_logging_service = LoggingService(
        sinks=sinks,
        enable_async=enable_async,
        buffer_size=buffer_size,
        flush_interval=flush_interval
    )

    return _default_logging_service