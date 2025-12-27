"""
1. 使用 Pydantic BaseModel 严格定义工具参数（禁止自动推断）
2. 真正的参数校验（类型、结构、必需参数）
3. 支持 LLM Function Calling（生成标准 JSON Schema）
4. 工具调用具有 State 语义（可被 Critic / Planner 观察）
5. Tool Execution Trace（记录到 state.observations）
"""
from typing import Dict, Callable, Any, Optional, Type, TypeVar
from functools import wraps
from pydantic import BaseModel, create_model, ValidationError
import json

from src.agent.state import State, ToolObservation

# 工具函数类型：必须接收 (state: State, args: BaseModel) -> Any
T = TypeVar('T', bound=BaseModel)


class ToolDefinition(BaseModel):
    """工具定义（严格 Schema）"""
    name: str
    description: str
    input_model: Type[BaseModel]  # Pydantic 模型，用于参数校验
    func: Callable[[State, BaseModel], Any]  # 工具函数
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换为 OpenAI Function Calling 格式的 JSON Schema
        
        Returns:
            OpenAI Function Calling 兼容的 schema
        """
        # 从 Pydantic 模型生成 JSON Schema
        json_schema = self.input_model.model_json_schema()
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", [])
                }
            }
        }
    
    def to_langchain_tool_schema(self) -> Dict[str, Any]:
        """
        转换为 LangChain Tool 格式的 JSON Schema
        
        Returns:
            LangChain Tool 兼容的 schema
        """
        json_schema = self.input_model.model_json_schema()
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", [])
            }
        }


class ToolRegistry:
    """
    Agent 级工具注册表
    
    核心能力：
    1. 强类型参数校验（Pydantic）
    2. Tool schema 可直接给 LLM（OpenAI / LangChain 兼容）
    3. 执行记录写入 State（observations）
    4. 错误结构化返回
    5. 可被 Critic / Planner 读取
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
    
    def register(
        self,
        name: str,
        description: str,
        input_model: Type[BaseModel]
    ):
        """
        装饰器：注册工具
        
        Args:
            name: 工具名称（唯一标识）
            description: 工具描述（用于 LLM Function Calling）
            input_model: Pydantic BaseModel，严格定义工具参数
            
        使用示例：
            class WeatherInput(BaseModel):
                city: str
                date: Optional[str] = None
            
            @tool_registry.register(
                name="get_weather",
                description="查询城市天气",
                input_model=WeatherInput
            )
            def get_weather(state: State, args: WeatherInput) -> str:
                result = ...
                # 工具执行会自动记录到 state.observations
                return result
        """
        def decorator(func: Callable[[State, BaseModel], Any]):
            # 验证函数签名
            if not callable(func):
                raise TypeError(f"工具 {name} 的 func 必须是可调用对象")
            
            # 创建工具定义
            tool_def = ToolDefinition(
                name=name,
                description=description,
                input_model=input_model,
                func=func
            )
            
            # 注册工具
            if name in self._tools:
                raise ValueError(f"工具 {name} 已注册，请使用不同的名称")
            
            self._tools[name] = tool_def
            
            # 返回原函数（不包装，因为调用时会在 execute 中处理）
            return func
        
        return decorator
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(name)
    
    def list_tools(self) -> Dict[str, ToolDefinition]:
        """列出所有已注册的工具"""
        return self._tools.copy()
    
    def get_openai_schemas(self) -> list[Dict[str, Any]]:
        """
        获取所有工具的 OpenAI Function Calling Schema
        
        Returns:
            OpenAI Function Calling 格式的工具列表
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]
    
    def get_langchain_schemas(self) -> list[Dict[str, Any]]:
        """
        获取所有工具的 LangChain Tool Schema
        
        Returns:
            LangChain Tool 格式的工具列表
        """
        return [tool.to_langchain_tool_schema() for tool in self._tools.values()]
    
    def execute(
        self,
        name: str,
        state: State,
        params: Dict[str, Any]
    ) -> Any:
        """
        执行工具
        1. 真正的参数校验（Pydantic ValidationError）
        2. 执行记录写入 state.observations（Tool Execution Trace）
        3. 错误结构化返回
        4. 支持 ReAct 观察循环
        """
        # 检查工具是否存在
        tool_def = self._tools.get(name)
        if not tool_def:
            raise ValueError(f"工具 {name} 未注册。可用工具: {list(self._tools.keys())}")
        
        # 参数校验（使用 Pydantic）
        try:
            # 将字典参数转换为 Pydantic 模型实例
            validated_args = tool_def.input_model(**params)
        except ValidationError as e:
            # 参数校验失败，记录到 observations
            observation = ToolObservation(
                tool=name,
                args=params,
                result=None,
                success=False,
                error=f"参数校验失败: {e.json()}"
            )
            state.observations.append(observation)
            raise ValueError(f"工具 {name} 参数校验失败: {e}")
        
        # 执行工具
        try:
            result = tool_def.func(state, validated_args)
            
            # 记录成功执行到 observations（Tool Execution Trace）
            observation = ToolObservation(
                tool=name,
                args=validated_args.model_dump(),
                result=result,
                success=True,
                error=None
            )
            state.observations.append(observation)
            
            return result
            
        except Exception as e:
            # 记录失败执行到 observations
            observation = ToolObservation(
                tool=name,
                args=validated_args.model_dump(),
                result=None,
                success=False,
                error=str(e)
            )
            state.observations.append(observation)
            
            # 重新抛出异常，让调用方处理
            raise Exception(f"工具 {name} 执行失败: {str(e)}")
    
    def validate_params(
        self,
        name: str,
        params: Dict[str, Any]
    ) -> BaseModel:
        """
        仅校验参数（不执行工具）
        """
        tool_def = self._tools.get(name)
        if not tool_def:
            raise ValueError(f"工具 {name} 未注册")
        
        return tool_def.input_model(**params)


# 全局工具注册表实例
tool_registry = ToolRegistry()
