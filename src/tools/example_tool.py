from typing import Optional

from pydantic import BaseModel, Field

from src.agent.state import State


class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，例如：北京、上海")
    date: Optional[str] = Field(
        default=None,
        description="日期，格式：YYYY-MM-DD，默认为今天"
    )

def get_weather(state: State, args: WeatherInput):
    if state.retrieved_docs:
        #  结合RAG 文档
        pass
    # 执行工具逻辑
    if args.date:
        result = f"{args.city}在{args.date}的天气：晴朗，温度 25°C，湿度 60%"
    else:
        result = f"{args.city}今天天气晴朗，温度 25°C，湿度 60%"

    if "weather_history" not in state.memory:
        state.memory["weather_history"] = []

    state.memory["weather_history"].append({
        "city": args.city,
        "date": args.date or "today",
        "result": result,
    })

    return result