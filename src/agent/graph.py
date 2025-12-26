from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.state import State
from src.agent.nodes.intent_router import IntentRouterNode
from src.agent.nodes.planner import PlannerNode
from src.agent.nodes.critic import CriticNode

# 构建状态图
builder = StateGraph(State)

# 添加意图识别节点
intent_router_node = IntentRouterNode()
planner_node = PlannerNode()
critic_node = CriticNode()


builder.add_node("intent_router", intent_router_node.run)
builder.add_node("planner", planner_node.run)
builder.add_node("critic", critic_node.run)

builder.add_edge(START,"intent_router")
builder.add_edge("intent_router","planner")
builder.add_edge("planner","critic")

def route_after_critic(state: State):
    decision = getattr(state,"critic_decision",None)
    if decision == "accept":
        print("检查通过，路由到END")
        return "END"
    elif decision == "reject":
        reason = getattr(state,"critic_reason","")
        print(reason)
        if "超出最大重试次数" in reason:
            return "END"
        else:
            return "planner"
    else:
        return "planner"

builder.add_conditional_edges("planner",
                              route_after_critic,
                              ["planner", "END"])


builder.add_edge("",END)

# 编译得到可运行的图
graph = builder.compile()
