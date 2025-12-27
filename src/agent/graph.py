from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.state import State
from src.agent.nodes.intent_router import IntentRouterNode
from src.agent.nodes.planner import PlannerNode
from src.agent.nodes.critic import CriticNode
from src.agent.nodes.draft_answer import DraftAnswerNode
from src.agent.nodes.retrieval import RetrievalNode
from src.agent.nodes.retrieval_decision import RetrievalDecisionNode

# 构建状态图
builder = StateGraph(State)

# 添加意图识别节点
intent_router_node = IntentRouterNode()
planner_node = PlannerNode()
critic_node = CriticNode()
draft_answer_node = DraftAnswerNode()
retrieval_node = RetrievalNode()
retrieval_decision_node = RetrievalDecisionNode()


builder.add_node("intent_router", intent_router_node.run)
builder.add_node("planner", planner_node.run)
builder.add_node("critic", critic_node.run)
builder.add_node("draft_answer", draft_answer_node.run)
builder.add_node("retrieval", retrieval_node.run)
builder.add_node("retrieval_decision", retrieval_decision_node.run)

builder.add_edge(START,"intent_router")
builder.add_edge("intent_router","planner")
builder.add_edge("planner","retrieval_decision")
builder.add_edge("retrieval","draft_answer")
builder.add_edge("draft_answer","critic")


def route_after_critic(state: State):
    decision = getattr(state,"critic_decision",None)
    print(f"当前执行状态为{decision}")
    if decision == "accept":
        print("检查通过，路由到END")
        return END
    elif decision == "reject":
        reason = getattr(state,"critic_reason","")
        print(reason)
        if "超出最大重试次数" in reason:
            return END
        else:
            return "planner"
    else:
        return "planner"

builder.add_conditional_edges("critic",
                              route_after_critic,
                              ["planner", END])

def route_after_retrieval(state: State):
    decision = getattr(state,"need_retrieval",True)
    if decision:
        return "retrieval"
    else:
        return "draft_answer"

builder.add_conditional_edges("retrieval_decision",
                              route_after_retrieval,
                              ["retrieval", "draft_answer"])

# 编译得到可运行的图
graph = builder.compile()
