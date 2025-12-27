
---

# AgentOps-Lite：设计方案

## 一、项目目标

> **让 LLM / Agent 在复杂任务下可控、可观测、可恢复，支持多模型、多工具、多知识源，并具备生产级后端工程能力。**

* 面向企业内部知识问答、任务自动化、技术支持等场景
* 解决不可控、黑盒、工具失败、上下文过长、RAG 滥用等问题

---

## 二、项目结构（对应 8 个能力点）

```
agentops_lite/
├─ app/
│  ├─ api/                     # FastAPI 接口层（后端系统能力）
│  │  ├─ endpoints.py          # 对外调用接口
│  │  └─ health.py             # 健康检查
│  │
│  ├─ agent/                   # Agent 核心工作流（架构 + Graph）
│  │  ├─ graph.py              # Graph DAG/FSM 定义
│  │  ├─ state.py              # AgentState 全局状态（Memory / Context / Long-term memory）
│  │  └─ nodes/                # 各 Node 实现
│  │      ├─ intent_router.py      # Intent 分流（LLM 调用与编排）
│  │      ├─ planner.py            # Plan-Execute 核心规划（Agent 架构）
│  │      ├─ retrieval_decision.py # 是否触发 RAG（RAG 系统 + 决策）
│  │      ├─ retrieval.py          # 向量检索节点（RAG）
│  │      ├─ tool_execution.py     # 工具调用节点（Tools / Function Calling）
│  │      ├─ draft_answer.py       # 草稿答案生成节点（LLM 调用）
│  │      └─ critic.py             # 反思/重试节点（可靠性与可观测性）
│  │
│  ├─ rag/                     # 向量检索 + Query Rewrite（RAG 系统）
│  │  ├─ vector_store.py
│  │  └─ retriever.py
│  │
│  ├─ tools/                   # 工具系统
│  │  ├─ tool_registry.py      # Tool 注册与 schema
│  │  └─ example_tool.py
│  │
│  ├─ services/                # 后端服务逻辑（API、并发、缓存、日志）
│  │  ├─ llm_client.py         # LLM 抽象接口（OpenAI / Claude / 本地模型）
│  │  ├─ agent_executor.py     # Graph 执行器
│  │  └─ logging_service.py    # trace / retry / eval
│  │
│  └─ config/                  # 配置
│     ├─ settings.py
│     └─ logging_config.py
├─ tests/                       # 单元测试与集成测试
├─ scripts/                     # demo / 测试脚本
├─ docker/                       # Docker 化部署
├─ requirements.txt
└─ README.md
```

---

## 三、模块与能力点映射

| 能力点            | 对应模块 / Node                                        | 实现要点                                                                   |
| -------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| **LLM 调用与编排**  | `llm_client.py` + `planner.py` + `draft_answer.py` | 多 Provider 支持（OpenAI / Claude / 本地），不同节点选用不同模型，支持 rate limit / timeout |
| **Agent 架构设计** | `graph.py` + `planner.py` + `critic.py`            | Plan-Execute 核心，局部 ReAct，Self-Reflection / Critic，可扩展 Multi-Agent      |
| **工具系统**       | `tool_execution.py` + `tools/tool_registry.py`     | 工具 schema 校验、Function Calling、失败捕获，Agent 能感知工具结果                       |
| **状态管理**       | `state.py` + Node 输入输出                             | Graph State、短期 Memory、Long-term Memory（Redis / DB）、上下文裁剪               |
| **工作流与图**      | `graph.py` + Node DAG                              | LangGraph / DAG / FSM，条件分支、循环、失败路径，清晰 Graph 调试和扩展                      |
| **RAG 系统**     | `retrieval_decision.py` + `retrieval.py`           | 决策是否检索，Query Rewrite，检索失败 fallback，Agent 与向量库耦合                        |
| **后端系统工程能力**   | `api/` + `services/`                               | FastAPI、async/await、缓存（Redis）、数据库（PostgreSQL）、队列（可选），长任务并发控制、限流、降级     |
| **可靠性与可观测性**   | `critic.py` + `logging_service.py`                 | trace_id、节点级日志、retry 策略、最大执行步数，支持 eval 分析和 debug                       |

---

## 四、落地实现思路

1. **Graph + Node**

   * 每个 Node 只处理自己职责，修改 AgentState 对应字段
   * Critic Node 判断失败并触发 retry
   * Graph Executor 顺序执行 Node + 异常捕获

2. **LLM 调用抽象**

   * llm_client.py 支持不同 Provider 切换
   * Planner 用高推理模型，Critic 用低成本模型
   * 支持 retry 和超时

3. **工具系统**

   * Tool Registry 管理所有可调用工具
   * schema 校验参数
   * 调用失败记录而非直接抛异常

4. **RAG 系统**

   * 决策是否触发检索
   * Query Rewrite → 向量库检索 → 文档摘要
   * 检索失败 fallback

5. **状态管理**

   * AgentState + Redis / PostgreSQL
   * 短期内存用于执行上下文
   * 长期内存用于历史问答或任务记录

6. **后端工程能力**

   * FastAPI 接口 + async
   * 并发控制，队列执行长任务
   * 日志与 trace 保存

7. **可靠性与可观测性**

   * trace_id 贯穿整个 Graph
   * 节点级日志记录输入/输出
   * Critic 节点 retry、最大步数限制
   * Eval 模块统计 Agent 成功率 / 错误类型

---

