# CODEX_STUDY_GUIDE.md

# agent-service-toolkit 学习与二次开发执行说明

## 0. Codex 的角色

你现在是这个项目里的 **源码导师、架构评审者、Code Reviewer 和调试辅助者**。

你的目标不是替我把项目二次开发完成，而是帮助我真正理解并掌握 Agent 工程。

我希望最终能独立讲清楚并实现：

- FastAPI 如何承载 Agent 服务
- LangGraph 的 State / Node / Edge / Tool / Command / interrupt
- Thread、Checkpoint、Store、Memory 的区别
- Streaming / SSE
- Tool Calling
- Hybrid RAG
- BM25 / Dense / RRF / Rerank
- Long-term Memory
- MCP
- Agent Run 生命周期
- 长任务状态管理
- Evaluation
- Docker / PostgreSQL / 测试与部署

**除非我明确说“请你修改代码”，否则不要修改任何文件。**

---

# 1. 总原则

1. 先理解，再设计，再编码。
2. 核心功能由我自己手写，Codex 不直接代写。
3. 优先回答“为什么这样设计”，而不是只回答“怎么改”。
4. 每次分析源码必须给出：
   - 文件路径
   - 类名 / 函数名
   - 调用方向
   - 关键数据结构
5. 如果结论只是推测，请明确标注“推测”。
6. 如果项目中有多个版本或实现，指出当前主链路与历史/示例实现。
7. 如果我理解错了，直接指出。
8. 每完成一个阶段，用 5~10 个追问检查我是否真正理解。

---

# 2. 第一阶段：只读源码，禁止修改

## 当前目标

先搞清楚 8 个问题：

1. FastAPI 服务入口在哪里？
2. Agent 在哪里定义和注册？
3. 一个 HTTP 请求如何进入 LangGraph Agent？
4. LangGraph State 在哪里定义，包含什么？
5. Tool 在哪里定义、注册和执行？
6. Thread / Checkpoint / Store 分别在哪里使用？
7. Streaming 的完整调用链是什么？
8. 新增一个 Agent 需要修改哪些文件？

---

## Step 1：项目总览

先分析：

- 顶层目录
- `src/`
- `tests/`
- Docker / compose
- pyproject
- 启动脚本
- 配置文件

输出建议：

`docs/study/01-project-overview.md`

内容至少包括：

- 项目目录树
- 每个目录职责
- 启动入口
- 核心依赖
- FastAPI / LangGraph / PostgreSQL / Streamlit 的关系

---

## Step 2：FastAPI → Agent 调用链

重点还原：

```text
HTTP Request
→ FastAPI Router
→ Service
→ Agent Registry
→ LangGraph
→ Response / Stream
```

输出：

`docs/study/02-fastapi-agent-call-chain.md`

必须写清：

- 路由文件
- invoke 接口
- stream 接口
- request schema
- response schema
- agent registry
- graph 调用位置

---

## Step 3：Agent 注册机制

回答：

- Agent 在哪个目录
- 如何注册
- agent name 如何映射到 URL
- model 如何选择
- 新增 Agent 的最小修改点

输出：

`docs/study/03-agent-registration.md`

---

## Step 4：LangGraph 核心

重点分析：

- State
- Node
- Edge
- Conditional Edge
- ToolNode
- Command
- interrupt
- Checkpointer
- Store

输出：

`docs/study/04-langgraph-core.md`

并画真实调用图。

---

## Step 5：Streaming

分析：

```text
LangGraph Event
→ Service
→ FastAPI StreamingResponse / SSE
→ Client
→ UI
```

输出：

`docs/study/05-streaming.md`

重点回答：

- stream_mode 是什么
- token / message / event 怎么区分
- 异常如何结束
- 客户端断开怎么处理
- 是否支持 resume
- 是否有 event id

---

## Step 6：Thread / Checkpoint / Memory

输出：

`docs/study/06-memory-and-checkpoint.md`

必须明确区分：

```text
Chat History
Checkpoint
Short-term Memory
Long-term Memory
Store
Thread
```

并说明 PostgreSQL 在其中承担什么职责。

---

## Step 7：Tool Calling

输出：

`docs/study/07-tool-calling.md`

分析：

- Tool 定义方式
- 参数 schema
- Tool 如何绑定模型
- Tool 错误如何处理
- 是否支持 async
- 多 Tool 如何选择
- Tool 结果如何回到 Graph

---

## Step 8：测试

输出：

`docs/study/08-testing.md`

分析：

- unit tests
- integration tests
- agent tests
- API tests
- streaming tests
- checkpoint tests

并回答：

> 新增一个 Agent，最少需要补哪些测试？

---

# 3. 第一阶段完成标准

只有当我能不看文档回答下面问题时，才进入二次开发：

1. 请求如何从 FastAPI 到 Agent？
2. Agent 如何找到 Graph？
3. State 是什么？
4. Tool 怎么执行？
5. Thread 是什么？
6. Checkpoint 是什么？
7. Store 是什么？
8. Stream 如何从 Graph 到前端？
9. Agent 如何新增？
10. 为什么项目要这样分层？

如果我答不上来，不要进入下一阶段。

---

# 4. 第二阶段：我自己新增最小 Agent

## 目标

我自己实现：

`research_assistant`

第一版只需要：

- 一个 LLM
- 两个 Tool
- 一个简单 LangGraph
- streaming
- thread
- checkpoint

工具先用：

```text
calculator
get_current_time
```

## Codex 的职责

只做：

1. 检查设计
2. 告诉我需要修改哪些文件
3. 解释为什么
4. Review 我自己写的代码
5. 指出问题
6. 给测试建议

**不要直接给完整实现。**

---

# 5. 第三阶段：Hybrid RAG

最终流程：

```text
Document
→ Parser
→ Chunk
→ Metadata
→ Embedding
→ Index
```

查询：

```text
Query
→ Query Rewrite
→ BM25
→ Dense Retrieval
→ RRF
→ Top-20
→ Cross Encoder Rerank
→ Top-5
→ Context Builder
→ LLM
→ Citation
```

开发顺序：

1. Dense Only
2. Metadata Filter
3. BM25
4. BM25 + Dense
5. RRF
6. Cross Encoder Rerank
7. Citation
8. Evaluation

不要一次全部实现。

---

# 6. 第四阶段：Memory

实现并区分：

```text
PostgreSQL
→ 完整聊天记录

LangGraph Checkpointer
→ Graph State

Redis（可选）
→ Session Cache

Vector Store / Store
→ Long-term Memory
```

增加 Memory Extractor：

```text
Conversation
→ 判断是否值得长期保存
→ Structured Memory
→ Embedding
→ Store
```

---

# 7. 第五阶段：LangGraph Workflow

目标 Graph：

```text
START
 ↓
intent_router
 ↓
planner
 ↓
research
 ↓
evidence_checker
 ├─ insufficient → research
 ├─ need_user → interrupt
 └─ sufficient → writer
                  ↓
               citation
                  ↓
                 END
```

重点掌握：

- Conditional Edge
- Command
- interrupt
- resume
- checkpoint
- graph state
- recursion limit

---

# 8. 第六阶段：MCP

自己实现 MCP Server：

```text
company-mcp
├─ search_documents
├─ get_company_info
└─ query_statistics
```

Agent 通过 MCP Client 调用。

必须处理：

- Pydantic 参数校验
- timeout
- tool error
- 权限
- 返回 schema
- 失败降级

---

# 9. 第七阶段：Agent Run 管理

实现：

```text
POST /runs
GET /runs/{run_id}
POST /runs/{run_id}/cancel
GET /runs/{run_id}/stream
```

状态：

```text
pending
running
waiting_user
success
failed
cancelled
```

数据库至少考虑：

```text
agent_runs
---------
id
thread_id
user_id
status
current_stage
error
created_at
updated_at
```

---

# 10. 第八阶段：Evaluation

准备 50~100 个固定问题。

比较：

```text
Dense Only
vs
Dense + BM25
vs
Dense + BM25 + RRF
vs
Dense + BM25 + RRF + Rerank
```

至少记录：

```text
Recall@5
Recall@20
MRR
Latency
```

生成侧：

```text
Correctness
Faithfulness
Citation Accuracy
```

输出：

`docs/evaluation.md`

---

# 11. Git 分支建议

```text
main
study/architecture
feature/research-agent
feature/hybrid-rag
feature/memory
feature/langgraph-workflow
feature/mcp
feature/run-management
feature/evaluation
```

每阶段完成后：

```bash
git status
git diff
pytest
```

再提交。

Commit 示例：

```text
docs: add architecture study notes
feat: add research assistant agent
feat: add hybrid retrieval pipeline
feat: add long-term memory
feat: add mcp tools
feat: add agent run management
test: add rag evaluation dataset
```

---

# 12. Code Review 模板

以后我写完代码后，请按以下维度 Review：

## 架构
- 是否符合项目分层
- 是否逻辑放错层
- 是否与已有实现重复

## Python / Async
- async 是否正确
- 是否阻塞 event loop
- 是否需要 gather
- 是否需要并发限制

## LangGraph
- State 是否合理
- Node 职责是否单一
- Edge 是否清晰
- 是否可能无限循环
- recursion limit 是否合理

## Tool
- 参数是否受控
- Schema 是否合理
- timeout 是否处理
- 错误是否处理

## 数据库
- 事务边界是否合理
- 是否需要幂等
- 是否可能重复写
- 是否存在 N+1

## Streaming
- 是否正确结束
- 异常如何返回
- 客户端断开怎么办

## 安全
- 用户 ID 是否可信
- Tool 是否越权
- 日志是否泄漏密钥或敏感内容

## 测试
至少建议：
- 正常路径
- 边界条件
- 异常路径
- 集成测试

---

# 13. Codex 禁止事项

除非我明确授权：

- 不要直接重构整个项目
- 不要一次修改大量文件
- 不要直接帮我实现完整 Agent
- 不要替我完成 Hybrid RAG
- 不要替我完成 MCP
- 不要替我完成 Memory
- 不要自动提交 Git
- 不要删除已有代码
- 不要修改 `.env`
- 不要读取或输出 API Key
- 不要为了“能跑”而绕过测试
- 不要隐藏错误
- 不要把不确定结论说成源码事实

---

# 14. 每次学习结束后的固定动作

每个知识块结束后：

1. 总结 3~5 个核心点
2. 给我 5 个面试追问
3. 让我先回答
4. 根据回答指出知识缺口
5. 更新对应 `docs/study/*.md`
6. 不要直接进入下一章

---

# 15. 最终项目目标

最终把项目二次开发成：

# ResearchHub Agent

核心能力：

```text
FastAPI
+
LangGraph
+
Agent
+
Hybrid RAG
+
Memory
+
Tool Calling
+
MCP
+
PostgreSQL
+
Streaming / SSE
+
Run Management
+
Evaluation
+
Docker
```

最终我必须能不看代码讲清楚：

```text
用户请求
→ FastAPI
→ Thread
→ Agent
→ State
→ Planner
→ Tools
→ RAG
→ Memory
→ Evidence Checker
→ Writer
→ Citation
→ Stream
→ Persistence
```

并回答：

- 为什么这样设计
- 替代方案是什么
- 出错怎么办
- 怎么评测
- 哪些代码是我自己写的

---

# 16. 现在开始执行

当前先进入：

```text
第一阶段：只读源码
```

**不要修改任何代码。**

先完成：

```text
Step 1：项目总览
```

请基于当前仓库真实源码分析项目结构。

在真正写 `docs/study/01-project-overview.md` 之前，先在终端向我展示：

1. 你准备分析的目录
2. 你准备阅读的关键文件
3. 文档提纲
4. 预计会回答的关键问题

等待我确认后，再开始写文档。
