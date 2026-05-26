# 技术设计文档

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Web UI    │  │   API 调用  │  │   微信/钉钉 │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼───────────────┼───────────────┼───────────────────┘
          │               │               │
          ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                       API 层（FastAPI）                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  POST /chat                                         │   │
│  │  - 接收用户消息                                       │   │
│  │  - 返回 Agent 回复                                   │   │
│  │  - 管理会话状态                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agent 层（LangGraph）                   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Router Agent                          │  │
│  │  职责：判断问题类型，分发给对应 Agent                   │  │
│  │  分类：知识库问题 / 订单问题 / 闲聊 / 转人工           │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│           ┌──────────────┼──────────────┐                  │
│           ▼              ▼              ▼                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │  RAG Agent   │ │  SQL Agent   │ │  Chat Agent  │      │
│  │              │ │              │ │              │      │
│  │ 知识库问答   │ │ 订单查询     │ │ 闲聊/问候    │      │
│  │ 检索文档回答 │ │ 自然语言转SQL│ │ 直接回复     │      │
│  └──────┬───────┘ └──────┬───────┘ └──────────────┘      │
│         │                │                                 │
└─────────┼────────────────┼─────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────┐ ┌─────────────────┐
│   ChromaDB      │ │   SQLite        │
│   向量数据库     │ │   关系数据库     │
│   存储文档Embedding│ │  存储订单数据   │
└─────────────────┘ └─────────────────┘
```

---

## 📦 模块设计

### 1. Router Agent（路由 Agent）

```python
# 职责：判断用户问题类型
# 输入：用户消息
# 输出：问题类别 + 分发给对应 Agent

问题分类：
  - knowledge_base: 知识库相关问题（产品功能、使用方法）
  - order_query: 订单相关问题（查订单、查物流）
  - chitchat: 闲聊（问候、感谢）
  - human_handoff: 需要人工处理（投诉、复杂问题）
```

### 2. RAG Agent（知识库问答 Agent）

```python
# 职责：基于知识库文档回答问题
# 流程：
#   1. 接收用户问题
#   2. 将问题转为向量
#   3. 在 ChromaDB 中检索相似文档
#   4. 将检索结果 + 问题发给 LLM
#   5. LLM 生成回答

关键技术：
  - 文档分割：RecursiveCharacterTextSplitter
  - 向量化：OpenAI Embedding / 本地模型
  - 检索策略：相似度检索 + MMR
```

### 3. SQL Agent（订单查询 Agent）

```python
# 职责：用自然语言查询订单数据库
# 流程：
#   1. 接收用户问题
#   2. Agent 分析数据库结构
#   3. 生成 SQL 查询
#   4. 执行 SQL
#   5. 将结果转为自然语言回答

关键技术：
  - SQLDatabase：封装数据库连接
  - create_sql_agent：创建 SQL Agent
  - 安全限制：只允许 SELECT 查询
```

---

## 🔑 核心代码结构

### 主 Agent（整合所有子 Agent）

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

# 定义工具
@tool
def ask_knowledge_base(question: str) -> str:
    """查询知识库，回答产品相关问题"""
    return rag_agent.invoke(question)

@tool
def query_order(query: str) -> str:
    """查询订单、物流等信息"""
    return sql_agent.invoke(query)

# 创建主 Agent
customer_service_agent = create_agent(
    model="openai:qwen-plus",
    tools=[ask_knowledge_base, query_order],
    system_prompt=CUSTOMER_SERVICE_PROMPT,
)
```

### FastAPI 接口

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    reply: str
    agent_used: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await customer_service_agent.ainvoke({
        "messages": [("user", request.message)]
    })
    return ChatResponse(
        reply=result["messages"][-1].content,
        agent_used="router"
    )
```

---

## 📊 数据库设计

### 订单表（orders）

```sql
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT,
    product_name TEXT,
    quantity INTEGER,
    total_price REAL,
    status TEXT,  -- pending/shipped/delivered
    created_at TIMESTAMP,
    shipped_at TIMESTAMP,
    delivered_at TIMESTAMP
);
```

### 用户表（users）

```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    phone TEXT,
    email TEXT,
    address TEXT
);
```

### 物流表（logistics）

```sql
CREATE TABLE logistics (
    logistics_id TEXT PRIMARY KEY,
    order_id TEXT,
    carrier TEXT,
    tracking_number TEXT,
    status TEXT,
    updated_at TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
```

---

## 🛡️ 安全设计

### SQL Agent 安全
```
1. 只允许 SELECT 查询
2. 禁止访问敏感表（users.password）
3. 限制返回行数（最多 100 行）
4. 使用只读数据库连接
```

### API 安全
```
1. 添加请求频率限制
2. 输入长度限制
3. 错误信息不暴露内部细节
```

---

## 📈 性能优化

### 1. 检索优化
- 使用 MMR（最大边际相关性）提高检索多样性
- 限制返回文档数量
- 使用本地 Embedding 模型减少 API 调用

### 2. 响应优化
- 使用流式输出（Streaming）
- 异步处理
- 添加缓存（常见问题）

### 3. 成本优化
- 限制 LLM 输出长度
- 使用更便宜的模型处理简单问题
- 缓存常见问题的回答

---

## 🧪 测试策略

### 单元测试
```python
# 测试 RAG Agent
def test_rag_agent():
    result = rag_agent.invoke("如何重置密码？")
    assert "密码" in result
    assert len(result) > 50

# 测试 SQL Agent
def test_sql_agent():
    result = sql_agent.invoke("查询订单 ORD001 的状态")
    assert "ORD001" in result
```

### 集成测试
```python
# 测试完整流程
def test_customer_service():
    result = agent.invoke("我的订单到哪了？")
    assert result is not None
    assert len(result) > 0
```

---

## 📝 面试常见问题

### Q: 为什么用多 Agent 而不是单个 Agent？
```
A: 
1. 专业化：每个 Agent 专注一类问题，效果更好
2. 可维护：修改一个 Agent 不影响其他
3. 可扩展：新增功能只需添加新 Agent
4. 成本控制：简单问题用小模型，复杂问题用大模型
```

### Q: 如何处理 LLM 幻觉？
```
A:
1. RAG 检索：基于真实文档回答，减少编造
2. Prompt 约束：明确要求"不知道就说不知道"
3. 结果验证：SQL 查询结果直接来自数据库
4. 置信度判断：低置信度时转人工
```

### Q: 如何评估系统效果？
```
A:
1. 准确率：回答是否正确
2. 召回率：是否能找到相关文档
3. 响应时间：用户等待时间
4. 用户满意度：人工评估
```
