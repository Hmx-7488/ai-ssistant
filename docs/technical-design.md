# 技术设计文档

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户层                               │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │   Web UI    │  │   CLI 聊天  │                          │
│  └──────┬──────┘  └──────┬──────┘                          │
└─────────┼───────────────┼──────────────────────────────────┘
          │               │
          ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    API 层（FastAPI :8084）                    │
│  POST /chat  |  GET /health  |  GET /（前端）               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Agent 层（LangGraph StateGraph）                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            RouterAgent（意图识别）                     │  │
│  │  关键词规则优先 → LLM兜底分类（懒加载）               │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  ▼          ▼          ▼          ▼          ▼          ▼  │
│ MenuQA   Recommend  Reservation  Complaint  Human    Fallback│
│ Agent     Agent      Agent       Agent     Transfer   Node  │
│                                                             │
│  每个Agent: 确定性快速路径 → ToolAgent(LLM+Tools)兜底      │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ToolAgent 工具层                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  9个 @tool 函数（LangChain Function Calling）        │   │
│  │  query_dish | search_dishes | get_full_menu          │   │
│  │  check_allergen | recommend_combo | recommend_for_people │
│  │  query_faq | get_promotions | get_restaurant_info    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    数据层                                    │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐    │
│  │  menu.json    │ │ restaurant_   │ │  ChromaDB     │    │
│  │  菜品数据     │ │ faq.json      │ │  向量数据库    │    │
│  │  (单一事实源)  │ │ FAQ知识库     │ │  RAG检索      │    │
│  └───────────────┘ └───────────────┘ └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 模块设计

### 1. RouterAgent（意图识别）

```
职责：判断用户问题类型，分发给对应Agent
输入：用户消息
输出：意图类别

8个意图：价格查询 | 菜品推荐 | 点餐咨询 | 订座服务 | 投诉建议 | 订单查询 | 转人工 | 闲聊互动

分类策略（优先级从高到低）：
  1. 菜系关键词（江西菜/川菜等）→ 点餐咨询
  2. 菜品名称优先（红烧肉/烤鱼等）→ 点餐咨询（价格词除外）
  3. 价格关键词 → 价格查询
  4. 意图优先级循环：转人工 > 投诉 > 订单 > 推荐 > 订座 > 点餐 > 闲聊
  5. LLM兜底（懒加载，规则未命中时自动调用）
```

### 2. ToolAgent（LLM + Function Calling 基类）

```python
class ToolAgent:
    """LLM + bind_tools() + 工具调用循环"""
    # LLM.bind_tools() 绑定工具列表
    # invoke() 循环：LLM推理 → 判断tool_calls → 执行工具 → 喂回结果 → 重复
    # 最多5轮工具调用，最终返回文本回复

4个专用ToolAgent：
  - MenuToolAgent: MENU_TOOLS + BOOKING_TOOLS
  - RecommendToolAgent: RECOMMEND_TOOLS + MENU_TOOLS
  - BookingToolAgent: BOOKING_TOOLS
  - ComplaintToolAgent: COMPLAINT_TOOLS
```

### 3. 9个工具函数（agent_tools.py）

```
query_dish(dish_name)           查菜品详情
search_dishes(category,tag,..)  按条件搜索菜品
get_full_menu()                 获取完整菜单
check_allergen(dish,allergen)   检查过敏原
recommend_combo(people,budget)  预算套餐推荐
recommend_for_people(type)      人群推荐
query_faq(question)             FAQ查询
get_promotions()                优惠活动
get_restaurant_info(type)       餐厅信息
```

### 4. FAQ引擎（faq_engine.py）

```
50+ 结构化问答，7个分类
匹配策略：菜品食材检查 → 加权关键词匹配（长词权重更高）
上下文排除：防止"热菜"匹配空调FAQ
100%确定性，不调用LLM
```

### 5. ConversationState（多轮对话状态）

```python
class ConversationState:
    people: int          # 人数
    budget: int          # 预算
    people_type: str     # 人群类型（儿童/老人/孕妇/健身）
    taste: str           # 口味偏好（辣/不辣/清淡）
    allergies: list      # 过敏原/忌口
    scene: str           # 场景（一人食/约会/聚餐）
    last_dishes: list    # 上轮推荐的菜品
    last_reply: str      # 上轮回复
    message_history: list # 最近10轮对话 [(user, ai), ...]
```

---

## 🔑 核心代码结构

### ToolAgent 基类（Function Calling）

```python
class ToolAgent:
    def __init__(self, llm, system_prompt: str, tools: list):
        self.tools_by_name = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools)

    def invoke(self, user_input: str, context: str = "", history: list = None) -> str:
        messages = [SystemMessage(content=self.system_prompt)]
        # 注入对话历史（最近3轮）
        if history:
            for user_msg, ai_msg in history[-3:]:
                messages.append(HumanMessage(content=user_msg))
                messages.append(AIMessage(content=ai_msg))
        messages.append(HumanMessage(content=user_input))

        for _ in range(5):  # 最多5轮工具调用
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            if not ai_msg.tool_calls:
                return ai_msg.content or ""
            for tc in ai_msg.tool_calls:
                result = self.tools_by_name[tc["name"]].invoke(tc["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return self.llm_with_tools.invoke(messages).content or ""
```

### 工具函数示例

```python
from langchain_core.tools import tool

@tool
def query_dish(dish_name: str) -> str:
    """Query details of a specific dish by name."""
    kb = get_unified_kb()
    dish = kb.find_dish(dish_name)
    if not dish:
        return f'未找到菜品 "{dish_name}"'
    return kb.format_dish_info(dish)

@tool
def recommend_combo(people: int = 3, budget: int = 0,
                    avoid_allergens: str = "", taste: str = "") -> str:
    """Recommend a combo of dishes for a group."""
    # 预算算法：套餐检查 → 分类搭配（主菜→配菜→汤→饮品）
    # 过敏原过滤 + 口味筛选
    ...
```

### LangGraph 状态图

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(GraphState)
graph.add_node("router", router_node)
graph.add_node("menu", menu_node)
graph.add_node("recommend", recommend_node)
graph.add_node("reservation", reservation_node)
graph.add_node("complaint", complaint_node)
graph.add_node("human", human_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_intent, {...})
for node in ["menu", "recommend", ...]:
    graph.add_edge(node, END)
```

---

## 📊 数据设计

### 菜品数据（menu.json - 单一事实源）

```json
{
  "菜品": [
    {
      "id": 1,
      "name": "招牌红烧肉",
      "price": 58,
      "category": "热菜",
      "description": "精选五花肉慢炖3小时，入口即化",
      "spice": "不辣",
      "portion": "适合2-3人",
      "tags": ["招牌", "人气必点", "下饭", "肉食"],
      "allergens": [],
      "customizable": true,
      "cook_time": "15分钟"
    }
  ],
  "套餐": [...],
  "人群标签": {...},
  "场景推荐": {...},
  "优惠活动": {...},
  "餐厅信息": {...}
}
```

### FAQ数据（restaurant_faq.json）

```json
{
  "营业信息": {
    "营业时间": {
      "keywords": ["营业", "几点", "开门", "关门"],
      "answer": "午餐 11:00-14:00  晚餐 17:00-22:00"
    }
  },
  "服务设施": {...},
  "优惠活动": {...}
}
```

### 会话状态（内存）

```python
# ConversationState - 每个session_id独立
sessions = {
    "user_123": ConversationState(
        people=3, budget=150, allergies=["不吃香菜"],
        message_history=[("推荐菜", "推荐红烧肉..."), ("换一个", "清炒时蔬...")]
    )
}
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
