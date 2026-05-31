# 🍚 饭小二 - 餐饮AI客服助手

基于 **LangChain + LangGraph + RAG + Function Calling** 的餐厅智能客服系统，人称"美食界相声演员"。

## ✨ 功能特点

- **真正多Agent架构**：ToolAgent + Function Calling，LLM 自主决策调用工具（9个专用工具函数）
- **LangGraph 编排**：8 个节点状态图路由，关键词规则优先 + LLM 兜底
- **RAG 增强**：ChromaDB 向量检索 + 结构化知识库混合检索
- **三层架构**：确定性查询 → 结构化推荐 → LLM+Tool Calling 兜底
- **多轮对话**：上下文继承 + 对话记忆（最近10轮），"换个清淡的""改成3个人"都能接住
- **约束识别**：人数、预算、人群、过敏原、口味偏好、场景
- **FAQ 引擎**：50+ 结构化问答，加权关键词匹配，100% 确定性
- **单一事实源**：`menu.json` 统一数据，一处修改全局生效

## 🚀 快速开始

### 1. 环境准备

```bash
git clone <repository-url>
cd ai-assistant

python -m venv .venv

# 激活虚拟环境（每次打开终端都要执行）
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置环境变量

项目根目录创建 `.env`：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 启动后端

```bash
.venv\Scripts\Activate.ps1

# 方式一：一键启动（推荐）
python start.py

# 方式二：手动启动
uvicorn src.api.main:app --host 0.0.0.0 --port 8084

# 方式三：开发模式（代码修改自动重载）
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8084
```

看到 `饭小二准备就绪！` 说明启动成功。

> 启动时会自动构建 RAG 向量知识库（menu.json + restaurant_faq.md → ChromaDB）

### 4. 打开前端

浏览器访问 **http://localhost:8084**，即可看到聊天界面。

> ⚠️ 前端是纯静态 HTML，**不需要 npm / Node.js**。
> 不要执行 `cd src/frontend && npm run dev`。

### 5. 测试

```bash
# 运行 40 项验收测试（需要先启动后端）
python preflight.py

# CLI 交互式聊天（终端直接对话）
python cli_chat.py

# curl 测试 API
curl -X POST http://localhost:8084/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"红烧肉多少钱？\"}"

# 多轮对话测试（同一 session_id）
curl -X POST http://localhost:8084/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"推荐几个菜\", \"session_id\": \"test1\"}"

curl -X POST http://localhost:8084/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"换个清淡的\", \"session_id\": \"test1\"}"
```

### 6. 停止

后端终端按 `Ctrl + C`。

---

## 📁 项目结构

```
ai-assistant/
├── src/
│   ├── agents/
│   │   ├── router_agent.py         # 意图识别（关键词规则 + LLM 兜底）
│   │   └── customer_service.py     # LangGraph 多Agent 入口 + ToolAgent
│   │       ├── ToolAgent           #   LLM + Function Calling 基类
│   │       ├── MenuToolAgent       #   菜品问答工具Agent（9个工具）
│   │       ├── RecommendToolAgent  #   推荐工具Agent
│   │       ├── BookingToolAgent    #   订座工具Agent
│   │       ├── ComplaintToolAgent  #   投诉工具Agent
│   │       ├── MenuQAAgent         #   菜品问答（快速路径 + ToolAgent兜底）
│   │       ├── RecommendAgent      #   推荐（预算/过敏/人群 + ToolAgent兜底）
│   │       ├── ReservationAgent    #   订座
│   │       ├── ComplaintAgent      #   投诉
│   │       └── HumanTransferAgent  #   转人工
│   ├── tools/
│   │   ├── agent_tools.py          # 9个 LangChain @tool 函数（Function Calling）
│   │   ├── unified_kb.py           # 统一知识库（单一事实源）
│   │   ├── faq_engine.py           # FAQ 引擎（50+ 结构化问答）
│   │   ├── vector_store.py         # ChromaDB 向量数据库 + RAG 检索
│   │   └── document_loader.py      # 文档加载器
│   ├── data/knowledge_base/
│   │   ├── menu.json               # 菜品数据（唯一事实源）
│   │   ├── restaurant_faq.json     # 结构化FAQ知识库
│   │   └── restaurant_faq.md       # 餐厅FAQ文档
│   ├── frontend/index.html         # 前端聊天页面
│   ├── api/main.py                 # FastAPI 后端入口
│   └── config.py                   # 配置
├── tests/test_rag.py               # 验收测试
├── preflight.py                    # 40 项演示前检查
├── cli_chat.py                     # CLI 交互式聊天
├── start.py                        # 一键启动脚本
├── requirements.txt
└── README.md
```

---

## 🏗️ 架构

### LangGraph 多Agent 工作流

```
用户输入
  │
  ▼
RouterAgent（关键词规则优先 → LLM兜底分类）
  │  ├─ 菜品名称优先匹配
  │  ├─ 菜系关键词匹配
  │  └─ 意图优先级循环（转人工 > 投诉 > 订单 > 推荐 > 订座 > 点餐 > 闲聊）
  │
  ▼
LangGraph StateGraph 路由（8个节点）
  ├─ MenuQAAgent      ← 价格查询 / 点餐咨询
  │    ├─ FAQ引擎（50+ 结构化问答）
  │    ├─ 确定性快速路径（50+ 条规则）
  │    └─ ToolAgent（LLM + Function Calling 兜底）
  ├─ RecommendAgent   ← 菜品推荐
  │    ├─ 预算算法 + 过敏原过滤 + 人群标签
  │    ├─ 多轮承接（换一个/还有/改成）
  │    └─ ToolAgent 兜底
  ├─ ReservationAgent ← 订座服务
  ├─ ComplaintAgent   ← 投诉建议（8类快速响应）
  ├─ HumanTransferAgent ← 转人工（400-888-6666）
  ├─ OrderAgent       ← 订单查询
  └─ Fallback         ← FAQ + RAG + LLM 兜底
```

### 三层处理

```
第一层：确定性查询（不调用 LLM）
  菜单 / 价格 / 营业时间 / 地址 / WiFi / 优惠 / FAQ

第二层：结构化推荐（不调用 LLM）
  人群 / 过敏原 / 预算套餐 / 场景 / 口味 / 多轮承接

第三层：LLM + Function Calling 兜底
  复杂问题 / 工具调用 / 闲聊 / 菜系外询问
```

### ToolAgent 工具调用（Function Calling）

```
ToolAgent 基类
  │  LLM.bind_tools() → 自主决定调用哪些工具
  │  循环：LLM推理 → 选择工具 → 执行 → 喂回结果 → 重复（最多5轮）
  │
  ├─ query_dish         查菜品详情
  ├─ search_dishes      按条件搜索菜品
  ├─ get_full_menu      获取完整菜单
  ├─ check_allergen     检查过敏原
  ├─ recommend_combo    预算套餐推荐
  ├─ recommend_for_people 人群推荐
  ├─ query_faq          FAQ查询
  ├─ get_promotions     优惠活动
  └─ get_restaurant_info 餐厅信息
```

### 多轮对话

```
用户：3个人150元预算，清淡一点
饭小二：推荐四人套餐 328元...

用户：换个不要辣的          ← 继承 3人150元
饭小二：换个口味：清炒时蔬 28元...

用户：还有其他的吗           ← 排除已推荐
饭小二：再来几道：招牌红烧肉 58元...
```

---

## 🎯 意图分类

| 意图 | 负责 Agent | 示例 |
|------|-----------|------|
| 价格查询 | MenuQAAgent | 红烧肉多少钱？人均消费？ |
| 菜品推荐 | RecommendAgent | 有什么推荐？3个人150元 |
| 点餐咨询 | MenuQAAgent | 全部菜单？辣吗？不要香菜？ |
| 订座服务 | ReservationAgent | 我想订座，有包间吗？ |
| 投诉建议 | ComplaintAgent | 菜太咸了，服务态度差 |
| 订单查询 | OrderAgent | 我的菜好了吗？ |
| 转人工 | HumanTransferAgent | 转人工、找真人客服 |
| 闲聊互动 | Fallback（FAQ+RAG+LLM） | 你好，今天心情不好 |

## 🔧 API 接口

### POST /chat

```json
// 请求
{ "message": "红烧肉多少钱？", "session_id": "user_123" }

// 响应
{ "reply": "招牌红烧肉 - 58元/份...", "success": true, "error": null }
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| message | string | 是 | 用户消息（1-500字） |
| session_id | string | 否 | 会话ID，不同 ID 状态隔离 |

### GET /health → 健康检查
### GET / → 前端聊天页面

---

## 📝 开发进度

- [x] 结构化知识库（menu.json 单一事实源）
- [x] 意图识别（关键词规则优先 + LLM 兜底）
- [x] 三层处理架构（确定性 → 结构化 → LLM+Tool Calling）
- [x] **LangGraph 多Agent 编排**（8节点状态图）
- [x] **ToolAgent + Function Calling**（9个工具函数，LLM自主决策）
- [x] **RAG 向量检索增强**（ChromaDB + text-embedding-v3）
- [x] **FAQ 引擎**（50+ 结构化问答，加权关键词匹配）
- [x] 多轮对话状态管理 + 对话记忆（最近10轮）
- [x] 约束识别（人群/过敏/预算/场景/口味）
- [x] 转人工功能（400-888-6666）
- [x] 前端聊天界面
- [x] CLI 交互式聊天
- [x] **40 项验收测试 100% 通过**
- [ ] 语音点餐支持
- [ ] 会员个性化推荐
- [ ] 数据分析看板

## 📄 许可证

MIT License

---

**饭小二语录**：客官，您来啦！今天想吃点啥？🍚✨
