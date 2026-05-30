# 🍚 饭小二 - 餐饮AI客服助手

基于 **LangChain + LangGraph + RAG** 的餐厅智能客服系统，人称"美食界相声演员"。

## ✨ 功能特点

- **LangGraph 多Agent**：6 个专用 Agent 各司其职，状态图编排路由
- **RAG 增强**：向量检索 + LLM 生成，知识库自动构建
- **三层架构**：确定性查询 → 结构化推荐 → LLM+RAG 兜底，高频问题秒回
- **多轮对话**：上下文继承，"换个清淡的""还有别的吗""改成3个人"都能接住
- **约束识别**：人数、预算、人群、过敏原、口味偏好、场景
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
# 运行 12 项验收测试
python tests/test_rag.py

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
│   │   ├── router_agent.py         # 意图识别（关键词规则 + 可选 LLM）
│   │   └── customer_service.py     # LangGraph 多Agent 入口
│   │       ├── MenuQAAgent         #   菜品问答（价格/菜单/详情）
│   │       ├── RecommendAgent      #   推荐（人群/过敏/预算/场景）
│   │       ├── ReservationAgent    #   订座
│   │       ├── ComplaintAgent      #   投诉
│   │       └── OrderAgent          #   订单
│   ├── tools/
│   │   ├── unified_kb.py           # 统一知识库（单一事实源）
│   │   ├── vector_store.py         # 向量数据库 + RAG 检索
│   │   └── document_loader.py      # 文档加载器
│   ├── data/knowledge_base/
│   │   ├── menu.json               # 菜品数据（唯一事实源）
│   │   └── restaurant_faq.md       # 餐厅FAQ
│   ├── frontend/index.html         # 前端聊天页面
│   ├── api/main.py                 # FastAPI 后端入口
│   └── config.py                   # 配置
├── tests/test_rag.py               # 12 项验收测试
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
RouterAgent（意图分类 + 多轮承接检测）
  │
  ▼
LangGraph StateGraph 路由
  ├─ MenuQAAgent      ← 价格查询 / 点餐咨询
  │    └─ 确定性快速路径 + RAG 兜底
  ├─ RecommendAgent   ← 菜品推荐
  │    └─ 人群/过敏/预算/场景/口味 + 多轮承接
  ├─ ReservationAgent ← 订座服务
  ├─ ComplaintAgent   ← 投诉建议
  ├─ OrderAgent       ← 订单查询
  └─ LLM + RAG 兜底   ← 闲聊 / 无法归类
```

### 三层处理

```
第一层：确定性查询（秒回，不调用 LLM）
  菜单 / 价格 / 营业时间 / 地址 / WiFi / 优惠

第二层：结构化推荐（秒回，不调用 LLM）
  人群 / 过敏原 / 预算套餐 / 场景 / 口味

第三层：LLM + RAG 兜底
  复杂问题 / 闲聊 / 菜系外询问
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
| 闲聊互动 | LLM+RAG 兜底 | 你好，今天心情不好 |

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
- [x] 意图识别（关键词规则 + 可选 LLM）
- [x] 三层处理架构
- [x] **LangGraph 多Agent 编排**
- [x] **RAG 向量检索增强**
- [x] 多轮对话状态管理
- [x] 约束识别（人群/过敏/预算/场景/口味）
- [x] 前端聊天界面
- [x] CLI 交互式聊天
- [x] 12 项验收测试 100% 通过
- [ ] 语音点餐支持
- [ ] 会员个性化推荐
- [ ] 数据分析看板

## 📄 许可证

MIT License

---

**饭小二语录**：客官，您来啦！今天想吃点啥？🍚✨
