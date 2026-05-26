# 🤖 基于 LangGraph 的多 Agent 智能客服平台

> AI Agent 驱动的企业级客服助手，支持知识库问答、订单查询、多轮对话

## 📌 项目简介

本项目基于 LangChain + LangGraph 构建了一个多 Agent 协作的智能客服系统。系统能够自动识别用户问题类型，将问题分发给不同的专业 Agent 处理，实现知识库问答、订单查询、闲聊等多种功能。

## ✨ 核心功能

- **知识库问答（RAG）**：基于产品文档自动回答用户问题
- **订单查询（SQL Agent）**：自然语言查询订单状态、物流信息
- **多 Agent 路由**：智能识别问题类型，分发给对应 Agent
- **多轮对话记忆**：记住上下文，支持指代消解
- **人工转接**：复杂问题自动转接人工客服

## 🏗️ 技术架构

```
用户提问
   │
   ▼
┌──────────────┐
│  主 Agent    │  ← 判断问题类型，分发给子 Agent
│  (Router)    │
└──────┬───────┘
       │
       ├──→ 知识库问题 ──→ RAG Agent ──→ ChromaDB ──→ 回答
       │
       ├──→ 订单问题 ──→ SQL Agent ──→ SQLite ──→ 回答
       │
       └──→ 复杂问题 ──→ 转人工
```

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Agent 框架 | LangChain + LangGraph | 构建多 Agent 系统 |
| 向量数据库 | ChromaDB | 存储文档 Embedding |
| 关系数据库 | SQLite | 存储订单数据 |
| LLM | 通义千问 (qwen-plus) | 大语言模型 |
| API 服务 | FastAPI | 部署 Agent 接口 |
| 前端（可选） | Streamlit | 演示界面 |

## 📁 项目结构

```
ai-assistant/
├── docs/                    # 开发文档
│   ├── development-plan.md  # 开发计划
│   ├── daily-tasks.md       # 每日任务
│   └── technical-design.md  # 技术设计
├── src/                     # 源代码
│   ├── agents/              # Agent 定义
│   │   ├── router_agent.py  # 路由 Agent
│   │   ├── rag_agent.py     # RAG Agent
│   │   └── sql_agent.py     # SQL Agent
│   ├── tools/               # 工具定义
│   ├── data/                # 数据文件
│   │   ├── knowledge_base/  # 知识库文档
│   │   └── sample.db        # 示例数据库
│   ├── api/                 # API 接口
│   └── config.py            # 配置文件
├── tests/                   # 测试
├── requirements.txt         # 依赖
├── .env                     # 环境变量
└── README.md                # 项目说明
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 3. 初始化数据库
python src/data/init_db.py

# 4. 构建知识库
python src/data/build_knowledge_base.py

# 5. 启动服务
python src/api/main.py
```

## 📝 开发日志

- [2026-05-26] 项目启动，完成架构设计
- [2026-05-27] Day 1：搭建 RAG Agent
- [2026-05-28] Day 2：搭建 SQL Agent
- [2026-05-29] Day 3：多 Agent 路由 + 记忆
- [2026-05-30] Day 4：FastAPI 部署
- [2026-05-31] Day 5：文档 + GitHub 上传

## 📄 License

MIT License
