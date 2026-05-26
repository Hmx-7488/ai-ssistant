# 每日任务清单

---

## Day 1（5月27日）：搭建 RAG Agent + 知识库

### 上午（3小时）
- [ ] 创建项目目录结构
- [ ] 初始化 requirements.txt
- [ ] 创建 .env 配置文件
- [ ] 准备知识库数据（产品FAQ、使用手册）

### 下午（3小时）
- [ ] 实现文档加载器（支持 txt/md/pdf）
- [ ] 实现文本分割器
- [ ] 实现 ChromaDB 向量存储
- [ ] 实现 RAG Agent（检索 + 回答）

### 晚上（2小时）
- [ ] 测试 RAG 效果
- [ ] 调整 Prompt 优化回答质量
- [ ] 提交代码到 Git

### 今日产出
```
src/agents/rag_agent.py       # RAG Agent
src/data/knowledge_base/      # 知识库文档
src/data/build_knowledge_base.py  # 构建脚本
```

---

## Day 2（5月28日）：搭建 SQL Agent + 订单数据库

### 上午（3小时）
- [ ] 设计订单数据库表结构
- [ ] 创建 SQLite 数据库
- [ ] 插入测试数据（订单、用户、商品）
- [ ] 实现数据库初始化脚本

### 下午（3小时）
- [ ] 实现 SQL Agent
- [ ] 测试自然语言转 SQL
- [ ] 处理复杂查询（多表关联、聚合）

### 晚上（2小时）
- [ ] 测试 SQL Agent 效果
- [ ] 优化 Prompt（减少幻觉）
- [ ] 提交代码到 Git

### 今日产出
```
src/agents/sql_agent.py       # SQL Agent
src/data/sample.db            # 订单数据库
src/data/init_db.py           # 初始化脚本
```

---

## Day 3（5月29日）：多 Agent 路由 + 多轮记忆

### 上午（3小时）
- [ ] 实现 Router Agent（问题分类）
- [ ] 实现 Agent 之间的调用关系
- [ ] 测试路由准确性

### 下午（3小时）
- [ ] 实现多轮对话记忆
- [ ] 处理指代消解（"刚才那个订单"）
- [ ] 集成测试完整流程

### 晚上（2小时）
- [ ] 端到端测试
- [ ] 修复问题
- [ ] 提交代码到 Git

### 今日产出
```
src/agents/router_agent.py    # 路由 Agent
src/agents/customer_service.py  # 主 Agent（整合所有子 Agent）
```

---

## Day 4（5月30日）：FastAPI 部署 + 接口测试

### 上午（3小时）
- [ ] 创建 FastAPI 应用
- [ ] 实现 /chat 接口
- [ ] 实现 /health 健康检查
- [ ] 添加错误处理

### 下午（3小时）
- [ ] 实现对话历史管理
- [ ] 添加日志记录
- [ ] 接口测试（Postman/curl）

### 晚上（2小时）
- [ ] 性能测试
- [ ] 优化响应速度
- [ ] 提交代码到 Git

### 今日产出
```
src/api/main.py               # FastAPI 应用
src/api/routes.py             # 路由定义
src/api/schemas.py            # 数据模型
```

---

## Day 5（5月31日）：文档 + GitHub 上传

### 上午（3小时）
- [ ] 完善 README.md
- [ ] 编写 API 文档
- [ ] 录制演示视频（可选）

### 下午（3小时）
- [ ] 整理代码结构
- [ ] 添加注释
- [ ] 上传 GitHub

### 晚上（2小时）
- [ ] 最终测试
- [ ] 修复 Bug
- [ ] 准备项目讲解要点

### 今日产出
```
GitHub 仓库（完整项目）
README.md（详细说明）
演示视频（可选）
```

---

## Day 6-7（6月2-3日）：加分功能（可选）

### 功能清单
- [ ] Streamlit 演示界面
- [ ] 人工转接功能
- [ ] 对话历史持久化
- [ ] 单元测试

---

## Day 8-9（6月4-5日）：面试准备

### 准备内容
- [ ] 项目架构讲解（2分钟版本）
- [ ] 技术难点和解决方案
- [ ] 常见面试问题准备
- [ ] 模拟面试练习

---

## Day 10（6月6日）：投递简历

### 行动清单
- [ ] 更新简历
- [ ] 整理 GitHub 项目
- [ ] 投递目标公司
- [ ] 准备面试问题
