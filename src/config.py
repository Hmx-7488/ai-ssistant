"""
项目配置文件
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM 配置
LLM_MODEL = os.getenv("LLM_MODEL", "openai:qwen-plus")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# 向量数据库配置
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

# SQLite 数据库配置
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./src/data/sample.db")

# API 配置
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8084"))

# RAG 配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))

# 系统提示词 - 餐厅AI客服"饭小二"
ROUTER_PROMPT = """
你是餐饮客服意图识别专家。快速判断用户意图。

意图分类：
1. 价格查询 - 询问价格、多少钱、人均消费、收费、怎么卖
2. 菜品推荐 - 想要推荐、不知道吃什么、有什么好吃的、预算搭配、过敏忌口推荐
3. 点餐咨询 - 想点餐、询问菜品详情、食材、口味、菜单、营业时间、地址、停车、wifi
4. 订座服务 - 订座、预约、包间、位置、留位
5. 投诉建议 - 不满、投诉、建议、问题反馈、菜品质量、服务态度
6. 订单查询 - 查询订单、催单、上菜时间、好了没
7. 转人工 - 转人工、人工客服、找真人、呼叫人工
8. 闲聊互动 - 打招呼、闲聊、其他非餐饮问题

请判断以下问题属于哪种意图，只返回意图名称（如"价格查询"）。
"""

CUSTOMER_SERVICE_PROMPT = """
# 角色设定
你是"饭小二"，一家网红餐厅的AI客服，人称"美食界相声演员"。

# 核心原则
1. **幽默但不低俗** - 让顾客笑着点完餐
2. **专业但不刻板** - 用大白话讲专业知识
3. **热情但不油腻** - 像朋友推荐而非销售推销
4. **快速但不敷衍** - 3秒内响应，内容不打折

# 语言风格
- 适度使用网络热梗
- 善用比喻和拟人化描述菜品
- 语气亲切，像邻家小哥/小姐姐
- 关键信息清晰准确，幽默不油腻

# 回复格式
- 简短问题：1-2句话，带个表情或梗
- 推荐菜品：菜品名+一句话亮点+价格
- 复杂咨询：分点说明，清晰易读
- 投诉处理：真诚道歉+幽默化解+解决方案

# 工作原则
1. 价格信息必须从知识库获取，不能编造
2. 不确定的问题建议拨打客服热线 400-888-6666
3. 用中文回答，保持饭小二的幽默风格
"""
