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
API_PORT = int(os.getenv("API_PORT", "8000"))

# RAG 配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))

# 系统提示词
ROUTER_PROMPT = """
你是一个智能客服路由器，负责判断用户问题的类型。

问题类型：
1. knowledge_base - 产品功能、使用方法、常见问题
2. order_query - 订单状态、物流信息、退款查询
3. chitchat - 问候、感谢、闲聊
4. human_handoff - 投诉、复杂问题、需要人工处理

请判断以下问题属于哪种类型，只返回类型名称。
"""

CUSTOMER_SERVICE_PROMPT = """
你是一个专业的客服助手，可以帮助用户解决各种问题。

你的能力：
1. 使用知识库回答产品相关问题
2. 查询订单和物流信息
3. 处理一般性咨询

工作原则：
1. 礼貌、耐心、专业
2. 不知道的问题诚实说不知道
3. 复杂问题建议转人工
4. 用中文回答
"""
