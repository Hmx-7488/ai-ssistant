"""
FastAPI 应用 - 餐厅AI客服API

提供 /chat 接口和前端页面
"""

import os
import sys

# 设置标准输出编码为UTF-8，解决Windows下的编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from src.agents.customer_service import CustomerServiceAgent
from src.config import API_HOST, API_PORT

# 创建 FastAPI 应用
app = FastAPI(
    title="饭小二餐饮AI客服",
    description="基于 LangChain + RAG 的餐厅智能客服系统",
    version="1.0.0",
)

# 添加 CORS 中间件（允许前端跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局客服代理（启动时初始化）
agent: Optional[CustomerServiceAgent] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(default="default")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "红烧肉多少钱？",
                "session_id": "default"
            }
        }


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str
    success: bool = True
    error: Optional[str] = None


@app.on_event("startup")
async def startup():
    """启动时初始化客服代理 + 构建 RAG 知识库"""
    global agent
    print("=" * 60)
    print("正在初始化饭小二客服系统...")
    print("=" * 60)

    # 构建 RAG 向量知识库
    try:
        from src.tools.vector_store import build_knowledge_base
        build_knowledge_base()
    except Exception as e:
        print(f"[WARN] RAG 知识库构建失败（不影响基本功能）: {e}")

    agent = CustomerServiceAgent()
    print("\n饭小二准备就绪！")


@app.get("/")
async def root():
    """首页 - 返回前端页面"""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": "欢迎使用饭小二餐饮AI客服！", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口

    Args:
        request: 聊天请求，包含用户消息

    Returns:
        聊天响应，包含饭小二的回复
    """
    try:
        if agent is None:
            return ChatResponse(
                reply="饭小二还在系围裙，请稍后再试~",
                success=False,
                error="agent_not_ready",
            )
        # 临时重定向 stdout 来避免编码错误
        import io
        old_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
        try:
            reply = agent.invoke(request.message, sid=request.session_id)
        finally:
            sys.stdout = old_stdout
        return ChatResponse(reply=reply, success=True)
    except Exception as e:
        import traceback
        try:
            error_msg = traceback.format_exc()
            print(f"[ERROR] {error_msg}")
        except:
            pass
        return ChatResponse(
            reply="抱歉，饭小二暂时开小差了，请稍后再试~",
            success=False,
            error=str(e),
        )


@app.get("/health")
async def health():
    """
    健康检查 - 返回系统各组件状态。
    演示前可 GET /health 快速确认一切就绪。
    """
    import os
    checks = {}

    # 1. Agent 是否就绪
    checks["agent_ready"] = agent is not None

    # 2. 知识库文件是否存在
    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "knowledge_base")
    menu_json = os.path.join(kb_dir, "menu.json")
    faq_md = os.path.join(kb_dir, "restaurant_faq.md")
    checks["menu_json_exists"] = os.path.exists(menu_json)
    checks["faq_md_exists"] = os.path.exists(faq_md)

    # 3. 向量库是否可用
    try:
        from src.tools.vector_store import get_vector_store
        vs = get_vector_store()
        doc_count = vs.get_document_count()
        checks["vector_store_docs"] = doc_count
        checks["vector_store_ok"] = doc_count > 0
    except Exception as e:
        checks["vector_store_ok"] = False
        checks["vector_store_error"] = str(e)

    # 4. 最小问答测试
    if agent is not None:
        try:
            reply = agent.invoke("红烧肉多少钱", sid="health_check")
            checks["min_qa_ok"] = "58" in reply
            checks["min_qa_reply"] = reply[:60]
        except Exception as e:
            checks["min_qa_ok"] = False
            checks["min_qa_error"] = str(e)

    # 汇总
    all_ok = (
        checks.get("agent_ready")
        and checks.get("menu_json_exists")
        and checks.get("faq_md_exists")
        and checks.get("vector_store_ok", False)
        and checks.get("min_qa_ok", False)
    )
    checks["status"] = "healthy" if all_ok else "degraded"
    return checks


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
