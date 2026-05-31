"""
饭小二启动脚本

使用方法：
    python start.py
"""

import os
import sys
import subprocess


def external_ai_enabled():
    """是否启用需要外部 API/向量库的增强能力。"""
    return (
        os.getenv("ENABLE_VECTOR_RAG", "0") == "1"
        or os.getenv("ENABLE_LLM_ROUTER", "0") == "1"
        or os.getenv("ENABLE_LLM_CHAT", "0") == "1"
    )


def main():
    print("=" * 60)
    print("饭小二餐饮AI客服系统启动器")
    print("=" * 60)
    
    # 检查虚拟环境
    venv_path = ".venv"
    if not os.path.exists(venv_path):
        print("[ERROR] 虚拟环境不存在，请先运行：")
        print("  python -m venv .venv")
        print(r"  .venv\Scripts\Activate.ps1")
        print("  pip install -r requirements.txt")
        return
    
    # 默认使用本地规则和结构化知识库，可无 API Key 演示。
    # 只有启用外部模型或向量 RAG 时才强制要求 .env。
    if external_ai_enabled() and not os.path.exists(".env"):
        print("[ERROR] .env 文件不存在，请先配置环境变量")
        print("参考 README.md 中的配置说明")
        return
    
    # 检查 chroma_db 是否存在
    if os.getenv("ENABLE_VECTOR_RAG", "0") == "1" and not os.path.exists("chroma_db"):
        print("[INFO] 知识库不存在，正在构建...")
        try:
            subprocess.run([sys.executable, "src/data/build_knowledge_base.py"], check=True)
            print("[OK] 知识库构建完成")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 知识库构建失败: {e}")
            return
    
    print("\n[INFO] 正在启动饭小二...")
    print("[INFO] 服务启动后访问: http://localhost:8085")
    print("[INFO] API 文档: http://localhost:8085/docs")
    print("[INFO] 按 Ctrl+C 停止服务\n")
    
    # 启动 FastAPI 服务
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8085"
        ], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] 饭小二已停止服务，下次再见！")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 启动失败: {e}")

if __name__ == "__main__":
    main()
