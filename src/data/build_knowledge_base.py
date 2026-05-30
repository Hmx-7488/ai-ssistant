"""
构建知识库脚本

作用：
1. 加载知识库目录下的所有文档
2. 分割成小块
3. 存入 ChromaDB 向量数据库

使用方法：
    python src/data/build_knowledge_base.py
"""

import os
import sys
import shutil

# 把项目根目录加到 Python 搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from src.config import CHROMA_PERSIST_DIR
from src.tools.document_loader import DocumentProcessor
from src.tools.vector_store import VectorStore


def build_knowledge_base(reset: bool = True):
    """
    构建知识库

    Args:
        reset: 是否重置向量数据库（清空旧数据重新构建）
    """
    print("=" * 60)
    print("开始构建餐厅知识库")
    print("=" * 60)

    # 1. 获取知识库目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_base_dir = os.path.join(current_dir, "knowledge_base")

    print(f"\n知识库目录: {knowledge_base_dir}")

    # 检查目录是否存在
    if not os.path.exists(knowledge_base_dir):
        print(f"[ERROR] 知识库目录不存在: {knowledge_base_dir}")
        return

    # 列出目录中的文件
    files = os.listdir(knowledge_base_dir)
    print(f"找到文件: {files}")

    # 2. 加载并分割文档
    print("\n" + "-" * 60)
    print("步骤 1: 加载并分割文档")
    print("-" * 60)

    processor = DocumentProcessor()
    docs = processor.process_directory(knowledge_base_dir)

    if not docs:
        print("[ERROR] 没有加载到任何文档！")
        return

    print(f"\n共加载 {len(docs)} 个文本块")

    # 3. 如果需要重置，先删除旧的向量数据库
    if reset and os.path.exists(CHROMA_PERSIST_DIR):
        print(f"\n正在删除旧的向量数据库: {CHROMA_PERSIST_DIR}")
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("[OK] 已删除")

    # 4. 存入向量数据库
    print("\n" + "-" * 60)
    print("步骤 2: 存入向量数据库")
    print("-" * 60)

    vector_store = VectorStore()
    vector_store.add_documents(docs)

    # 5. 验证
    print("\n" + "-" * 60)
    print("步骤 3: 验证")
    print("-" * 60)

    count = vector_store.get_document_count()
    print(f"向量数据库中文档数量: {count}")

    # 6. 测试搜索
    print("\n" + "-" * 60)
    print("步骤 4: 测试搜索")
    print("-" * 60)

    test_queries = [
        "红烧肉多少钱？",
        "有什么推荐的菜？",
        "营业到几点？",
    ]

    for query in test_queries:
        print(f"\n问题: {query}")
        print("-" * 40)
        results = vector_store.search(query, top_k=2)
        for i, doc in enumerate(results):
            print(f"  结果 {i+1}: {doc.page_content[:80]}...")

    print("\n" + "=" * 60)
    print("知识库构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    build_knowledge_base(reset=True)
