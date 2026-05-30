"""
向量数据库工具

作用：
1. 把文本块变成向量（Embedding）
2. 存入 ChromaDB
3. 提供相似度搜索功能
"""

import os
import sys
import json

# 把项目根目录加到 Python 搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K,
)


class VectorStore:
    """向量数据库：存储和搜索文档"""

    @staticmethod
    def _safe_print(msg: str):
        """安全打印，避免编码错误"""
        try:
            print(msg)
        except UnicodeEncodeError:
            try:
                print(msg.encode('ascii', 'replace').decode('ascii'))
            except:
                pass

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        """
        初始化向量数据库

        Args:
            persist_dir: ChromaDB 数据存储目录
            collection_name: 集合名称（类似数据库的表名）
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # 初始化 Embedding 模型
        # 使用通义千问的 text-embedding-v3 模型
        # 它会把文字变成 1024 维的向量
        # check_embedding_ctx_length=False 解决中文编码问题
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-v3",
            check_embedding_ctx_length=False,
        )

        # 初始化 ChromaDB
        # persist_directory 指定数据存储位置
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        self._safe_print(f"[VectorStore] 初始化完成")
        self._safe_print(f"  存储目录: {persist_dir}")
        self._safe_print(f"  集合名称: {collection_name}")

    def add_documents(self, documents: List[Document]) -> None:
        """
        把文档存入向量数据库（分批添加，避免 Embedding API 批次大小限制）

        Args:
            documents: Document 列表
        """
        if not documents:
            self._safe_print("[VectorStore] 没有文档需要添加")
            return

        batch_size = 10
        total = len(documents)
        self._safe_print(f"[VectorStore] 正在添加 {total} 个文档（每批 {batch_size}）...")

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
            self._safe_print(f"  已添加 {min(i + batch_size, total)}/{total}")

        self._safe_print(f"[VectorStore] 添加完成！")

    def search(self, query: str, top_k: int = TOP_K) -> List[Document]:
        """
        搜索最相关的文档

        过程：
        1. 把用户问题变成向量
        2. 在 ChromaDB 里找最相似的 top_k 个文档
        3. 返回结果

        Args:
            query: 用户的问题
            top_k: 返回几个最相关的文档

        Returns:
            最相关的文档列表
        """
        self._safe_print(f"[VectorStore] 搜索: {query}")
        self._safe_print(f"  返回 top {top_k} 个结果")

        # similarity_search 会：
        # 1. 把 query 变成向量
        # 2. 计算与所有文档的相似度
        # 3. 返回最相似的 top_k 个
        results = self.vector_store.similarity_search(query, k=top_k)

        self._safe_print(f"  找到 {len(results)} 个相关文档")
        return results

    def get_retriever(self, top_k: int = TOP_K):
        """
        返回一个 retriever 对象

        retriever 是 LangChain 的标准接口，很多组件都能直接用
        比如 RetrievalQA 链就需要传入 retriever

        Args:
            top_k: 返回几个最相关的文档

        Returns:
            retriever 对象
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

    def get_document_count(self) -> int:
        """返回数据库中的文档数量"""
        return self.vector_store._collection.count()

    def build_from_menu_json(self, json_path: str) -> int:
        """
        从 menu.json 构建向量知识库，把结构化数据转为可检索的文档。

        Args:
            json_path: menu.json 的路径

        Returns:
            添加的文档数量
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        # 菜品 → 每道菜一个文档
        for item in data.get("菜品", []):
            content = f"菜品：{item['name']}\n"
            content += f"价格：{item['price']}元\n"
            content += f"分类：{item['category']}\n"
            content += f"描述：{item['description']}\n"
            content += f"辣度：{item['spice']}\n"
            content += f"分量：{item['portion']}\n"
            if item.get('tags'):
                content += f"标签：{', '.join(item['tags'])}\n"
            if item.get('allergens'):
                content += f"过敏原：{', '.join(item['allergens'])}\n"
            if item.get('customizable'):
                content += f"可定制：是"
                if item.get('customizable_items'):
                    content += f"（{', '.join(item['customizable_items'])}）"
                content += "\n"
            if item.get('pairing'):
                content += f"推荐搭配：{', '.join(item['pairing'])}\n"
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": "menu.json", "type": "菜品", "name": item['name']}
            ))

        # 套餐
        for suite in data.get("套餐", []):
            content = f"套餐：{suite['name']}\n"
            content += f"价格：{suite['price']}元（原价{suite['original_price']}元）\n"
            content += f"包含：{', '.join(suite['items'])}\n"
            content += f"适合：{', '.join(suite['suitable'])}\n"
            content += f"说明：{suite['description']}\n"
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": "menu.json", "type": "套餐", "name": suite['name']}
            ))

        # 人群标签
        for people, info in data.get("人群标签", {}).items():
            content = f"人群：{people}\n"
            content += f"推荐菜品：{', '.join(info.get('推荐', []))}\n"
            if info.get('避免'):
                content += f"避免菜品：{', '.join(info['避免'])}\n"
            content += f"说明：{info.get('说明', '')}\n"
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": "menu.json", "type": "人群标签", "name": people}
            ))

        # 忌口规则
        for rule, info in data.get("忌口规则", {}).items():
            content = f"忌口：{rule}\n"
            if info.get('避免菜品'):
                content += f"避免菜品：{', '.join(info['避免菜品'])}\n"
            if info.get('推荐'):
                content += f"推荐菜品：{', '.join(info['推荐'])}\n"
            content += f"说明：{info.get('说明', '')}\n"
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": "menu.json", "type": "忌口规则", "name": rule}
            ))

        # 场景推荐
        for scene, info in data.get("场景推荐", {}).items():
            content = f"场景：{scene}\n"
            content += f"推荐：{', '.join(info.get('推荐', []))}\n"
            content += f"说明：{info.get('说明', '')}\n"
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": "menu.json", "type": "场景推荐", "name": scene}
            ))

        # 餐厅信息
        rest = data.get("餐厅信息", {})
        if rest:
            content = "餐厅信息\n"
            for k, v in rest.items():
                if isinstance(v, dict):
                    content += f"{k}：\n"
                    for kk, vv in v.items():
                        content += f"  {kk}：{vv}\n"
                else:
                    content += f"{k}：{v}\n"
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": "menu.json", "type": "餐厅信息"}
            ))

        # 优惠活动
        promos = data.get("优惠活动", {})
        if promos:
            content = "优惠活动\n"
            for k, v in promos.items():
                content += f"{k}：{v}\n"
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": "menu.json", "type": "优惠活动"}
            ))

        self._safe_print(f"[VectorStore] 从 menu.json 生成 {len(documents)} 个文档")
        self.add_documents(documents)
        return len(documents)

    def build_from_md_files(self, knowledge_base_dir: str) -> int:
        """
        从 .md 文件构建向量知识库。

        Args:
            knowledge_base_dir: 知识库目录路径

        Returns:
            添加的文档数量
        """
        from src.tools.document_loader import DocumentProcessor
        processor = DocumentProcessor()
        docs = processor.process_directory(knowledge_base_dir)
        if docs:
            self._safe_print(f"[VectorStore] 从 MD 文件生成 {len(docs)} 个文档")
            self.add_documents(docs)
        return len(docs)


_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """单例获取 VectorStore"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


def build_knowledge_base() -> int:
    """构建向量知识库（menu.json + .md 文件）"""
    vs = get_vector_store()
    total = 0

    menu_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "knowledge_base", "menu.json"
    )
    if os.path.exists(menu_json):
        total += vs.build_from_menu_json(menu_json)

    kb_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "knowledge_base"
    )
    if os.path.exists(kb_dir):
        total += vs.build_from_md_files(kb_dir)

    print(f"[RAG] 知识库构建完成，共 {total} 个文档")
    return total


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    RAG 检索：根据用户问题从向量库中召回相关上下文。

    Args:
        query: 用户问题
        top_k: 返回最相关的 top_k 个文档

    Returns:
        拼接好的上下文文本，可直接拼入 LLM prompt
    """
    vs = get_vector_store()
    docs = vs.search(query, top_k=top_k)
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[参考{i}] {doc.page_content}")
    return "\n\n".join(parts)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 RAG 知识库构建与检索")
    print("=" * 60)

    build_knowledge_base()

    print(f"\n文档总数: {get_vector_store().get_document_count()}")

    print("\n" + "=" * 60)
    print("测试检索")
    print("=" * 60)

    queries = [
        "红烧肉多少钱",
        "老人吃什么",
        "花生过敏",
        "营业时间",
        "订座",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        ctx = retrieve_context(q)
        print(f"A:\n{ctx[:300]}")
