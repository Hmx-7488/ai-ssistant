"""
文档加载和分割工具

作用：
1. 读取知识库目录下的所有文档（.md / .txt / .pdf）
2. 将文档分割成小块（chunk），方便后续向量化
"""

import os
import sys
import warnings

# 把项目根目录加到 Python 搜索路径，解决直接运行时找不到 src 包的问题
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# 忽略 langchain-community 的弃用警告（目前仍可正常使用）
warnings.filterwarnings("ignore", message=".*langchain-community.*")

from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """文档处理器：加载和分割文档"""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """
        初始化文档处理器

        Args:
            chunk_size: 每个文本块的最大字符数（默认 500）
            chunk_overlap: 相邻文本块的重叠字符数（默认 50）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 创建文本分割器
        # RecursiveCharacterTextSplitter 会按照段落 -> 句子 -> 词 的顺序尝试分割
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",   # 段落分隔
                "\n",     # 换行
                "。",     # 中文句号
                "！",     # 中文感叹号
                "？",     # 中文问号
                ".",      # 英文句号
                "!",      # 英文感叹号
                "?",      # 英文问号
                " ",      # 空格
                "",       # 字符级别（兜底）
            ],
        )

    def load_documents(self, directory: str) -> List[Document]:
        """
        加载目录中的所有文档

        Args:
            directory: 文档目录路径

        Returns:
            Document 列表，每个 Document 包含 page_content 和 metadata
        """
        documents = []

        # 1. 加载 .md 文件
        try:
            md_loader = DirectoryLoader(
                directory,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader,
                show_progress=True,
                silent_errors=True,
            )
            md_docs = md_loader.load()
            documents.extend(md_docs)
            print(f"  [OK] 加载了 {len(md_docs)} 个 Markdown 文件")
        except Exception as e:
            print(f"  [WARN] 加载 Markdown 文件时出错: {e}")

        # 2. 加载 .txt 文件
        try:
            txt_loader = DirectoryLoader(
                directory,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
                silent_errors=True,
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"  [OK] 加载了 {len(txt_docs)} 个文本文件")
        except Exception as e:
            print(f"  [WARN] 加载文本文件时出错: {e}")

        # 3. 加载 .json 文件（扣子知识库导出常用格式）
        try:
            json_loader = DirectoryLoader(
                directory,
                glob="**/*.json",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
                silent_errors=True,
            )
            json_docs = json_loader.load()
            documents.extend(json_docs)
            print(f"  [OK] 加载了 {len(json_docs)} 个 JSON 文件")
        except Exception as e:
            print(f"  [WARN] 加载 JSON 文件时出错: {e}")

        print(f"  [OK] 共加载 {len(documents)} 个文档")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成小块

        为什么要分割？
        - LLM 的上下文窗口有限，不能把整个文档塞进去
        - 小块更容易匹配用户的提问
        - chunk_overlap 确保上下文不会在分割点丢失

        Args:
            documents: 原始文档列表

        Returns:
            分割后的文档列表
        """
        split_docs = self.text_splitter.split_documents(documents)
        print(f"  [OK] 分割为 {len(split_docs)} 个文本块")
        return split_docs

    def process_directory(self, directory: str) -> List[Document]:
        """
        处理目录中的所有文档（加载 + 分割，一步到位）

        Args:
            directory: 文档目录路径

        Returns:
            处理后的文档列表
        """
        print(f"\n[DocumentProcessor] 开始处理目录: {directory}")
        documents = self.load_documents(directory)
        if not documents:
            print("  [WARN] 没有找到任何文档！")
            return []
        return self.split_documents(documents)


# ============================================================
# 便捷函数，供其他模块调用
# ============================================================

def load_and_split_documents(directory: str) -> List[Document]:
    """加载并分割文档"""
    processor = DocumentProcessor()
    return processor.process_directory(directory)


# ============================================================
# 测试代码：直接运行此文件可以测试
# ============================================================

if __name__ == "__main__":
    # 获取知识库目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_base_dir = os.path.join(current_dir, "..", "data", "knowledge_base")

    print("=" * 60)
    print("测试文档加载器")
    print("=" * 60)

    processor = DocumentProcessor()
    docs = processor.process_directory(knowledge_base_dir)

    print(f"\n{'=' * 60}")
    print(f"加载结果：共 {len(docs)} 个文本块")
    print(f"{'=' * 60}")

    # 打印前 3 个文本块看看效果
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- 文本块 {i + 1} ---")
        print(f"来源: {doc.metadata.get('source', '未知')}")
        print(f"长度: {len(doc.page_content)} 字符")
        print(f"内容:\n{doc.page_content[:200]}...")
        print()
