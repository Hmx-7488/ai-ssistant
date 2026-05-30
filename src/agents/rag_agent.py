"""
RAG Agent - 知识库问答代理

作用：
1. 接收用户问题
2. 从向量数据库检索相关文档
3. 结合 LLM 生成回答
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import CUSTOMER_SERVICE_PROMPT
from src.tools.vector_store import VectorStore


class RAGAgent:
    """RAG Agent: 基于知识库的问答代理"""

    def __init__(self):
        """初始化 RAG Agent"""
        print("[RAGAgent] 初始化...")

        # 1. 初始化向量数据库
        self.vector_store = VectorStore()
        self.retriever = self.vector_store.get_retriever()

        # 2. 初始化 LLM
        self.llm = ChatOpenAI(
            model="qwen-plus",
            temperature=0.7,
        )

        # 3. 创建 Prompt 模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CUSTOMER_SERVICE_PROMPT + "\n\n以下是检索到的相关信息:\n{context}"),
            ("human", "{question}"),
        ])

        # 4. 构建 RAG Chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("[RAGAgent] 初始化完成")

    def invoke(self, question: str) -> str:
        """
        回答用户问题

        Args:
            question: 用户问题

        Returns:
            回答内容
        """
        print(f"\n[RAGAgent] 收到问题: {question}")

        # 检索相关文档
        docs = self.retriever.invoke(question)
        print(f"[RAGAgent] 检索到 {len(docs)} 个相关文档")

        # 生成回答
        answer = self.chain.invoke(question)

        print(f"[RAGAgent] 生成回答完成")
        return answer


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 RAG Agent")
    print("=" * 60)

    agent = RAGAgent()

    test_questions = [
        "红烧肉多少钱？",
        "有什么推荐的菜？",
        "营业到几点？",
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"问题: {question}")
        print("-" * 60)
        answer = agent.invoke(question)
        print(f"回答: {answer}")
