"""
Router Agent - 意图识别代理

作用：
1. 接收用户输入
2. 识别用户意图（价格查询、菜品推荐、订座服务等）
3. 返回意图类别
"""

import os
import sys
from typing import Dict, Iterable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import ROUTER_PROMPT


class RouterAgent:
    """Router Agent: 意图识别代理"""

    # 意图类别
    INTENTS = [
        "价格查询",
        "菜品推荐",
        "点餐咨询",
        "订座服务",
        "投诉建议",
        "订单查询",
        "转人工",
        "闲聊互动",
    ]

    KEYWORDS: Dict[str, Iterable[str]] = {
        "投诉建议": ["投诉", "不满", "差评", "太咸", "太淡", "难吃", "服务态度", "不好吃", "建议", "问题"],
        "转人工": ["转人工", "人工客服", "转人工客服", "真人客服", "找人工", "人工服务", "呼叫人工", "接人工", "转接人工", "给我转人工"],
        "订单查询": ["订单", "催", "好了没", "好了吗", "上菜", "等多久", "还没上", "进度"],
        "订座服务": ["订座", "订位", "预约", "包间", "座位", "位置", "明天", "今晚"],
        "价格查询": ["多少钱", "价格", "几块", "几元", "贵吗", "收费", "费用", "人均", "消费", "报价", "放香菜", "有放", "怎么卖", "卖多少", "什么价"],
        "菜品推荐": ["推荐", "好吃", "特色", "招牌", "不知道吃什么", "来点", "必点", "预算", "人吃", "吃什么", "爸妈吃", "有什么", "过敏", "小宝宝", "宝宝", "能吃", "辣的", "辣一点", "辣味", "想吃辣", "清淡的", "下饭的", "想吃", "元"],
        "点餐咨询": ["我要吃", "我要一份", "来一份", "点一份", "点餐", "辣吗", "忌口", "食材", "分量", "素食", "儿童餐", "营业", "地址", "wifi", "打包", "外卖", "会员", "优惠", "菜单", "所有菜", "全部菜", "能不能", "可以吗", "不要香菜", "有wifi", "停车", "发票", "生日", "有什么汤", "有什么饮品", "有什么甜品", "饮料", "喝的", "果汁", "茶水", "宝宝椅", "团购", "排队", "等位", "加饭", "续碗", "靠窗", "安静", "几人份", "几个人吃", "够吃", "放了什么", "有放", "热菜", "凉菜"],
        "闲聊互动": ["你好", "您好", "嗨", "在吗", "谢谢", "你是谁", "再见", "心情不好", "不开心", "难过"],
    }

    # 菜品名称关键词（优先级最高：含菜品名 → 点餐咨询，避免被订座等误匹配）
    _DISH_KEYWORDS = [
        "红烧肉", "烤鱼", "水煮牛肉", "鱼香肉丝", "宫保鸡丁", "清炒时蔬",
        "清蒸鲈鱼", "白灼虾", "炒饭", "面条", "蛋花汤", "布丁", "虾仁",
        "鸡丁", "鲈鱼", "牛肉", "时蔬", "鸡丁", "米饭", "馒头", "饺子",
        "西瓜汁", "酸梅汤", "牛腩", "排骨", "虾", "鱼",
    ]

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

    def __init__(self):
        """初始化 Router Agent"""
        self._safe_print("[RouterAgent] 初始化...")
        self.chain = None
        self._llm_initialized = False
        self._safe_print("[RouterAgent] 初始化完成（LLM路由将按需加载）")

    def _ensure_llm_chain(self):
        """Lazy-init LLM chain on first use."""
        if self._llm_initialized:
            return
        self._llm_initialized = True
        try:
            llm = ChatOpenAI(
                model=os.getenv("CHAT_MODEL", "qwen-plus"),
                temperature=0.1,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", ROUTER_PROMPT),
                ("human", "{input}"),
            ])
            self.chain = prompt | llm | StrOutputParser()
            self._safe_print("[RouterAgent] LLM路由已加载")
        except Exception as e:
            self._safe_print(f"[RouterAgent] LLM路由加载失败: {e}")
            self.chain = None

    def invoke(self, user_input: str) -> str:
        """
        识别用户意图

        Args:
            user_input: 用户输入

        Returns:
            意图类别名称
        """
        self._safe_print(f"\n[RouterAgent] 识别意图: {user_input}")

        # 优先使用确定性规则，保证演示、测试和无网络环境稳定。
        intent = self._classify_by_rules(user_input)
        if intent:
            self._safe_print(f"[RouterAgent] 规则识别结果: {intent}")
            return intent

        # 规则未命中 → 用 LLM 识别长尾意图
        self._ensure_llm_chain()
        if self.chain is None:
            self._safe_print("[RouterAgent] LLM不可用，归类为闲聊互动")
            return "闲聊互动"

        try:
            intent = self.chain.invoke({"input": user_input}).strip()
        except Exception as e:
            self._safe_print(f"[RouterAgent] LLM调用失败: {e}")
            return "闲聊互动"

        # 验证意图是否在预定义类别中
        if intent not in self.INTENTS:
            self._safe_print(f"[RouterAgent] 未识别的意图: {intent}，归类为闲聊互动")
            intent = "闲聊互动"

        self._safe_print(f"[RouterAgent] 识别结果: {intent}")
        return intent

    def _classify_by_rules(self, user_input: str) -> str:
        """基于关键词和优先级识别意图。"""
        normalized = user_input.lower().strip()
        if not normalized:
            return "闲聊互动"

        # 菜系查询（"有江西菜吗"、"有没有川菜"）→ 点餐咨询
        _CUISINE_KW = ["江西菜", "川菜", "粤菜", "湘菜", "鲁菜", "日料", "韩餐", "西餐", "泰国菜"]
        if any(kw in normalized for kw in _CUISINE_KW):
            return "点餐咨询"

        # 0. 菜品名称优先 → 点餐咨询（"蛋炒饭是几个人吃的" → 点餐，不是订座）
        if any(kw in normalized for kw in self._DISH_KEYWORDS):
            # 但如果同时有价格关键词，走价格查询
            if any(kw in normalized for kw in self.KEYWORDS["价格查询"]):
                return "价格查询"
            return "点餐咨询"

        # 1. 多意图场景里价格优先，例如"我想吃鱼，多少钱？"
        if any(keyword in normalized for keyword in self.KEYWORDS["价格查询"]):
            return "价格查询"

        for intent in ["转人工", "投诉建议", "订单查询", "菜品推荐", "订座服务", "点餐咨询", "闲聊互动"]:
            if any(keyword in normalized for keyword in self.KEYWORDS[intent]):
                return intent

        return ""


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Router Agent")
    print("=" * 60)

    agent = RouterAgent()

    test_inputs = [
        "红烧肉多少钱？",
        "有什么推荐的菜？",
        "我想订座",
        "菜太咸了！",
        "你好",
        "我的菜好了吗？",
    ]

    for user_input in test_inputs:
        print(f"\n输入: {user_input}")
        intent = agent.invoke(user_input)
        print(f"意图: {intent}")
