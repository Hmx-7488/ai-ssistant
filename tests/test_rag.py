"""
饭小二客服系统测试

运行方法：
    .venv/Scripts/python tests/test_rag.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding='utf-8')

from src.agents.customer_service import CustomerServiceAgent


def test_rag_system():
    """测试饭小二客服系统"""
    print("=" * 60)
    print("饭小二客服系统测试")
    print("=" * 60)

    agent = CustomerServiceAgent()

    test_cases = [
        {
            "input": "红烧肉多少钱？",
            "expected_intent": "价格查询",
            "description": "价格查询测试",
            "must_contain": ["58"]
        },
        {
            "input": "人均消费多少？",
            "expected_intent": "价格查询",
            "description": "人均消费测试",
            "must_contain": ["人均"]
        },
        {
            "input": "有什么推荐的菜？",
            "expected_intent": "菜品推荐",
            "description": "菜品推荐测试",
            "must_contain": ["招牌红烧肉"]
        },
        {
            "input": "我想吃鱼",
            "expected_intent": "点餐咨询",
            "description": "点餐咨询测试",
            "must_contain": ["秘制烤鱼", "88"]
        },
        {
            "input": "我想订座",
            "expected_intent": "订座服务",
            "description": "订座服务测试",
            "must_contain": ["日期"]
        },
        {
            "input": "营业到几点？",
            "expected_intent": "点餐咨询",
            "description": "营业时间查询",
            "must_contain": ["11:00", "22:00"]
        },
        {
            "input": "推荐老人吃的菜",
            "expected_intent": "菜品推荐",
            "description": "老人推荐测试",
            "must_contain": ["老人"]
        },
        {
            "input": "花生过敏推荐吃什么",
            "expected_intent": "菜品推荐",
            "description": "过敏原过滤测试",
            "must_contain": ["花生"]
        },
        {
            "input": "3个人150元预算",
            "expected_intent": "菜品推荐",
            "description": "预算推荐测试",
            "must_contain": ["150"]
        },
        {
            "input": "全部菜单",
            "expected_intent": "点餐咨询",
            "description": "菜单查询测试",
            "must_contain": ["菜单"]
        },
        {
            "input": "那个多少钱？",
            "expected_intent": "价格查询",
            "description": "模糊价格边界测试",
            "must_contain": ["哪道菜"]
        },
        {
            "input": "我想吃鱼，多少钱？",
            "expected_intent": "价格查询",
            "description": "多意图价格优先测试",
            "must_contain": ["秘制烤鱼", "88"]
        },
    ]

    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}: {test['description']}")
        print(f"输入: {test['input']}")
        print("-" * 60)

        intent = agent.router.invoke(test['input'])
        print(f"识别意图: {intent} (期望: {test['expected_intent']})")

        # 注意：参数名是 sid，不是 session_id
        response = agent.invoke(test['input'], sid=f"test-{i}")
        print(f"回复: {response[:120]}")

        missing = [text for text in test["must_contain"] if text not in response]
        intent_ok = intent == test['expected_intent']
        passed = intent_ok and not missing
        results.append({
            "test": test['description'],
            "passed": passed,
            "intent": intent,
            "intent_ok": intent_ok,
            "missing": missing,
        })

    # 汇总
    print(f"\n{'=' * 60}")
    print("测试汇总")
    print("=" * 60)

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        parts = []
        if not r['intent_ok']:
            parts.append(f"意图不符({r['intent']})")
        if r['missing']:
            parts.append(f"缺失:{r['missing']}")
        suffix = f" - {'; '.join(parts)}" if parts else ""
        print(f"[{status}] {r['test']}{suffix}")

    print(f"\n通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.0f}%)")


if __name__ == "__main__":
    test_rag_system()
