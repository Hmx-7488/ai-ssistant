"""
饭小二客服系统完整测试

覆盖执行计划验证矩阵 A-E:
  A. 单轮基础能力
  B. 推荐与约束能力
  C. 多轮承接能力
  D. 边界与兜底能力
  E. 演示稳定性

运行方法：
    .venv/Scripts/python tests/test_rag.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding='utf-8')

from src.agents.customer_service import CustomerServiceAgent


def run_tests():
    """运行完整验证矩阵"""
    print("=" * 60)
    print("饭小二客服系统 - 完整验证矩阵")
    print("=" * 60)

    agent = CustomerServiceAgent()
    results = []

    def check(name, ok, detail=""):
        results.append({"name": name, "ok": ok, "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))

    # ===== A. 单轮基础能力 =====
    print("\n[A] 单轮基础能力")
    for i, (q, kw) in enumerate([
        ("你好", ["你好", "欢迎", "饭小二"]),
        ("全部菜单", ["菜单"]),
        ("宫保鸡丁多少钱", ["42"]),
        ("人均消费多少", ["50-80", "人均"]),
        ("营业到几点", ["11:00"]),
        ("你们地址在哪", ["88"]),
        ("有wifi吗", ["Guest", "WiFi"]),
        ("有包间吗", ["包间"]),
        ("可以开发票吗", ["发票"]),
        ("可以打包吗", ["打包"]),
    ]):
        r = agent.invoke(q, sid=f"a_{i}")
        ok = any(k in r for k in kw)
        check(q, ok, r[:50])

    # ===== B. 推荐与约束能力 =====
    print("\n[B] 推荐与约束能力")
    for i, (q, kw) in enumerate([
        ("推荐老人吃的菜", ["老人"]),
        ("推荐小孩吃的菜", ["儿童"]),
        ("孕妇吃什么比较合适", ["孕妇"]),
        ("花生过敏推荐吃什么", ["花生"]),
        ("海鲜过敏推荐吃什么", ["海鲜"]),
        ("我不吃辣推荐一下", ["辣"]),
        ("3个人150元预算清淡一点", ["150"]),
        ("一个人晚上吃点什么", ["一人"]),
        ("推荐下饭的菜", ["下饭"]),
    ]):
        r = agent.invoke(q, sid=f"b_{i}")
        ok = any(k in r for k in kw)
        check(q, ok, r[:50])

    # ===== C. 多轮承接能力 =====
    print("\n[C] 多轮承接能力")

    # C1: 老人推荐 -> 换一个
    sid = "c1"
    r1 = agent.invoke("推荐老人吃的菜", sid=sid)
    r2 = agent.invoke("换一个", sid=sid)
    check("老人->换一个", r1 != r2)

    # C2: 小孩推荐 -> 清淡
    sid = "c2"
    r1 = agent.invoke("推荐小孩吃的菜", sid=sid)
    r2 = agent.invoke("再来个清淡点的", sid=sid)
    check("小孩->清淡", r1 != r2)

    # C3: 预算 -> 改人数
    sid = "c3"
    r1 = agent.invoke("3个人150元预算", sid=sid)
    r2 = agent.invoke("改成两个人", sid=sid)
    check("3人150->2人", "2" in r2)

    # C4: 过敏 -> 还有别的吗
    sid = "c4"
    r1 = agent.invoke("花生过敏推荐吃什么", sid=sid)
    r2 = agent.invoke("还有别的吗", sid=sid)
    check("花生过敏->还有别的", r1 != r2)

    # C5: 订座 -> 具体信息
    sid = "c5"
    r1 = agent.invoke("我想订座", sid=sid)
    r2 = agent.invoke("明天晚上6个人", sid=sid)
    check("订座->6人", "6" in r2)

    # ===== D. 边界与兜底能力 =====
    print("\n[D] 边界与兜底能力")
    for i, (q, kw) in enumerate([
        ("那个多少钱", ["哪道菜", "哪一道", "明确"]),
        ("你们有江西菜吗", ["江西"]),
        ("可以不要香菜吗", ["香菜"]),
        ("我现在心情不好", ["甜品", "难过", "安慰", "开心", "汤", "红烧肉"]),
        ("我对这道菜不满意", ["抱歉", "不满"]),
    ]):
        r = agent.invoke(q, sid=f"d_{i}")
        ok = any(k in r for k in kw)
        check(q, ok, r[:50])

    # ===== E. 演示稳定性 =====
    print("\n[E] 演示稳定性")
    fail = 0
    for i in range(20):
        r = agent.invoke("全部菜单", sid=f"e_{i}")
        if "菜单" not in r:
            fail += 1
    check(f"20次连续调用 ({20-fail}/20)", fail == 0)

    # 关键问答无乱码
    r = agent.invoke("全部菜单", sid="e_utf8")
    has_garbled = any(ord(c) > 0xFFFF for c in r)
    check("无乱码", not has_garbled)

    # ===== 汇总 =====
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    if passed == total:
        print(f"  ALL PASS {passed}/{total} - Ready to demo!")
    else:
        print(f"  {passed}/{total} passed, {total - passed} need attention")
        for r in results:
            if not r["ok"]:
                print(f"    FAIL: {r['name']}")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
