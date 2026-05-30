# -*- coding: utf-8 -*-
"""Demo preflight check - 30 seconds before presenting"""
import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request

BASE = "http://localhost:8084"
passed = 0
total = 0

def check(name, ok, detail=""):
    global passed, total
    total += 1
    if ok:
        passed += 1
        print("  PASS  " + name)
    else:
        print("  FAIL  " + name + (" - " + detail if detail else ""))

def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("\n" + "=" * 50)
print("  Preflight Check")
print("=" * 50)

# 1. connectivity
print("\n[1] Connectivity")
try:
    h = get("/health")
    check("Backend reachable", True)
except Exception as e:
    check("Backend reachable", False, str(e))
    print("  Start backend: uvicorn src.api.main:app --host 0.0.0.0 --port 8084")
    sys.exit(1)

# 2. components
print("\n[2] Components")
check("Agent ready", h.get("agent_ready"))
check("menu.json", h.get("menu_json_exists"))
check("faq.md", h.get("faq_md_exists"))
check("Vector store", h.get("vector_store_ok"))
check("Min QA", h.get("min_qa_ok"), h.get("min_qa_error", ""))

# 3. deterministic - each query uses its own session to avoid state bleed
print("\n[3] Deterministic")
for i, (q, kw) in enumerate([
    ("全部菜单", ["菜单"]),
    ("宫保鸡丁多少钱", ["42"]),
    ("人均消费多少", ["50-80"]),
    ("营业到几点", ["11:00"]),
    ("你们地址在哪", ["88"]),
    ("有wifi吗", ["Guest"]),
    ("有包间吗", ["包间"]),
    ("可以开发票吗", ["发票"]),
    ("可以打包吗", ["打包"]),
    ("有停车位吗", ["20"]),
    ("有优惠吗", ["8"]),
    ("有儿童餐吗", ["儿童"]),
]):
    try:
        r = post("/chat", {"message": q, "session_id": f"pf_d{i}"})
        ok = r.get("success") and any(k in r.get("reply","") for k in kw)
        check(q, ok, r.get("reply","")[:60])
    except Exception as e:
        check(q, False, str(e))

# 4. recommendation - each query uses its own session
print("\n[4] Recommendation")
for i, (q, kw) in enumerate([
    ("推荐老人吃的菜", ["老人"]),
    ("推荐小孩吃的菜", ["儿童"]),
    ("花生过敏推荐吃什么", ["花生"]),
    ("3个人150元预算", ["150"]),
    ("我想订座", ["日期"]),
]):
    try:
        r = post("/chat", {"message": q, "session_id": f"pf_r{i}"})
        ok = r.get("success") and any(k in r.get("reply","") for k in kw)
        check(q, ok, r.get("reply","")[:60])
    except Exception as e:
        check(q, False, str(e))

# 5. boundary - each query uses its own session
print("\n[5] Boundary")
for i, (q, kw) in enumerate([
    ("那个多少钱", ["哪道菜"]),
    ("你们有江西菜吗", ["江西"]),
    ("可以不要香菜吗", ["香菜"]),
    ("菜太咸了", ["抱歉"]),
]):
    try:
        r = post("/chat", {"message": q, "session_id": f"pf_b{i}"})
        ok = r.get("success") and any(k in r.get("reply","") for k in kw)
        check(q, ok, r.get("reply","")[:60])
    except Exception as e:
        check(q, False, str(e))

# 6. multi-turn
print("\n[6] Multi-turn")
sid = "pf_mt"
try:
    r1 = post("/chat", {"message": "推荐老人吃的菜", "session_id": sid})
    r2 = post("/chat", {"message": "换一个", "session_id": sid})
    ok = r1.get("success") and r2.get("success") and r2.get("reply","") != r1.get("reply","")
    check("old man -> switch", ok, r2.get("reply","")[:60])
except Exception as e:
    check("old man -> switch", False, str(e))

sid2 = "pf_mt2"
try:
    r1 = post("/chat", {"message": "3个人150元", "session_id": sid2})
    r2 = post("/chat", {"message": "改成两个人", "session_id": sid2})
    ok = r1.get("success") and r2.get("success") and "2" in r2.get("reply","")
    check("3p 150y -> 2p", ok, r2.get("reply","")[:60])
except Exception as e:
    check("3p 150y -> 2p", False, str(e))

# summary
print("\n" + "=" * 50)
if passed == total:
    print(f"  ALL PASS {passed}/{total} - Ready to demo!")
else:
    print(f"  {passed}/{total} passed, {total-passed} need attention")
print("=" * 50)
