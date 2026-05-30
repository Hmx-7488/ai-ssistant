# -*- coding: utf-8 -*-
"""CLI interactive chat with 饭小二"""
import sys, json, urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://localhost:8084/chat"
SID = "cli_session"

def chat(msg):
    data = json.dumps({"message": msg, "session_id": SID}).encode("utf-8")
    req = urllib.request.Request(BASE, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("reply", "ERROR")
    except Exception as e:
        return f"[连接失败] {e}"

print("=" * 50)
print("  饭小二 CLI 聊天 (输入 quit 退出)")
print("=" * 50)
print()

while True:
    try:
        user_input = input("你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        print("饭小二: 下次再来呀~")
        break
    reply = chat(user_input)
    print(f"饭小二: {reply}")
    print()

