# -*- coding: utf-8 -*-
"""
FAQ Engine - 结构化FAQ知识库

从 restaurant_faq.json 加载所有问答对，提供确定性匹配。
不依赖LLM，不依赖向量检索，100%确定性。
"""
import os
import json
from typing import Optional

_FAQ = None

def _load_faq():
    global _FAQ
    if _FAQ is not None:
        return _FAQ
    path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base", "restaurant_faq.json")
    if not os.path.exists(path):
        _FAQ = {}
        return _FAQ
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # Flatten into list of (keywords, answer, category)
    _FAQ = []
    for category, items in raw.items():
        for name, entry in items.items():
            _FAQ.append({
                "name": name,
                "category": category,
                "keywords": entry.get("keywords", []),
                "answer": entry.get("answer", ""),
            })
    return _FAQ

# 特殊菜品食材映射
_DISH_INGREDIENTS = {
    "番茄蛋花汤": "番茄蛋花汤的食材：新鲜番茄、鸡蛋、葱花。不含生姜。可以要求不放葱。",
    "招牌红烧肉": "招牌红烧肉的食材：五花肉、酱油、冰糖、八角、桂皮。慢炖3小时。",
    "秘制烤鱼": "秘制烤鱼的食材：鲜活鲈鱼、独家秘制酱料。配菜：豆芽、藕片、木耳。默认不放香菜。",
    "水煮牛肉": "水煮牛肉的食材：牛里脊、花椒、干辣椒、豆芽、莴笋。可调辣度。",
    "宫保鸡丁": "宫保鸡丁的食材：鸡胸肉、花生、干辣椒、花椒、葱段。含花生过敏原。",
    "鱼香肉丝": "鱼香肉丝的食材：猪肉丝、木耳、胡萝卜、青椒、笋丝。微辣。",
    "清炒时蔬": "清炒时蔬的食材：当季新鲜时蔬（每日不同），清淡少油。",
    "清蒸鲈鱼": "清蒸鲈鱼的食材：鲜活鲈鱼、姜丝、葱丝、蒸鱼豉油。少油少盐。",
    "白灼虾": "白灼虾的食材：鲜虾、姜片。蘸姜醋汁食用。含虾过敏原。",
    "招牌炒饭": "招牌炒饭的食材：米饭、虾仁、鸡蛋、火腿丁、青豆。",
    "手工面条": "手工面条的食材：面粉、鸡蛋。可选浇头：牛肉/炸酱/番茄鸡蛋。",
    "芒果布丁": "芒果布丁的食材：芒果、牛奶、糖、吉利丁。含乳制品。",
    "鲜榨西瓜汁": "鲜榨西瓜汁：纯西瓜鲜榨，不加水不加糖。",
    "酸梅汤": "酸梅汤的食材：乌梅、山楂、桂花、冰糖。古法熬制。",
}

def match_faq(query: str) -> Optional[str]:
    """
    从FAQ知识库中匹配答案。
    
    返回: 匹配到的答案字符串，或 None 表示未匹配。
    """
    faq = _load_faq()
    if not faq:
        return None
    
    q = query.lower().strip()

    # 1. 先检查菜品食材问题（"XX有放YY吗" / "YY有YY吗" / "XX里面有什么食材"）
    is_ingredient_q = any(w in q for w in ["放", "有", "食材", "里面", "什么", "成分", "配料"])
    if is_ingredient_q:
        for dish_name, info in _DISH_INGREDIENTS.items():
            # 支持部分匹配：用户说"红烧肉"也能匹配"招牌红烧肉"
            short_name = dish_name.replace("招牌", "").replace("秘制", "").replace("手工", "").replace("鲜榨", "")
            if dish_name in q or short_name in q:
                return info
    
    # 2. 关键词匹配（加权 + 短词上下文排除，防止"热菜"匹配"空调"）
    _CONTEXT_EXCLUDES = {
        "热": ["菜", "汤", "炒", "煮", "蒸", "灼", "饭", "面", "辣", "乎"],
    }
    best_match = None
    best_score = 0
    for entry in faq:
        score = 0
        for kw in entry["keywords"]:
            kw_lower = kw.lower()
            if kw_lower in q:
                # 上下文排除：如果短关键词后面紧跟菜品相关字，跳过
                if kw_lower in _CONTEXT_EXCLUDES:
                    idx = q.index(kw_lower)
                    after = q[idx + len(kw_lower):idx + len(kw_lower) + 2]
                    if any(c in after for c in _CONTEXT_EXCLUDES[kw_lower]):
                        continue
                score += len(kw_lower)
        if score > best_score:
            best_score = score
            best_match = entry

    if best_match and best_score > 0:
        return best_match["answer"]
    
    return None
