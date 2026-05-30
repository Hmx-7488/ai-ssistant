import os
import sys
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


@dataclass
class Dish:
    """菜品数据类"""
    id: int
    name: str
    price: int
    category: str
    description: str
    spice: str
    portion: str
    tags: List[str]
    avoid: List[str]
    allergens: List[str]
    customizable: bool
    cook_time: str
    pairing: List[str]


class UnifiedKnowledgeBase:
    """统一知识库：单一事实源"""

    def __init__(self, data_file: str = None):
        if data_file is None:
            data_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "data", "knowledge_base", "menu.json"
            )

        self.data_file = data_file
        self._load_data()
        self._build_indexes()

    def _load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.dishes = []
        for item in self.data.get("菜品", []):
            dish = Dish(
                id=item["id"],
                name=item["name"],
                price=item["price"],
                category=item["category"],
                description=item["description"],
                spice=item["spice"],
                portion=item["portion"],
                tags=item.get("tags", []),
                avoid=item.get("avoid", []),
                allergens=item.get("allergens", []),
                customizable=item.get("customizable", False),
                cook_time=item.get("cook_time", ""),
                pairing=item.get("pairing", [])
            )
            self.dishes.append(dish)

        self.suites = self.data.get("套餐", [])
        self.people_tags = self.data.get("人群标签", {})
        self.avoid_rules = self.data.get("忌口规则", {})
        self.scene_recommend = self.data.get("场景推荐", {})
        self.promotions = self.data.get("优惠活动", {})
        self.restaurant_info = self.data.get("餐厅信息", {})

    def _build_indexes(self):
        self.name_index = {dish.name: dish for dish in self.dishes}
        self.alias_index = {
            "红烧肉": "招牌红烧肉",
            "肉": "招牌红烧肉",
            "鱼": "秘制烤鱼",
            "烤鱼": "秘制烤鱼",
            "牛肉": "水煮牛肉",
            "鸡": "宫保鸡丁",
            "鸡丁": "宫保鸡丁",
            "素菜": "清炒时蔬",
            "蔬菜": "清炒时蔬",
            "炒饭": "招牌炒饭",
            "面": "手工面条",
            "面条": "手工面条",
            "汤": "番茄蛋花汤",
            "蛋花汤": "番茄蛋花汤",
            "西瓜汁": "鲜榨西瓜汁",
            "酸梅汤": "酸梅汤",
            "布丁": "芒果布丁",
            "甜品": "芒果布丁",
        }
        self.tag_index = {}
        for dish in self.dishes:
            for tag in dish.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                self.tag_index[tag].append(dish)

    def find_dish(self, query: str) -> Optional[Dish]:
        query = query.strip().lower()
        for dish in self.dishes:
            if dish.name in query or query in dish.name:
                return dish
        for alias, dish_name in self.alias_index.items():
            if alias in query:
                return self.name_index.get(dish_name)
        return None

    def get_dish_by_name(self, name: str) -> Optional[Dish]:
        return self.name_index.get(name)

    def get_dishes_by_category(self, category: str) -> List[Dish]:
        return [d for d in self.dishes if d.category == category]

    def get_dishes_by_tag(self, tag: str) -> List[Dish]:
        return self.tag_index.get(tag, [])

    def get_non_spicy_dishes(self) -> List[Dish]:
        return [d for d in self.dishes if "不辣" in d.spice or d.spice == "无"]

    def get_allergen_free_dishes(self, allergen: str) -> List[Dish]:
        return [d for d in self.dishes if allergen not in d.allergens]

    def recommend_for_people(self, people_type: str) -> Dict:
        for key in self.people_tags:
            if key in people_type or people_type in key:
                return self.people_tags[key]
        return {}

    def get_avoid_info(self, avoid_type: str) -> Dict:
        for key in self.avoid_rules:
            if key in avoid_type or avoid_type in key:
                return self.avoid_rules[key]
        return {}

    def get_scene_recommend(self, scene: str) -> Dict:
        for key in self.scene_recommend:
            if key in scene or scene in key:
                return self.scene_recommend[key]
        return {}

    def get_suite_by_people(self, people: int) -> Optional[Dict]:
        if people <= 1:
            for suite in self.suites:
                if "一人" in suite["name"]:
                    return suite
        elif people <= 2:
            for suite in self.suites:
                if "双人" in suite["name"]:
                    return suite
        elif people <= 4:
            for suite in self.suites:
                if "四人" in suite["name"]:
                    return suite
        return None

    def recommend_by_budget(self, budget: int, people: int) -> List[List[Dish]]:
        per_person = budget / people
        recommendations = []
        if per_person <= 30:
            combos = [
                [self.name_index["招牌炒饭"], self.name_index["酸梅汤"]],
                [self.name_index["手工面条"], self.name_index["清炒时蔬"]],
            ]
            recommendations.extend(combos)
        elif per_person <= 60:
            combos = [
                [self.name_index["鱼香肉丝"], self.name_index["番茄蛋花汤"], self.name_index["米饭"] if "米饭" in self.name_index else self.name_index["招牌炒饭"]],
            ]
            recommendations.extend(combos)
        return recommendations

    def format_dish_info(self, dish: Dish) -> str:
        lines = [f"{dish.name} - {dish.price}元/份"]
        lines.append(f"  {dish.description}")
        if dish.spice != "无":
            lines.append(f"  辣度：{dish.spice}")
        if dish.portion:
            lines.append(f"  分量：{dish.portion}")
        if dish.allergens:
            lines.append(f"  过敏原：{', '.join(dish.allergens)}")
        return "\n".join(lines)


_kb_instance = None

def get_unified_kb() -> UnifiedKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = UnifiedKnowledgeBase()
    return _kb_instance
