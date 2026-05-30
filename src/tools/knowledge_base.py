"""
轻量知识库读取与检索。

这个模块不依赖向量模型，用于演示、测试和外部 API 不稳定时的兜底回答。
"""

import os
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Dish:
    name: str
    price: int
    description: str
    spicy: str = ""
    portion: str = ""
    category: str = ""
    funny_description: str = ""
    pairing: tuple[str, ...] = ()
    cook_time: str = ""


class RestaurantKnowledgeBase:
    """从本地 Markdown 知识库中提取菜单和 FAQ 信息。"""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                "knowledge_base",
            )
        self.base_dir = os.path.abspath(base_dir)
        self.menu_text = self._read("menu.md")
        self.faq_text = self._read("restaurant_faq.md")
        self.products = self._read_json("products.json")
        self.faq = self._read_json("faq.json")
        self.dishes = self._parse_products(self.products) or self._parse_menu(self.menu_text)

    def _read(self, filename: str) -> str:
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def _read_json(self, filename: str) -> dict:
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _parse_products(self, data: dict) -> List[Dish]:
        dishes: List[Dish] = []
        for category in data.get("菜品分类", []):
            category_name = category.get("分类名", "")
            for item in category.get("菜品", []):
                price = item.get("price")
                if price is None:
                    continue
                dishes.append(
                    Dish(
                        name=item.get("name", ""),
                        price=int(price),
                        description=item.get("标准描述") or item.get("description") or "",
                        spicy=self._spicy_label(item.get("辣度")),
                        portion=item.get("份量", ""),
                        category=category_name,
                        funny_description=item.get("幽默描述", ""),
                        pairing=tuple(item.get("搭配建议", [])),
                        cook_time=item.get("烹饪时间", ""),
                    )
                )
        return dishes

    def _parse_menu(self, text: str) -> List[Dish]:
        dishes: List[Dish] = []
        current_category = ""
        sections = re.split(r"\n(?=## |### )", text)

        for section in sections:
            category_match = re.match(r"##\s+(.+)", section)
            if category_match and not section.startswith("###"):
                current_category = self._clean_title(category_match.group(1))
                continue

            title_match = re.match(r"###\s+(.+)", section)
            price_match = re.search(r"\*\*价格\*\*：(\d+)元", section)
            if not title_match or not price_match:
                continue

            name = self._clean_title(title_match.group(1))
            description = self._field(section, "描述")
            spicy = self._field(section, "辣度")
            portion = self._field(section, "分量")
            dishes.append(
                Dish(
                    name=name,
                    price=int(price_match.group(1)),
                    description=description,
                    spicy=spicy,
                    portion=portion,
                    category=current_category,
                )
            )

        return dishes

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"[*#`]", "", title).strip()
        title = re.sub(r"^[^\w\u4e00-\u9fff]+", "", title).strip()
        return title

    @staticmethod
    def _field(section: str, name: str) -> str:
        match = re.search(rf"\*\*{name}\*\*：(.+)", section)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _spicy_label(level: object) -> str:
        labels = {
            0: "不辣",
            1: "微辣",
            2: "微辣",
            3: "中辣",
            4: "重辣",
            5: "特辣",
        }
        try:
            return labels.get(int(level), str(level))
        except (TypeError, ValueError):
            return str(level or "")

    def find_dish(self, query: str) -> Optional[Dish]:
        """按菜名、关键词或同类菜品查找最匹配的一道菜。"""
        query = query.strip()
        if not query:
            return None

        for dish in self.dishes:
            if dish.name in query or query in dish.name:
                return dish

        aliases = {
            "红烧肉": "招牌红烧肉",
            "肉": "招牌红烧肉",
            "鱼": "秘制烤鱼",
            "烤鱼": "秘制烤鱼",
            "牛肉": "水煮牛肉",
            "鸡": "宫保鸡丁",
            "鸡丁": "宫保鸡丁",
            "素": "清炒时蔬",
            "蔬菜": "清炒时蔬",
            "炒饭": "招牌炒饭",
            "面": "手工面条",
            "汤": "番茄蛋花汤",
            "饮料": "酸梅汤",
            "饮品": "酸梅汤",
            "甜品": "芒果布丁",
        }
        for keyword, dish_name in aliases.items():
            if keyword in query:
                return self.get_dish_by_name(dish_name)
        return None

    def get_dish_by_name(self, name: str) -> Optional[Dish]:
        for dish in self.dishes:
            if dish.name == name:
                return dish
        return None

    def recommend(self, query: str, limit: int = 3) -> List[Dish]:
        """根据用户输入推荐菜品。"""
        lowered = query.lower()
        if any(word in query for word in ["鱼", "海鲜"]):
            names = ["秘制烤鱼", "番茄蛋花汤", "酸梅汤"]
        elif any(word in query for word in ["辣", "川菜", "重口"]):
            names = ["水煮牛肉", "鱼香肉丝", "秘制烤鱼"]
        elif any(word in query for word in ["素", "清淡", "不辣"]):
            names = ["清炒时蔬", "番茄蛋花汤", "手工面条"]
        elif any(word in lowered for word in ["one", "single"]) or "一人" in query:
            names = ["招牌炒饭", "清炒时蔬", "酸梅汤"]
        else:
            names = ["招牌红烧肉", "秘制烤鱼", "水煮牛肉"]

        dishes = [dish for name in names if (dish := self.get_dish_by_name(name))]
        return dishes[:limit]

    def search_text(self, query: str, limit: int = 3) -> List[str]:
        """用简单关键词评分检索 Markdown 段落。"""
        json_paragraphs = []
        for group in self.faq.get("常见问题", {}).values():
            if not isinstance(group, list):
                continue
            for item in group:
                answer = item.get("幽默版") or item.get("answer")
                if answer:
                    keywords = " ".join(item.get("keywords", []))
                    json_paragraphs.append(f"{item.get('question', '')}\n{keywords}\n{answer}")

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", self.menu_text + "\n\n" + self.faq_text)
            if paragraph.strip()
        ] + json_paragraphs
        tokens = self._tokens(query)
        scored = []
        for paragraph in paragraphs:
            score = sum(1 for token in tokens if token and token in paragraph)
            if score:
                scored.append((score, paragraph))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [paragraph for _, paragraph in scored[:limit]]

    @staticmethod
    def _tokens(query: str) -> Iterable[str]:
        words = re.findall(r"[\u4e00-\u9fff]{1,4}|[A-Za-z0-9]+", query)
        return set(words + list(query))


@lru_cache(maxsize=1)
def get_restaurant_kb() -> RestaurantKnowledgeBase:
    return RestaurantKnowledgeBase()
