from langchain_core.tools import tool
from typing import Optional
from collections import Counter

from src.tools.unified_kb import get_unified_kb
from src.tools.faq_engine import match_faq


# ============================================================
#  Menu Tools
# ============================================================

@tool
def query_dish(dish_name: str) -> str:
    """Query details of a specific dish by name.

    Args:
        dish_name: The name of the dish to look up (e.g. "红烧肉", "烤鱼")

    Returns:
        Detailed information about the dish including price, description,
        spice level, portion size, allergens, and customization options.
    """
    kb = get_unified_kb()
    dish = kb.find_dish(dish_name)
    if not dish:
        return f'未找到菜品 "{dish_name}"'
    return kb.format_dish_info(dish)


@tool
def search_dishes(
    category: str = "",
    tag: str = "",
    max_price: int = 0,
    spice_level: str = "",
) -> str:
    """Search dishes by category, tag, max price, or spice level.

    Args:
        category: Dish category filter (e.g. "热菜", "凉菜", "汤品", "主食", "饮品", "甜品")
        tag: Tag filter (e.g. "人气必点", "清淡", "下饭", "海鲜")
        max_price: Maximum price per dish in yuan (0 = no limit)
        spice_level: Spice level filter ("不辣", "微辣", "中辣", "重辣")

    Returns:
        List of matching dishes with their key details.
    """
    kb = get_unified_kb()
    results = list(kb.dishes)

    if category:
        results = [d for d in results if category in d.category]
    if tag:
        results = [d for d in results if tag in d.tags]
    if max_price > 0:
        results = [d for d in results if d.price <= max_price]
    if spice_level:
        results = [d for d in results if spice_level in d.spice]

    if not results:
        return "没有找到符合条件的菜品"

    return "\n\n".join(kb.format_dish_info(d) for d in results)


@tool
def get_full_menu() -> str:
    """Get the complete restaurant menu organized by category.

    Returns:
        Full menu with all dishes, prices, and descriptions.
    """
    kb = get_unified_kb()
    categories = {}
    for d in kb.dishes:
        categories.setdefault(d.category, []).append(d)

    sections = []
    for cat, dishes in categories.items():
        lines = [f"=== {cat} ==="]
        for d in dishes:
            line = f"  {d.name}  {d.price}元/份  {d.description}"
            if d.spice and d.spice != "无":
                line += f" [{d.spice}]"
            lines.append(line)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ============================================================
#  Recommend Tools
# ============================================================

@tool
def check_allergen(dish_name: str, allergen: str) -> str:
    """Check whether a dish contains a specific allergen or ingredient.

    Args:
        dish_name: Name of the dish to check (e.g. "水煮牛肉")
        allergen: The allergen or ingredient to check for (e.g. "花生", "虾", "香菜", "葱")

    Returns:
        Whether the dish contains the specified allergen/ingredient.
    """
    kb = get_unified_kb()
    dish = kb.find_dish(dish_name)
    if not dish:
        return f'未找到菜品 "{dish_name}"'

    sources = []
    if allergen in dish.allergens:
        sources.append("过敏原列表")
    if allergen in dish.description:
        sources.append("菜品描述")
    if allergen in str(dish.tags):
        sources.append("标签")
    if allergen in dish.avoid:
        sources.append("忌口列表")

    if sources:
        return f"{dish.name} 含有 {allergen}（来源：{', '.join(sources)}），请注意避开"
    return f"{dish.name} 的信息中未发现 {allergen}"


@tool
def recommend_combo(
    people: int = 3,
    budget: int = 0,
    avoid_allergens: str = "",
    taste: str = "",
) -> str:
    """Recommend a combo of dishes for a group, optionally respecting budget and allergens.

    Args:
        people: Number of diners (default 3)
        budget: Total budget in yuan (0 = no budget limit)
        avoid_allergens: Comma-separated allergens/ingredients to avoid (e.g. "花生,香菜")
        taste: Preferred taste profile (e.g. "清淡", "辣", "酸甜")

    Returns:
        Recommended combo with individual prices and total.
    """
    kb = get_unified_kb()
    avoid_list = [a.strip() for a in avoid_allergens.split(",") if a.strip()] if avoid_allergens else []

    candidates = list(kb.dishes)

    # Filter allergens
    for allergen in avoid_list:
        candidates = [
            d for d in candidates
            if allergen not in d.allergens
            and allergen not in d.description
            and allergen not in str(d.tags)
        ]

    if not candidates:
        return "根据您的忌口要求，暂时无法推荐合适的菜品组合，建议联系人工客服"

    # Taste filter
    if taste:
        if "清淡" in taste:
            sf = [d for d in candidates if "不辣" in d.spice or d.spice == "无"]
            if sf:
                candidates = sf
        elif "辣" in taste:
            sf = [d for d in candidates if "辣" in d.spice and d.spice != "不辣"]
            if sf:
                candidates = sf

    # Suite check
    suite = kb.get_suite_by_people(people)
    if suite and (budget == 0 or suite.get("price", 0) <= budget):
        suite_dishes = ", ".join(suite.get("dishes", []))
        suite_name = suite.get("name", "套餐")
        suite_price = suite.get("price", "?")
        return f"推荐套餐：{suite_name} ({suite_price}元)\n包含：{suite_dishes}"

    # Budget-based combo building
    per_person = budget / people if budget > 0 and people > 0 else 0

    if per_person > 0:
        mains = [d for d in candidates if d.category == "热菜"]
        sides = [d for d in candidates if d.category in ("凉菜", "素菜")]
        soups = [d for d in candidates if d.category == "汤品"]
        drinks = [d for d in candidates if d.category == "饮品"]
        rice_list = [d for d in candidates if d.category == "主食"]

        combo = []
        spent = 0
        limit = budget

        # 1 main per 2 people
        n_mains = max(1, people // 2)
        mains_sorted = sorted(mains, key=lambda d: d.price, reverse=True)
        for i in range(min(n_mains, len(mains_sorted))):
            if spent + mains_sorted[i].price <= limit:
                combo.append(mains_sorted[i])
                spent += mains_sorted[i].price

        # 1 side
        for s in sorted(sides, key=lambda d: d.price):
            if spent + s.price <= limit:
                combo.append(s)
                spent += s.price
                break

        # 1 soup
        for s in sorted(soups, key=lambda d: d.price):
            if spent + s.price <= limit:
                combo.append(s)
                spent += s.price
                break

        # Drinks: 1 per person
        drink_count = 0
        for dr in sorted(drinks, key=lambda d: d.price):
            while drink_count < people and spent + dr.price <= limit:
                combo.append(dr)
                spent += dr.price
                drink_count += 1
            if drink_count >= people:
                break

        # Rice
        for r in sorted(rice_list, key=lambda d: d.price):
            for _ in range(people):
                if spent + r.price <= limit:
                    combo.append(r)
                    spent += r.price

        if not combo:
            return f"{budget}元预算对于{people}位用餐来说比较紧张，建议适当增加预算或选择简餐"

        counts = Counter(d.name for d in combo)
        lines = [f"为您{people}位推荐的{budget}元套餐："]
        for name, cnt in counts.items():
            price = next(d.price for d in combo if d.name == name)
            suffix = f" x{cnt}" if cnt > 1 else ""
            lines.append(f"  - {name}{suffix}  {price}元/份")
        lines.append(f"合计：约{spent}元")
        return "\n".join(lines)

    # No budget: suggest popular picks
    popular = [d for d in candidates if "人气" in str(d.tags) or "必点" in str(d.tags)]
    if not popular:
        popular = candidates[:5]
    lines = [f"为{people}位推荐的人气菜品："]
    for d in popular[:min(people + 2, len(popular))]:
        lines.append(kb.format_dish_info(d))
    return "\n".join(lines)


@tool
def recommend_for_people(people_type: str) -> str:
    """Get dish recommendations tailored to a specific type of diner.

    Args:
        people_type: Type of diners (e.g. "情侣约会", "家庭聚餐", "商务宴请", "带小孩", "老人")

    Returns:
        Tailored recommendations for the specified dining scenario.
    """
    kb = get_unified_kb()
    rec = kb.recommend_for_people(people_type)
    if not rec:
        return f'暂无针对 "{people_type}" 的专属推荐'

    dishes = rec.get("推荐菜品", [])
    reason = rec.get("推荐理由", "")

    lines = [f"【{people_type}专属推荐】"]
    if reason:
        lines.append(f"推荐理由：{reason}")
    lines.append("")
    lines.append("推荐菜品：")

    for d_name in dishes:
        dish = kb.get_dish_by_name(d_name)
        if dish:
            lines.append(kb.format_dish_info(dish))
        else:
            lines.append(f"  - {d_name}")
    return "\n".join(lines)


# ============================================================
#  FAQ Tools
# ============================================================

@tool
def query_faq(question: str) -> str:
    """Look up the answer to a frequently asked question about the restaurant.

    Args:
        question: The customer's question (e.g. "有没有停车位", "可以带宠物吗", "营业到几点")

    Returns:
        The best matching FAQ answer, or a note that no match was found.
    """
    result = match_faq(question)
    if result:
        return result
    return "暂无该问题的FAQ答案，建议使用其他工具查询或转人工"


# ============================================================
#  Restaurant Info Tools
# ============================================================

@tool
def get_promotions() -> str:
    """Get current restaurant promotions and special offers.

    Returns:
        List of active promotions with details.
    """
    kb = get_unified_kb()
    promos = kb.promotions
    if not promos:
        return "暂无当前优惠活动"

    lines = ["当前优惠活动："]
    for name, detail in promos.items():
        if isinstance(detail, dict):
            desc = detail.get("description", detail.get("描述", ""))
            lines.append(f"  - {name}: {desc}")
        else:
            lines.append(f"  - {name}: {detail}")
    return "\n".join(lines)


@tool
def get_restaurant_info(info_type: str = "") -> str:
    """Get general restaurant information such as hours, address, parking, etc.

    Args:
        info_type: Type of info to retrieve (e.g. "营业时间", "地址", "电话", "停车", "wifi", "包厢").
                   If empty, returns all available info.

    Returns:
        Requested restaurant information.
    """
    kb = get_unified_kb()
    info = kb.restaurant_info
    if not info:
        return "暂无餐厅信息"

    if info_type:
        for key, val in info.items():
            if info_type in key or key in info_type:
                if isinstance(val, dict):
                    return f"{key}: {val.get('description', val.get('描述', str(val)))}"
                return f"{key}: {val}"
        return f'未找到关于 "{info_type}" 的信息'

    lines = ["餐厅信息："]
    for key, val in info.items():
        if isinstance(val, dict):
            desc = val.get("description", val.get("描述", str(val)))
            lines.append(f"  - {key}: {desc}")
        else:
            lines.append(f"  - {key}: {val}")
    return "\n".join(lines)


# ============================================================
#  Tool Group Lists
# ============================================================

MENU_TOOLS = [query_dish, search_dishes, get_full_menu]
RECOMMEND_TOOLS = [check_allergen, recommend_combo, recommend_for_people]
BOOKING_TOOLS = [query_faq, get_restaurant_info]
COMPLAINT_TOOLS = [query_faq, get_restaurant_info, get_promotions]