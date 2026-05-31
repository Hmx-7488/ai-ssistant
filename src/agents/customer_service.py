# -*- coding: utf-8 -*-
import os, sys, re, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from src.agents.router_agent import RouterAgent
from src.tools.unified_kb import get_unified_kb
from src.tools.faq_engine import match_faq
from src.tools.vector_store import retrieve_context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser

# 中文数字 → 阿拉伯数字
_CN_DIGITS = {"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}

def _parse_int(text: str, default: int = 0) -> int:
    """从文本中提取整数，支持阿拉伯数字和中文数字。"""
    # 先试阿拉伯数字
    m = re.search(r"(\d+)", text)
    if m: return int(m.group(1))
    # 再试中文数字
    for ch, val in _CN_DIGITS.items():
        if ch in text: return val
    return default


# ═══════════════════════════════════════════════════════════
# ToolAgent：LLM + Function Calling 工具调用基类
# ═══════════════════════════════════════════════════════════

class ToolAgent:
    """
    Base class for LLM agents with tool calling (Function Calling).

    Subclass and set self.system_prompt + self.tools in __init__.
    invoke() runs a loop: LLM → tool_calls → execute → feed back → repeat.
    """
    def __init__(self, llm, system_prompt: str, tools: list):
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools)

    def invoke(self, user_input: str, context: str = "", history: list = None) -> str:
        """
        Run the tool-calling loop and return a text response.

        Args:
            user_input: Current user message
            context: Extra context string (e.g. RAG results)
            history: List of (user_msg, assistant_msg) tuples for multi-turn
        """
        sys_msg = self.system_prompt
        if context:
            sys_msg += f"\n\n当前上下文：{context}"
        messages = [SystemMessage(content=sys_msg)]

        # Inject conversation history
        if history:
            for user_msg, ai_msg in history[-3:]:  # last 3 turns max
                messages.append(HumanMessage(content=user_msg))
                messages.append(AIMessage(content=ai_msg))

        messages.append(HumanMessage(content=user_input))

        for _ in range(5):  # max 5 tool-calling rounds
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                return ai_msg.content or ""

            for tc in ai_msg.tool_calls:
                tool_fn = self.tools_by_name.get(tc["name"])
                if tool_fn:
                    result = tool_fn.invoke(tc["args"])
                else:
                    result = f"未知工具: {tc['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # Final call without tools to get a text response
        final = self.llm_with_tools.invoke(messages)
        return final.content or ""


# ═══════════════════════════════════════════════════════════
# 专用工具调用 Agent（LLM + Tools 版本）
# ═══════════════════════════════════════════════════════════

class MenuToolAgent(ToolAgent):
    """菜品问答 Agent（LLM + Tools 版本）"""
    def __init__(self, llm):
        from src.tools.agent_tools import MENU_TOOLS, BOOKING_TOOLS
        super().__init__(
            llm,
            system_prompt=(
                "你是饭小二餐厅的菜品问答专家。你的职责是回答顾客关于菜品、价格、"
                "菜单、食材、营业信息等问题。\n"
                "规则：\n"
                "1. 使用工具查询菜品信息，不要编造\n"
                "2. 回复简洁，1-2句话\n"
                "3. 语气温暖亲切，适当用emoji\n"
                "4. 涉及过敏原要特别提醒\n"
                "5. 不确定的信息建议联系人工客服"
            ),
            tools=MENU_TOOLS + BOOKING_TOOLS,
        )


class RecommendToolAgent(ToolAgent):
    """推荐 Agent（LLM + Tools 版本）"""
    def __init__(self, llm):
        from src.tools.agent_tools import RECOMMEND_TOOLS, MENU_TOOLS
        super().__init__(
            llm,
            system_prompt=(
                "你是饭小二，幽默的餐厅推荐专家。你的职责是根据顾客需求推荐菜品。\n"
                "规则：\n"
                "1. 使用工具查询和推荐菜品，不要编造\n"
                "2. 有预算时，用 recommend_combo 工具搭配套餐\n"
                "3. 有过敏/忌口时，用 check_allergen 工具逐一检查\n"
                "4. 有人群类型时，用 recommend_for_people 工具\n"
                "5. 回复带emoji，突出亮点，2-3句话\n"
                "6. 推荐后主动询问是否满意、有无忌口"
            ),
            tools=RECOMMEND_TOOLS + MENU_TOOLS,
        )


class BookingToolAgent(ToolAgent):
    """订座 Agent（LLM + Tools 版本）"""
    def __init__(self, llm):
        from src.tools.agent_tools import BOOKING_TOOLS
        super().__init__(
            llm,
            system_prompt=(
                "你是饭小二订座助手。你的职责是帮顾客预订座位。\n"
                "规则：\n"
                "1. 引导顾客提供：日期时间、人数、是否需要包间、联系电话\n"
                "2. 使用 get_restaurant_info 查询包间信息\n"
                "3. 回复简洁，2句话内\n"
                "4. 确认信息后告知预订成功"
            ),
            tools=BOOKING_TOOLS,
        )


class ComplaintToolAgent(ToolAgent):
    """投诉 Agent（LLM + Tools 版本）"""
    def __init__(self, llm):
        from src.tools.agent_tools import COMPLAINT_TOOLS
        super().__init__(
            llm,
            system_prompt=(
                "你是饭小二投诉处理专员。你的职责是处理顾客投诉和不满。\n"
                "规则：\n"
                "1. 真诚道歉，不推卸责任\n"
                "2. 给出具体解决方案（重做/换菜/免单/补偿）\n"
                "3. 2句话内，语气诚恳\n"
                "4. 严重问题建议转人工客服"
            ),
            tools=COMPLAINT_TOOLS,
        )


class ConversationState:
    """Multi-turn conversation state to track user constraints."""

    # ── Follow-up 关键词 ──
    FOLLOW_UP_KW = ["还有","换个","刚才","改成","重新","再来",
                     "其他的","别的","换一个","不要这个","换个清淡","再来一个"]
    # "不要"后面跟食材/忌口词时，是饮食偏好而非 follow-up
    _PREFERENCE_SUFFIX = ["香菜","葱","姜","蒜","辣","花椒","蒜蓉","洋葱","芹菜","香葱"]

    def __init__(self):
        self.people = 0
        self.budget = 0
        self.people_type = ""
        self.taste = ""
        self.allergies = []
        self.scene = ""
        self.last_dishes = []
        self.last_reply = ""
        self.last_intent = ""
        self.message_history = []  # [(user_msg, ai_reply), ...]

    def reset_constraints(self):
        """重置推荐约束（保留对话历史 last_dishes/last_reply）。"""
        self.people = 0
        self.budget = 0
        self.people_type = ""
        self.taste = ""
        self.allergies = []
        self.scene = ""

    def update_from(self, text: str):
        """从用户文本提取约束条件。"""
        # 人数：支持阿拉伯数字和中文数字
        people_num = 0
        m = re.search(r"(\d+)\s*个人?", text)
        if m: people_num = int(m.group(1))
        elif re.search(r"[一二两三四五六七八九十]\s*个人", text):
            people_num = _parse_int(text)
        if people_num > 0: self.people = people_num
        # 预算：多种表达方式
        budget = 0
        m = re.search(r"预算\s*(\d+)", text)
        if m: budget = int(m.group(1))
        if not budget: m = re.search(r"(\d+)\s*[元块]", text)
        if m: budget = int(m.group(1))
        # "推荐300的菜" / "吃200的" / "点150的"
        if not budget:
            m = re.search(r"(?:推荐|吃|点|来)\s*(\d{2,})\s*的", text)
            if m: budget = int(m.group(1))
        # "200怎么吃" / "200块吃什么" / "200够吗"
        if not budget:
            m = re.search(r"(\d{2,})\s*(?:怎么|够|咋)", text)
            if m: budget = int(m.group(1))
        if budget > 0: self.budget = budget
        for k in ["宝宝","小宝宝","婴儿","baby","儿童","小孩","孩子"]:
            if k in text: self.people_type = "儿童"; break
        for k in ["老人","长辈","爸妈","父母","爷爷","奶奶"]:
            if k in text: self.people_type = "老人"; break
        for k in ["孕妇"]:
            if k in text: self.people_type = "孕妇"; break
        for k in ["健身","减脂","减肥"]:
            if k in text: self.people_type = "健身减脂"; break
        for k in ["素食"]:
            if k in text: self.people_type = "素食者"; break
        for k in ["清淡","少油","少盐"]:
            if k in text: self.taste = "清淡"
        for k in ["辣","麻辣"]:
            if k in text and "不" not in text[:text.index(k)]: self.taste = "辣"
        for k in ["不辣","不要辣","不吃辣","免辣"]:
            if k in text: self.taste = "不辣"
        # 过敏原：只在有负面语境时才提取（过敏/不吃/不要/避开/不能吃/忌口）
        _neg_ctx = any(k in text for k in ["过敏","不吃","不要","避开","不能吃","忌口","难受"])
        if _neg_ctx:
            allergen_map = {"花生":"花生","海鲜":"海鲜","鸡蛋":"鸡蛋","牛肉":"牛肉",
                            "乳制品":"乳制品","麸质":"麸质","虾":"虾","鱼类":"鱼类"}
            for keyword, allergen in allergen_map.items():
                if keyword in text and allergen not in self.allergies:
                    self.allergies.append(allergen)
        for k in ["一人食","一个人","自己吃"]:
            if k in text: self.scene = "一人食"; break
        for k in ["约会","两人","两个人","情侣"]:
            if k in text: self.scene = "两人约会"; break
        for k in ["聚餐","聚会","家庭","一家人"]:
            if k in text: self.scene = "家庭聚餐"; break
        for k in ["夜宵","宵夜","晚上加餐"]:
            if k in text: self.scene = "夜宵"; break
        for k in ["下饭","配饭","超级下饭"]:
            if k in text: self.scene = "下饭"; break
        avoid_kw = {"香菜":"香菜","葱":"葱","姜":"姜","蒜":"蒜","不要香菜":"香菜"}
        for kw, val in avoid_kw.items():
            if kw in text and f"不吃{val}" not in self.allergies:
                if "不要" in text or "不吃" in text or "去" in text:
                    self.allergies.append(f"不吃{val}")
        self.last_intent = ""

    def is_follow_up(self, text: str) -> bool:
        """
        判断是否为上一轮的多轮承接。

        严格条件（必须同时满足）：
        1. 上一轮有推荐结果（last_dishes 非空）
        2. 当前输入很短（≤12字）—— 长句几乎不可能是 follow-up
        3. 包含 follow-up 关键词，或是一个简短回答（数字/日期）
        """
        if not self.last_dishes:
            return False
        t = text.strip()
        # 条件2: 太长的句子不是 follow-up
        if len(t) > 12:
            return False
        # 条件3a: 包含 follow-up 关键词
        # 但 "不要香菜" 等忌口表达不算 follow-up
        _is_preference = t.startswith("不要") and any(t.endswith(s) or f"不要{s}" in t for s in self._PREFERENCE_SUFFIX)
        if _is_preference:
            return False
        for kw in self.FOLLOW_UP_KW:
            if kw in t:
                return True
        # 条件3b: 纯数字/日期/时间回答（承接订座等场景）
        if re.match(r"^[\d\s:/\-\.月日号点晚上早上午下午]+$", t) and len(t) <= 10:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "people": self.people, "budget": self.budget,
            "people_type": self.people_type, "taste": self.taste,
            "allergies": self.allergies, "scene": self.scene,
            "last_dishes": self.last_dishes, "last_intent": self.last_intent,
        }

    def context_summary(self) -> str:
        """Human-readable context for prompts."""
        parts = []
        if self.people > 0: parts.append(f"{self.people}人")
        if self.budget > 0: parts.append(f"预算{self.budget}元")
        if self.people_type: parts.append(self.people_type)
        if self.taste: parts.append(f"口味:{self.taste}")
        if self.allergies: parts.append(f"忌口:{','.join(self.allergies)}")
        if self.scene: parts.append(f"场景:{self.scene}")
        return "，".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════
# 专用 Agent（每个 Agent 有自己的 LLM 链 + RAG）
# ═══════════════════════════════════════════════════════════

class MenuQAAgent:
    """菜品问答 Agent：价格、菜单、食材、营业信息。"""
    def __init__(self, llm):
        self.kb = get_unified_kb()
        self.tool_agent = MenuToolAgent(llm)
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "你是饭小二餐厅的菜品问答专家。根据知识库回答，不编造。回复简洁，1-2句。"),
            ("human", "{input}")
        ]) | llm | StrOutputParser()

    def invoke(self, inp: str, state: ConversationState) -> str:
        t = inp.lower()

        # ── 0. FAQ 知识库优先匹配（50+ 结构化问答，100% 确定性） ──
        faq_answer = match_faq(inp)
        if faq_answer and faq_answer != "dynamic":
            return faq_answer

        # ── 确定性快速路径（不调用 LLM） ──
        # 菜系查询（我们没有的菜系）
        for cuisine in ["江西", "川菜", "粤菜", "湘菜", "鲁菜", "苏菜", "浙菜", "闽菜", "徽菜", "日料", "韩餐", "西餐", "泰国菜", "印度菜"]:
            if cuisine in t:
                return f"我们是家常菜融合餐厅，暂时没有{cuisine}，但有红烧肉、烤鱼、水煮牛肉等招牌硬菜，要不要看看菜单？😋"
        if "菜单" in t or "所有菜" in t or "全部菜" in t:
            return self._format_menu()
        if "营业" in t or "几点" in t or "开门" in t:
            hrs = self.kb.restaurant_info.get("营业时间", {})
            return f"午餐 {hrs.get('午餐','11:00-14:00')}  晚餐 {hrs.get('晚餐','17:00-22:00')}，周末提前半小时开门~"
        if "地址" in t or "在哪" in t:
            return "美食街88号饭小二餐厅，地铁3号线美食街站A口步行3分钟~"
        if "wifi" in t or "无线" in t or "网络" in t:
            return "WiFi：FanXiaoEr_Guest  密码：fanxiaoer888~"
        if "打包" in t or "外带" in t:
            return "打包盒免费！外卖搜饭小二餐厅就行~"
        if "会员" in t or "优惠" in t or "折扣" in t:
            promos = self.kb.promotions
            return "优惠活动：\n" + "\n".join(f"• {k}：{v}" for k, v in promos.items())
        if "人均" in t or "消费" in t:
            info = self.kb.restaurant_info.get("人均消费", {})
            return "人均消费：\n" + "\n".join(f"{k}：{v}" for k, v in info.items())
        if "外卖" in t:
            return "美团/饿了么搜饭小二餐厅下单，保温袋配送，到手还是热乎的~"
        if "发票" in t:
            return "支持电子发票和纸质发票，消费后跟服务员说一声就行~"
        if "停车" in t:
            return "门口20个免费车位，周末较紧张，建议地铁出行~"
        if "儿童餐" in t or "小孩餐" in t:
            return "儿童套餐35元：小份炒饭+蛋花汤+果汁，还免费提供儿童座椅和餐具~"
        if "生日" in t:
            return "生日特权：全单9折+送长寿面+甜品，提前3天预约还有气球布置~"
        if "包间" in t:
            return "4个包间：小包间(4-6人)低消300、大包间(8-12人)低消600、VIP(10-16人)低消1000带KTV，提前3天订~"
        # 按分类查询（汤/饮品/甜品/主食）
        for cat, label in [("汤","汤品"),("饮品","饮品"),("饮料","饮品"),("甜品","甜品"),("主食","主食"),("甜点","甜品")]:
            if cat in t:
                ds = self.kb.get_dishes_by_category(label)
                if ds:
                    lines = [f"**{d.name}** {d.price}元 - {d.description[:20]}" for d in ds]
                    return f"{label}有这些：\n" + "\n".join(lines)
        # 辣度查询：只在用户提到具体菜品时走快速路径，否则交给推荐Agent
        if "辣" in t and "烤鱼" in t:
            return "秘制烤鱼可选微辣/中辣/特辣，跟服务员说一声就行~"
        if "辣" in t:
            _is_recommend_q = any(w in t for w in ["推荐","想吃","有什么","来点","吃什么","辣的","辣一点","辣味"])
            if not _is_recommend_q:
                d = self.kb.find_dish(inp)
                if d and d.customizable:
                    return f"{d.name}可以调辣度，跟服务员说就行~"
        # "不要香菜" 单独出现才走快速路径；复合查询（有预算/人数）跳过
        if "不要" in inp and "香菜" in inp:
            has_budget_or_people = re.search(r"\d+\s*(?:元|块|个人|人)|预算", inp)
            if not has_budget_or_people:
                return "可以！下单时跟服务员说不要香菜，厨房会单独处理~"
        # 温度偏好（"有冰的吗" / "可以加热吗"）
        if re.search(r"冰的|冰镇|加冰|常温|热的|加热|温的", inp):
            d = self.kb.find_dish(inp)
            name = d.name if d else ""
            if "西瓜" in inp or "汁" in inp:
                return f"{'西瓜汁' if not name else name}默认常温，可以加冰，跟服务员说一声就行~"
            if "酸梅" in inp or "汤" in inp:
                return f"{'酸梅汤' if not name else name}可以选冰的或常温，夏天推荐冰镇的！跟服务员说一声~"
            return "饮品都可以选冰的或常温，跟服务员说一声就行~"
        # 食材成分提问（"XX有放YY吗"）— 只匹配具体食材，排除"冰的/大份/小份"等
        _non_ingredient = r"冰|热|大份|小份|多|少|辣|咸|甜|酸"
        ingredient_q = re.search(r"有放(.+?)吗|放了(.+?)吗", inp)
        if not ingredient_q:
            m = re.search(r"有(.{1,4}?)吗", inp)
            if m and not re.search(_non_ingredient, m.group(1)):
                ingredient_q = m
        if ingredient_q:
            ingredient = (ingredient_q.group(1) or ingredient_q.group(2) or ingredient_q.group(3) or "").strip()
            d = self.kb.find_dish(inp)
            if d:
                # Check avoid list and allergens
                has_it = ingredient in str(d.allergens) or ingredient in str(getattr(d, 'avoid', []))
                note = f"{d.name}的配菜有：豆芽、藕片、木耳等" if "烤鱼" in d.name else ""
                if ingredient == "香菜":
                    return f"{d.name}默认不放香菜，如果有忌口下单时跟服务员确认一下就行~"
                if note:
                    return note + f"。{ingredient}的话跟服务员说一声可以单独处理~"
                return f"{d.name}的具体配料建议跟服务员确认一下哦~"
            # General question
            if ingredient == "香菜":
                return "大部分菜默认不放香菜，有忌口的话下单时跟服务员说一声就行~"
        # 点菜意图（"我要/来/点一份XX"）
        order_m = re.search(r"[我要来点来搞]\s*[一份个]?\s*(.+)", inp)
        if order_m:
            dish_name = order_m.group(1).strip().rstrip("，。！?")
            d = self.kb.find_dish(dish_name)
            if d:
                state.last_dishes = [d.name]
                allergy_warn = ""
                if d.allergens:
                    allergy_warn = f"\n⚠️ 提醒：含过敏原（{'、'.join(d.allergens)}）"
                return f"好的！**{d.name}** {d.price}元 已记下~{allergy_warn}\n有忌口或过敏吗？要不要清淡/不辣？"
        # 常见主食/配料（不在菜品列表但顾客常问）
        _STAPLES = {
            "米饭": "米饭免费续碗！点菜自动送~",
            "白饭": "米饭免费续碗！点菜自动送~",
            "馒头": "馒头2元/个，手工现蒸~",
            "煎饼": "葱油煎饼8元，香脆可口~",
            "饺子": "手工水饺28元/份（20个），可选猪肉白菜/韭菜鸡蛋~",
        }
        for name, answer in _STAPLES.items():
            if name in inp:
                return answer
        # 菜品详情查询 + 主动追问忌口
        d = self.kb.find_dish(inp)
        if d:
            state.last_dishes = [d.name]
            info = self.kb.format_dish_info(d)
            # 如果用户是在问价格（不是纯好奇），追加忌口确认
            if any(w in inp for w in ["多少钱", "价格", "来一份", "我要"]):
                info += "\n有忌口或过敏吗？"
            return info
        # 模糊指代 → 反问（只在确实找不到任何菜品时触发）
        if any(w in t for w in ["那个","这个"]) and "人均" not in t:
            return "想问哪道菜价格？说个菜名我帮你查~"
        # RAG 兜底 → ToolAgent（LLM + Function Calling）
        ctx = retrieve_context(inp, top_k=3)
        context = ctx if ctx else ""
        return self.tool_agent.invoke(inp, context=context, history=state.message_history)

    def _format_menu(self) -> str:
        cats = {}
        for d in self.kb.dishes:
            cats.setdefault(d.category, []).append(d)
        lines = ["菜单来啦~"]
        for cat, ds in cats.items():
            lines.append(f"\n【{cat}】")
            for d in ds:
                lines.append(f"  {d.name} {d.price}元")
        return "\n".join(lines)


class RecommendAgent:
    """推荐 Agent：人群、过敏、预算、场景、口味推荐。"""
    def __init__(self, llm):
        self.kb = get_unified_kb()
        self.tool_agent = RecommendToolAgent(llm)
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "你是饭小二，幽默的餐厅推荐专家。根据菜品信息推荐，突出亮点，带emoji。"),
            ("human", "{input}")
        ]) | llm | StrOutputParser()

    # 食材/类别关键词 → 过滤条件
    _CATEGORY_FILTERS = {
        "素菜": lambda d: "素食" in d.tags or "蔬菜" in d.tags,
        "素食": lambda d: "素食" in d.tags or "蔬菜" in d.tags,
        "肉": lambda d: "肉食" in d.tags,
        "海鲜": lambda d: "海鲜" in d.tags,
        "鱼": lambda d: "海鲜" in d.tags,
        "汤": lambda d: d.category == "汤品",
        "主食": lambda d: d.category == "主食",
        "饮品": lambda d: d.category == "饮品",
        "甜品": lambda d: d.category == "甜品",
        "下饭": lambda d: "下饭" in d.tags,
    }

    def _filter_by_keyword(self, inp: str):
        """根据输入中的类别关键词筛选菜品。"""
        t = inp.lower()
        for kw, filt in self._CATEGORY_FILTERS.items():
            if kw in t:
                return [d for d in self.kb.dishes if filt(d)]
        return None

    def invoke(self, inp: str, state: ConversationState, is_follow: bool) -> str:
        if is_follow and state.last_dishes:
            return self._follow_up(inp, state)
        # 人群+预算/人数组合 → _for_people（内部处理预算）
        if state.people_type:
            return self._for_people(state)
        # 有预算时，预算优先（内部处理过敏原）
        if state.budget > 0:
            if state.people == 0: state.people = 3  # 默认3人
            return self._by_budget(state)
        if state.allergies:
            return self._allergen_safe(state)
        if state.scene:
            return self._by_scene(state)
        if state.taste == "辣":
            return self._spicy(state)
        if state.taste == "不辣":
            return self._non_spicy(state)
        if state.taste == "清淡":
            return self._light(state)
        # 类别关键词筛选（素菜/海鲜/汤/主食...）
        filtered = self._filter_by_keyword(inp)
        if filtered:
            picks = filtered[:4]
            lines = [f"**{d.name}** {d.price}元 - {d.description[:25]}" for d in picks]
            state.last_dishes = [d.name for d in picks]
            reply = "推荐这些：\n" + "\n".join(lines)
            state.last_reply = reply
            return reply
        # 默认：特色菜 / 招牌推荐 + 引导补充信息
        t = inp.lower()
        sig = self.kb.get_dishes_by_tag("招牌")
        if "特色" in t or "招牌" in t:
            picks = sig[:3]
            lines = [f"**{d.name}** {d.price}元 - {d.description[:35]}" for d in picks]
            reply = "我们的特色菜：\n" + "\n".join(lines) + "\n想了解哪道菜的详情？或者告诉我人数和预算，帮你搭配~"
        else:
            # 随机选2-3道不同方向的菜，避免千篇一律
            all_dishes = list(self.kb.dishes)
            random.shuffle(all_dishes)
            picks = all_dishes[:3]
            lines = [f"**{d.name}** {d.price}元 - {d.description[:25]}" for d in picks]
            reply = "给你几个方向：\n" + "\n".join(lines) + "\n有忌口吗？几个人吃、预算多少？帮你精准推荐~"
        state.last_dishes = [d.name for d in picks]
        state.last_reply = reply
        return reply

    def _for_people(self, state):
        info = self.kb.recommend_for_people(state.people_type)
        if not info:
            ctx = retrieve_context(f"推荐{state.people_type}吃的菜")
            return self.tool_agent.invoke(f"推荐{state.people_type}吃的菜", context=ctx or "", history=state.message_history)
        recs = info.get("推荐", [])
        note = info.get("说明", "")

        # 如果有预算/人数约束 → 在人群筛选基础上做预算搭配
        if state.budget > 0:
            # 将人群推荐的菜品名映射为实际 Dish 对象，再加上同类别菜品
            people_dish_names = set(recs)
            people_dishes = [d for n in recs if (d := self.kb.get_dish_by_name(n))]
            people_tags = set()
            for d in people_dishes:
                people_tags.update(d.tags)

            # 候选：人群推荐菜品 + 有相同标签的菜品
            candidates = []
            seen = set()
            for d in self.kb.dishes:
                if d.name in people_dish_names or d.tags & people_tags:
                    if d.name not in seen:
                        candidates.append(d)
                        seen.add(d.name)
            # 补充清淡菜品（老人/儿童/孕妇通常偏好清淡）
            if not candidates:
                candidates = [d for d in self.kb.dishes if "清淡" in d.tags]
            if not candidates:
                candidates = list(self.kb.dishes)

            if state.people == 0: state.people = 3
            meat_mains = [d for d in candidates if d.category == "招牌菜" or (d.category == "热菜" and "蔬菜" not in d.tags)]
            veg_sides = [d for d in candidates if d.category in {"主食", "汤品", "甜品"} or (d.category == "热菜" and "蔬菜" in d.tags)]
            drinks = [d for d in candidates if d.category == "饮品"]
            combo, total = [], 0
            for d in sorted(meat_mains, key=lambda d: d.price):
                if total + d.price <= state.budget and len(combo) < state.people:
                    combo.append(d); total += d.price
            for d in sorted(veg_sides, key=lambda d: d.price):
                if total + d.price <= state.budget and d not in combo:
                    combo.append(d); total += d.price; break
            if state.people >= 3:
                remaining = [d for d in veg_sides if d not in combo]
                for d in sorted(remaining, key=lambda d: d.price):
                    if total + d.price <= state.budget:
                        combo.append(d); total += d.price; break
            for d in sorted(drinks, key=lambda d: d.price):
                if total + d.price <= state.budget and d not in combo:
                    combo.append(d); total += d.price; break

            if combo:
                lines = [f"**{d.name}** {d.price}元" for d in combo]
                reply = f"给{state.people_type}推荐，{state.people}人{state.budget}元搭配：\n" + "\n".join(lines) + f"\n共{total}元，余{state.budget - total}元~"
                if note: reply += f"\n({note})"
                state.last_dishes = [d.name for d in combo]
                state.last_reply = reply
                return reply

        # 无预算或预算不足 → 返回人群推荐
        dishes = [f"**{d.name}** {d.price}元 - {d.description[:30]}..."
                  for n in recs[:4] if (d := self.kb.get_dish_by_name(n))]
        state.last_dishes = recs[:4]
        reply = f"给{state.people_type}推荐~\n" + "\n".join(dishes) + (f"\n({note})" if note else "")
        state.last_reply = reply
        return reply

    def _allergen_safe(self, state):
        safe = list(self.kb.dishes)
        # "不吃X" 类的忌口：从菜品描述/标签中排除含该食材的菜品
        _avoid_ingredients = {
            "不吃香菜": ["香菜"], "不吃葱": ["葱"], "不吃姜": ["姜"],
            "不吃蒜": ["蒜"], "不吃花椒": ["花椒"],
        }
        for a in state.allergies:
            if a in _avoid_ingredients:
                # 排除 description 或 tags 中含有该食材的菜品
                for ingredient in _avoid_ingredients[a]:
                    safe = [d for d in safe if ingredient not in d.description and ingredient not in str(d.tags)]
            else:
                # 标准过敏原：直接排除 allergens 列表中含有的
                safe = [d for d in safe if a not in d.allergens]
        recs = safe[:4] or self.kb.dishes[:3]
        lines = [f"**{d.name}** {d.price}元" for d in recs]
        reply = f"避开{','.join(state.allergies)}，推荐：\n" + "\n".join(lines)
        state.last_dishes = [d.name for d in recs]
        state.last_reply = reply
        return reply

    def _by_budget(self, state):
        # 套餐：价格必须在预算内
        for s in self.kb.suites:
            if s["price"] > state.budget:
                continue
            if (state.people <= 2 and "双人" in s["name"]) or \
               (3 <= state.people <= 4 and "四人" in s["name"]) or \
               (state.people == 1 and "一人" in s["name"]):
                reply = f"{state.people}人{state.budget}元，推荐 **{s['name']}**（{s['price']}元）：\n{', '.join(s['items'])}\n{s['description']}，省{s['original_price']-s['price']}元~"
                state.last_dishes = s["items"]
                state.last_reply = reply
                return reply

        # 口味+过敏原约束筛选候选菜品
        candidates = list(self.kb.dishes)
        # 过敏原过滤
        _avoid_map = {"不吃香菜":"香菜","不吃葱":"葱","不吃姜":"姜","不吃蒜":"蒜"}
        for a in state.allergies:
            if a in _avoid_map:
                ingredient = _avoid_map[a]
                candidates = [d for d in candidates if ingredient not in d.description and ingredient not in str(d.tags)]
            else:
                candidates = [d for d in candidates if a not in d.allergens]
        if state.taste == "清淡":
            # 严格只选有"清淡"标签的菜品
            candidates = [d for d in candidates if "清淡" in d.tags]

        # 区分主菜和配菜（蔬菜热菜归为配菜，不算主菜）
        meat_mains = [d for d in candidates if d.category == "招牌菜" or (d.category == "热菜" and "蔬菜" not in d.tags)]
        veg_sides = [d for d in candidates if d.category in {"主食", "汤品", "甜品"} or (d.category == "热菜" and "蔬菜" in d.tags)]
        drinks = [d for d in candidates if d.category == "饮品"]
        combo, total = [], 0

        # 1. 主菜：尽量每人一道（优先便宜的）
        for d in sorted(meat_mains, key=lambda d: d.price):
            if total + d.price <= state.budget and len(combo) < state.people:
                combo.append(d); total += d.price

        # 2. 至少一道蔬菜/汤/主食
        for d in sorted(veg_sides, key=lambda d: d.price):
            if total + d.price <= state.budget and d not in combo:
                combo.append(d); total += d.price
                break

        # 3. 再加一道主食或汤（如果预算充足且人多）
        remaining_sides = [d for d in veg_sides if d not in combo]
        if state.people >= 3:
            for d in sorted(remaining_sides, key=lambda d: d.price):
                if total + d.price <= state.budget:
                    combo.append(d); total += d.price
                    break

        # 4. 剩余预算加饮品
        for d in sorted(drinks, key=lambda d: d.price):
            if total + d.price <= state.budget and d not in combo:
                combo.append(d); total += d.price
                break

        if combo:
            lines = [f"**{d.name}** {d.price}元" for d in combo]
            label = f"（{state.taste}）" if state.taste else ""
            reply = f"{state.people}人{state.budget}元{label}搭配：\n" + "\n".join(lines) + f"\n共{total}元，余{state.budget-total}元~"
            state.last_dishes = [d.name for d in combo]
            state.last_reply = reply
            return reply
        return f"{state.people}人{state.budget}元，推荐招牌炒饭+清炒时蔬，经济实惠~"

    def _by_scene(self, state):
        info = self.kb.get_scene_recommend(state.scene)
        if info:
            recs = info.get("推荐", [])
            note = info.get("说明", "")
            dishes = []
            for name in recs[:4]:
                d = self.kb.get_dish_by_name(name)
                if d: dishes.append(f"**{d.name}** {d.price}元")
                else:
                    for s in self.kb.suites:
                        if name in s["name"]: dishes.append(f"**{s['name']}** {s['price']}元")
            state.last_dishes = recs[:4]
            reply = f"{state.scene}推荐：\n" + "\n".join(dishes) + (f"\n({note})" if note else "")
            state.last_reply = reply
            return reply
        ctx = retrieve_context(f"{state.scene}场景推荐")
        return self.tool_agent.invoke(f"{state.scene}推荐菜品", context=ctx or "", history=state.message_history)

    def _spicy(self, state):
        dishes = [d for d in self.kb.dishes if "辣" in d.spice and "不辣" not in d.spice]
        lines = [f"**{d.name}** {d.price}元 - {d.description[:30]}" for d in dishes[:4]]
        reply = "辣味好菜🌶️：\n" + "\n".join(lines) + "\n辣度可选：微辣/中辣/特辣，跟服务员说就行~"
        state.last_dishes = [d.name for d in dishes[:4]]
        state.last_reply = reply
        return reply

    def _non_spicy(self, state):
        dishes = [d for d in self.kb.dishes if "不辣" in d.spice][:4]
        lines = [f"**{d.name}** {d.price}元" for d in dishes]
        reply = "不辣好菜：\n" + "\n".join(lines)
        state.last_dishes = [d.name for d in dishes]
        state.last_reply = reply
        return reply

    def _light(self, state):
        dishes = [d for d in self.kb.dishes if "清淡" in d.tags][:4]
        lines = [f"**{d.name}** {d.price}元" for d in (dishes or self.kb.dishes[:3])]
        reply = "清淡系🥗：\n" + "\n".join(lines)
        state.last_dishes = [d.name for d in (dishes or self.kb.dishes[:3])]
        state.last_reply = reply
        return reply

    def _follow_up(self, inp, state):
        t = inp.lower()
        exclude = set(state.last_dishes)

        # 口味约束更新（无论哪种follow-up模式都先检查）
        if "清淡" in t: state.taste = "清淡"
        if "不辣" in t or "不要辣" in t: state.taste = "不辣"

        if "换个" in t or "不要" in t or "换一个" in t:
            cands = [d for d in self.kb.dishes if d.name not in exclude]
            if state.taste == "清淡": cands = [d for d in cands if "清淡" in d.tags]
            elif state.taste == "不辣": cands = [d for d in cands if "不辣" in d.spice]
            recs = cands[:3] or [d for d in self.kb.dishes if d.name not in exclude][:3]
            lines = [f"**{d.name}** {d.price}元" for d in recs]
            state.last_dishes = [d.name for d in recs]
            return "换个口味：\n" + "\n".join(lines)
        if "还有" in t or "其他的" in t or "别的" in t or "再" in t:
            cands = [d for d in self.kb.dishes if d.name not in exclude]
            if state.taste == "清淡": cands = [d for d in cands if "清淡" in d.tags]
            elif state.taste == "不辣": cands = [d for d in cands if "不辣" in d.spice]
            recs = cands[:3] or self.kb.dishes[:3]
            lines = [f"**{d.name}** {d.price}元" for d in recs]
            state.last_dishes = [d.name for d in recs]
            label = f"（{state.taste}）" if state.taste else ""
            return f"再来几道{label}：\n" + "\n".join(lines)
        # 预算修改（优先于"改成"，避免误匹配人数）
        if "预算" in t:
            m = re.search(r"(\d+)", t)
            if m:
                state.budget = int(m.group(1))
                if state.people > 0: return self._by_budget(state)
                return f"好的，预算改成{state.budget}元！几个人吃？"
        # 人数修改（"改成X个人" 或 "改成两个人"）
        if "改成" in t or "改" in t:
            if "个人" in t or "人" in t:
                n = _parse_int(t)
                if n > 0:
                    state.people = n
                    if state.budget > 0: return self._by_budget(state)
                    return f"好的，改成{state.people}人了！有预算要求吗？"
        return self.invoke(inp, state, False)


class ReservationAgent:
    """订座 Agent。"""
    def __init__(self, llm):
        self.tool_agent = BookingToolAgent(llm)
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "你是饭小二订座助手。引导用户确认日期、人数、是否包间、电话。"),
            ("human", "{input}")
        ]) | llm | StrOutputParser()

    def invoke(self, inp: str, state: ConversationState) -> str:
        parts = []
        if state.people > 0: parts.append(f"{state.people}人")
        # 快速路径
        if parts:
            return f"好的，{','.join(parts)}！请告诉我具体日期时间和联系电话~"
        if any(w in inp for w in ["包间", "包厢"]):
            return self.tool_agent.invoke(inp, context=state.context_summary(), history=state.message_history)
        return "请告诉我：日期时间、人数、是否需要包间、联系电话~"


class ComplaintAgent:
    """投诉 Agent。"""
    FAST_COMPLAINTS = {
        "太咸": "抱歉！盐罐子打翻了😅 立刻安排重做一份，再送您一杯酸梅汤解腻~",
        "太淡": "收到！马上让厨房加把劲~重做一份马上来！",
        "难吃": "实在抱歉！您看：1.重做 2.换个菜 3.这道菜免单，您选哪个？",
        "不好吃": "实在抱歉！您看：1.重做 2.换个菜 3.这道菜免单，您选哪个？",
        "不新鲜": "非常抱歉！食材新鲜是底线，立刻给您重做+免单！",
        "等太久": "抱歉让您久等了！马上帮您催厨房，再送份小食垫垫~",
        "态度差": "非常抱歉！服务态度问题我们零容忍，马上安排值班经理来道歉！",
        "头发": "天呐！太抱歉了！立刻重做+这道菜免单+送甜品补偿！",
    }

    def __init__(self, llm):
        self.tool_agent = ComplaintToolAgent(llm)
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "饭小二处理投诉。真诚道歉+给方案。2句话内。"),
            ("human", "{input}")
        ]) | llm | StrOutputParser()

    def invoke(self, inp: str, state: ConversationState) -> str:
        # 快速路径：常见投诉直接回复
        t = inp.lower()
        for keyword, reply in self.FAST_COMPLAINTS.items():
            if keyword in t:
                return reply
        # LLM 兜底 → ToolAgent（LLM + Function Calling）
        return self.tool_agent.invoke(f"顾客说：{inp}", context=state.context_summary(), history=state.message_history)


class OrderAgent:
    """订单 Agent。"""
    def invoke(self, inp: str, state: ConversationState) -> str:
        return "您的菜品正在厨房精心制作中，稍等片刻美味就到！着急的话我帮您催一下~"


class HumanTransferAgent:
    """转人工 Agent：用户要求转接人工客服时触发。"""
    def invoke(self, inp: str, state: ConversationState) -> str:
        return (
            "好的，正在为您转接人工客服，请稍等~\n"
            "📞 也可直接拨打：400-888-6666\n"
            "⏰ 人工客服工作时间：9:00-22:00\n"
            "饭小二会一直在这，随时叫我回来也行哦~"
        )


class CustomerServiceAgent:
    """
    LangGraph 多Agent编排的客服入口。

    架构：
      RouterAgent（意图分类）
        → MenuQAAgent      （价格/菜单/菜品详情/营业信息）
        → RecommendAgent   （人群/过敏/预算/场景/口味推荐 + 多轮承接）
        → ReservationAgent （订座）
        → ComplaintAgent   （投诉）
        → OrderAgent       （订单查询）
        → LLM+RAG 兜底     （闲聊 / 无法归类）
    """

    def __init__(self):
        print("[CS] Init (multi-agent)...")
        self.router = RouterAgent()
        self.sessions = {}
        self.llm = ChatOpenAI(model="qwen-plus", temperature=0.7)

        # 初始化各专用 Agent
        self.menu_agent    = MenuQAAgent(self.llm)
        self.rec_agent     = RecommendAgent(self.llm)
        self.res_agent     = ReservationAgent(self.llm)
        self.comp_agent    = ComplaintAgent(self.llm)
        self.order_agent   = OrderAgent()
        self.human_agent   = HumanTransferAgent()

        # RAG 兜底链（闲聊 / 无法归类时使用）
        self.rag_chain = ChatPromptTemplate.from_messages([
            ("system", "你是饭小二，网红餐厅AI客服，幽默风趣。根据知识库回答，不编造。回复1-3句话，适当emoji。\n\n{context}"),
            ("human", "{input}")
        ]) | self.llm | StrOutputParser()

        # ── LangGraph 工作流 ──
        self._build_graph()

        print("[CS] Multi-agent ready")

    def _build_graph(self):
        """构建 LangGraph 状态图。"""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class GraphState(TypedDict):
            inp: str
            intent: str
            reply: str
            state_obj: object       # ConversationState
            is_follow: bool

        def router_node(state: GraphState) -> GraphState:
            cs = state["state_obj"]
            is_follow = cs.is_follow_up(state["inp"])
            if is_follow and cs.last_dishes:
                return {**state, "intent": "follow_recommend", "is_follow": True}
            # 新查询：重置约束再提取新约束
            cs.reset_constraints()
            cs.update_from(state["inp"])
            intent = self.router.invoke(state["inp"])
            cs.last_intent = intent
            return {**state, "intent": intent, "is_follow": False}

        def menu_node(state: GraphState) -> GraphState:
            reply = self.menu_agent.invoke(state["inp"], state["state_obj"])
            return {**state, "reply": reply}

        def recommend_node(state: GraphState) -> GraphState:
            reply = self.rec_agent.invoke(state["inp"], state["state_obj"], state["is_follow"])
            return {**state, "reply": reply}

        def reservation_node(state: GraphState) -> GraphState:
            reply = self.res_agent.invoke(state["inp"], state["state_obj"])
            return {**state, "reply": reply}

        def complaint_node(state: GraphState) -> GraphState:
            reply = self.comp_agent.invoke(state["inp"], state["state_obj"])
            return {**state, "reply": reply}

        def order_node(state: GraphState) -> GraphState:
            reply = self.order_agent.invoke(state["inp"], state["state_obj"])
            return {**state, "reply": reply}

        def human_node(state: GraphState) -> GraphState:
            reply = self.human_agent.invoke(state["inp"], state["state_obj"])
            return {**state, "reply": reply}

        def fallback_node(state: GraphState) -> GraphState:
            # 快速路径：常见问候/告别不走LLM
            fast = {
                "你好": "你好呀~饭小二在线营业中！今天想吃点啥？🍚",
                "您好": "您好呀~饭小二为您服务！想吃点啥？",
                "嗨": "嗨~饭小二来啦！想吃点啥？",
                "hello": "Hello~饭小二在线！想吃啥？",
                "在吗": "在的在的~饭小二24小时在线！",
                "谢谢": "不客气~好吃再来呀！😊",
                "谢谢你": "不客气~饭小二随时为您服务！",
                "再见": "下次再来呀~饭小二随时恭候！👋",
                "拜拜": "拜拜~下次见！👋",
                "886": "886~下次再来呀！",
            }
            t = state["inp"].strip().lower()
            if t in fast:
                return {**state, "reply": fast[t]}
            # FAQ 兜底：在 RAG 之前先查结构化 FAQ
            faq = match_faq(state["inp"])
            if faq and faq != "dynamic":
                return {**state, "reply": faq}
            ctx = retrieve_context(state["inp"], top_k=3)
            if not ctx:
                ctx = "（无相关知识库内容）"
            reply = self.rag_chain.invoke({"input": state["inp"], "context": ctx})
            return {**state, "reply": reply}

        def route_intent(state: GraphState) -> str:
            intent = state["intent"]
            if intent == "follow_recommend": return "recommend"
            if intent == "价格查询":   return "menu"
            if intent == "菜品推荐":   return "recommend"
            if intent == "点餐咨询":   return "menu"
            if intent == "订座服务":   return "reservation"
            if intent == "投诉建议":   return "complaint"
            if intent == "订单查询":   return "order"
            if intent == "转人工":    return "human"
            return "fallback"

        graph = StateGraph(GraphState)

        graph.add_node("router",      router_node)
        graph.add_node("menu",        menu_node)
        graph.add_node("recommend",   recommend_node)
        graph.add_node("reservation", reservation_node)
        graph.add_node("complaint",   complaint_node)
        graph.add_node("order",       order_node)
        graph.add_node("human",       human_node)
        graph.add_node("fallback",    fallback_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges("router", route_intent, {
            "recommend":    "recommend",
            "menu":         "menu",
            "reservation":  "reservation",
            "complaint":    "complaint",
            "order":        "order",
            "human":        "human",
            "fallback":     "fallback",
        })

        for node in ["menu", "recommend", "reservation", "complaint", "order", "human", "fallback"]:
            graph.add_edge(node, END)

        self.graph = graph.compile()

    def _get_state(self, sid: str) -> ConversationState:
        if sid not in self.sessions:
            self.sessions[sid] = ConversationState()
        return self.sessions[sid]

    def invoke(self, inp: str, sid: str = "default") -> str:
        cs = self._get_state(sid)
        result = self.graph.invoke({
            "inp": inp,
            "intent": "",
            "reply": "",
            "state_obj": cs,
            "is_follow": False,
        })
        reply = result.get("reply", "")
        cs.last_reply = reply
        # Save to conversation history (keep last 10 turns)
        cs.message_history.append((inp, reply))
        if len(cs.message_history) > 10:
            cs.message_history = cs.message_history[-10:]
        return reply
