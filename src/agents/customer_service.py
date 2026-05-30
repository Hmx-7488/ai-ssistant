# -*- coding: utf-8 -*-
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from src.agents.router_agent import RouterAgent
from src.tools.unified_kb import get_unified_kb
from src.tools.vector_store import retrieve_context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ConversationState:
    """Multi-turn conversation state to track user constraints."""

    FOLLOW_UP = ["还有","换个","不要","刚才","改成","重新","再来","其他的","别的","换一个","不要这个"]

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

    def update_from(self, text: str):
        """Extract constraints from user text."""
        m = re.search(r"(\d+)\s*个人?", text)
        if m:
            self.people = int(m.group(1))
        m = re.search(r"(\d+)\s*元", text)
        if m:
            self.budget = int(m.group(1))
        for k in ["儿童","小孩","孩子","baby","婴儿"]:
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
        allergen_map = {"花生":"花生","海鲜":"海鲜","鸡蛋":"鸡蛋","牛肉":"牛肉","乳制品":"乳制品","麸质":"麸质","虾":"虾","鱼类":"鱼类"}
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
        """Check if this is a follow-up to the previous turn."""
        t = text.strip()
        if len(t) <= 8:
            for w in self.FOLLOW_UP:
                if w in t:
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
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "你是饭小二餐厅的菜品问答专家。根据知识库回答，不编造。回复简洁，1-2句。"),
            ("human", "{input}")
        ]) | llm | StrOutputParser()

    def invoke(self, inp: str, state: ConversationState) -> str:
        t = inp.lower()
        # ── 确定性快速路径（不调用 LLM） ──
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
        # 辣度/定制查询
        if "辣" in t and "烤鱼" in t:
            return "秘制烤鱼可选微辣/中辣/特辣，跟服务员说一声就行~"
        if "辣" in t and not any(w in t for w in ["推荐","菜单","过敏","预算"]):
            d = self.kb.find_dish(inp)
            if d and d.customizable:
                return f"{d.name}可以调辣度，跟服务员说就行~"
        if "不要" in inp and "香菜" in inp:
            return "可以！下单时跟服务员说不要香菜，厨房会单独处理~"
        # 菜品详情查询
        d = self.kb.find_dish(inp)
        if d:
            state.last_dishes = [d.name]
            return self.kb.format_dish_info(d)
        # 模糊指代 → 反问
        if any(w in t for w in ["那个","这个","多少钱"]) and "人均" not in t:
            return "想问哪道菜价格？说个菜名我帮你查~"
        # RAG 兜底
        ctx = retrieve_context(inp, top_k=3)
        return self.chain.invoke({"input": inp + ("\n参考：" + ctx if ctx else "")})

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
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "你是饭小二，幽默的餐厅推荐专家。根据菜品信息推荐，突出亮点，带emoji。"),
            ("human", "{input}")
        ]) | llm | StrOutputParser()

    def invoke(self, inp: str, state: ConversationState, is_follow: bool) -> str:
        if is_follow and state.last_dishes:
            return self._follow_up(inp, state)
        if state.people_type:
            return self._for_people(state)
        if state.allergies:
            return self._allergen_safe(state)
        if state.budget > 0 and state.people > 0:
            return self._by_budget(state)
        if state.scene:
            return self._by_scene(state)
        if state.taste == "不辣":
            return self._non_spicy(state)
        if state.taste == "清淡":
            return self._light(state)
        # 默认招牌推荐
        sig = self.kb.get_dishes_by_tag("招牌")
        lines = [f"**{d.name}** {d.price}元 - {d.description[:25]}..." for d in sig[:3]]
        reply = "推荐招牌菜：\n" + "\n".join(lines)
        state.last_dishes = [d.name for d in sig[:3]]
        state.last_reply = reply
        return reply

    def _for_people(self, state):
        info = self.kb.recommend_for_people(state.people_type)
        if not info:
            ctx = retrieve_context(f"推荐{state.people_type}吃的菜")
            return self.chain.invoke({"input": f"推荐{state.people_type}吃的菜\n{ctx}"})
        recs = info.get("推荐", [])
        note = info.get("说明", "")
        dishes = [f"**{d.name}** {d.price}元 - {d.description[:30]}..."
                  for n in recs[:4] if (d := self.kb.get_dish_by_name(n))]
        state.last_dishes = recs[:4]
        reply = f"给{state.people_type}推荐~\n" + "\n".join(dishes) + (f"\n({note})" if note else "")
        state.last_reply = reply
        return reply

    def _allergen_safe(self, state):
        safe = list(self.kb.dishes)
        for a in state.allergies:
            if not a.startswith("不吃"):
                safe = [d for d in safe if a not in d.allergens]
        recs = safe[:4] or self.kb.dishes[:3]
        lines = [f"**{d.name}** {d.price}元" for d in recs]
        reply = f"避开{','.join(state.allergies)}，推荐：\n" + "\n".join(lines)
        state.last_dishes = [d.name for d in recs]
        state.last_reply = reply
        return reply

    def _by_budget(self, state):
        pp = state.budget / state.people
        for s in self.kb.suites:
            if (state.people <= 2 and "双人" in s["name"]) or \
               (3 <= state.people <= 4 and "四人" in s["name"]) or \
               (state.people == 1 and "一人" in s["name"]):
                reply = f"{state.people}人{state.budget}元，推荐 **{s['name']}**（{s['price']}元）：\n{', '.join(s['items'])}\n{s['description']}，省{s['original_price']-s['price']}元~"
                state.last_dishes = s["items"]
                state.last_reply = reply
                return reply
        # 自由搭配
        affordable = sorted(self.kb.dishes, key=lambda d: d.price)
        combo, total = [], 0
        for d in affordable:
            if total + d.price <= state.budget:
                combo.append(d); total += d.price
            if len(combo) >= state.people + 1: break
        if combo:
            lines = [f"**{d.name}** {d.price}元" for d in combo]
            reply = f"{state.people}人{state.budget}元搭配：\n" + "\n".join(lines) + f"\n共{total}元，余{state.budget-total}元加饮品~"
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
        return self.chain.invoke({"input": f"{state.scene}推荐菜品\n{ctx}"})

    def _non_spicy(self, state):
        dishes = [d for d in self.kb.dishes if "不辣" in d.spice][:4]
        lines = [f"**{d.name}** {d.price}元" for d in dishes]
        reply = "不辣好菜：\n" + "\n".join(lines)
        state.last_dishes = [d.name for d in dishes]
        state.last_reply = reply
        return reply

    def _light(self, state):
        dishes = [d for d in self.kb.dishes if "清淡" in d.tags or "不辣" in d.spice][:4]
        lines = [f"**{d.name}** {d.price}元" for d in (dishes or self.kb.dishes[:3])]
        reply = "清淡系：\n" + "\n".join(lines)
        state.last_dishes = [d.name for d in (dishes or self.kb.dishes[:3])]
        state.last_reply = reply
        return reply

    def _follow_up(self, inp, state):
        t = inp.lower()
        exclude = set(state.last_dishes)
        if "换个" in t or "不要" in t or "换一个" in t:
            if "清淡" in t: state.taste = "清淡"
            if "不辣" in t or "不要辣" in t: state.taste = "不辣"
            cands = [d for d in self.kb.dishes if d.name not in exclude]
            if state.taste == "清淡": cands = [d for d in cands if "清淡" in d.tags or "不辣" in d.spice]
            elif state.taste == "不辣": cands = [d for d in cands if "不辣" in d.spice]
            recs = cands[:3] or [d for d in self.kb.dishes if d.name not in exclude][:3]
            lines = [f"**{d.name}** {d.price}元" for d in recs]
            state.last_dishes = [d.name for d in recs]
            return "换个口味：\n" + "\n".join(lines)
        if "还有" in t or "其他的" in t or "别的" in t:
            cands = [d for d in self.kb.dishes if d.name not in exclude][:3] or self.kb.dishes[:3]
            lines = [f"**{d.name}** {d.price}元" for d in cands]
            state.last_dishes = [d.name for d in cands]
            return "再来几道：\n" + "\n".join(lines)
        if "改成" in t:
            m = re.search(r"(\d+)\s*个人?", t)
            if m:
                state.people = int(m.group(1))
                if state.budget > 0: return self._by_budget(state)
                return f"好的，改成{state.people}人了！有预算要求吗？"
        return self.invoke(inp, state, False)


class ReservationAgent:
    """订座 Agent。"""
    def __init__(self, llm):
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "你是饭小二订座助手。引导用户确认日期、人数、是否包间、电话。"),
            ("human", "{input}")
        ]) | llm | StrOutputParser()

    def invoke(self, inp: str, state: ConversationState) -> str:
        parts = []
        if state.people > 0: parts.append(f"{state.people}人")
        if parts:
            return f"好的，{','.join(parts)}！请告诉我具体日期时间和联系电话~"
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
        # LLM 兜底（不调 RAG，投诉场景不需要知识库）
        return self.chain.invoke({"input": f"顾客说：{inp}"})


class OrderAgent:
    """订单 Agent。"""
    def invoke(self, inp: str, state: ConversationState) -> str:
        return "您的菜品正在厨房精心制作中，稍等片刻美味就到！着急的话我帮您催一下~"


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
            if not is_follow:
                cs.update_from(state["inp"])
            # 多轮承接直接走推荐，不再走意图分类
            if is_follow and cs.last_dishes:
                return {**state, "intent": "follow_recommend", "is_follow": True}
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
            return "fallback"

        graph = StateGraph(GraphState)

        graph.add_node("router",      router_node)
        graph.add_node("menu",        menu_node)
        graph.add_node("recommend",   recommend_node)
        graph.add_node("reservation", reservation_node)
        graph.add_node("complaint",   complaint_node)
        graph.add_node("order",       order_node)
        graph.add_node("fallback",    fallback_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges("router", route_intent, {
            "recommend":    "recommend",
            "menu":         "menu",
            "reservation":  "reservation",
            "complaint":    "complaint",
            "order":        "order",
            "fallback":     "fallback",
        })

        for node in ["menu", "recommend", "reservation", "complaint", "order", "fallback"]:
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
        return reply
