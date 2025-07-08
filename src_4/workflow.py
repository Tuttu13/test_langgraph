from langgraph.graph import END, StateGraph
from models.state import ChatState

from nodes import fetch_restaurants, generate_answer, parse_user


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("parse_user", parse_user)
    g.add_node("fetch", fetch_restaurants)
    g.add_node("answer", generate_answer)

    g.set_entry_point("parse_user")
    g.set_finish_point("answer")

    g.add_edge("parse_user", "fetch")
    g.add_edge("fetch", "answer")
    # 「足りない情報がある場合」は answer → parse_user へ戻す
    g.add_conditional_edges(
        "answer",
        lambda s: s.response_text and "教えてください" in s.response_text,
        {True: "parse_user", False: END},
    )
    return g.compile()
