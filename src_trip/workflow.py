# workflow.py
# LangGraph を組み立てて compile する

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes import answering_node, check_node, selection_node
from state import State


def build_workflow() -> StateGraph:
    workflow = StateGraph(State)
    workflow.add_node("selection", selection_node)
    workflow.add_node("answering", answering_node)
    workflow.add_node("check", check_node)

    # フロー定義
    workflow.set_entry_point("selection")
    workflow.add_edge("selection", "answering")
    workflow.add_edge("answering", "check")

    # 品質チェックに応じた分岐
    workflow.add_conditional_edges(
        "check", lambda s: s.current_judge, {True: END, False: "selection"}
    )
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
