# main.py
# 実際にワークフローを動かすエントリーポイント

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from state import State
from workflow import build_workflow


def run_sync():
    workflow = build_workflow()

    while True:
        query = input("質問を入力してください: ")

        if query.lower() in ["exit", "quit"]:
            print("終了します。")
            break

        print("=================================")
        print("質問:", query)

        input_query = [
            HumanMessage(
                [
                    {"type": "text", "text": f"{query}"},
                ]
            )
        ]

        result_state: State = workflow.invoke(
            {"messages": input_query}, config={"configurable": {"thread_id": "12345"}}
        )
        print("----- 回答 -----")
        print(result_state["messages"][-1].content)
        print("\n品質 OK?:", result_state["current_judge"])
        if not result_state["current_judge"]:
            print("理由:", result_state["judgement_reason"])


if __name__ == "__main__":
    run_sync()
