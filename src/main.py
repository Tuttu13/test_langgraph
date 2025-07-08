# main.py
# 実際にワークフローを動かすエントリーポイント

from state import State
from workflow import build_workflow


def run_sync(query: str):
    workflow = build_workflow()
    result_state: State = workflow.invoke(State(query=query))
    print("----- 回答 -----")
    print(result_state["messages"][0])
    print("\n品質 OK?:", result_state["current_judge"])
    if not result_state["current_judge"]:
        print("理由:", result_state["judgement_reason"])


if __name__ == "__main__":
    run_sync("生成AIについて教えてください")
