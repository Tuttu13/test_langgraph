import argparse

from dotenv import load_dotenv
from models.state import ChatState

from workflow import build_graph

load_dotenv()  # .env を読み込む


def main() -> None:
    # parser = argparse.ArgumentParser(description="HotPepper 飲食店レコメンド Bot")
    # parser.add_argument(
    #     "--query", type=str, required=True, help="ユーザーの要望を自然文で入力"
    # )
    # args = parser.parse_args()

    args = "東京から仙台に旅行に行きます。"
    print("【ユーザー要望】", args)
    # LangGraph を起動
    graph = build_graph()
    init_state = ChatState(user_query=args)
    final_state = graph.invoke(init_state)

    print(final_state["response_text"])


if __name__ == "__main__":
    main()

# python -m src_3.main --query "仙台で美味しい"
