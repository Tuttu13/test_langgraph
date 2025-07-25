import operator
from typing import Annotated, TypedDict

from IPython.display import Image, display
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ここでモデル・温度など共通設定
model = ChatOpenAI(model="gpt-4o", temperature=0.0)


# messageを作成する
message = [
    SystemMessage(
        content="あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"
    ),
    MessagesPlaceholder("messages"),
]

# messageからプロンプトを作成
prompt = ChatPromptTemplate.from_messages(message)

# chainとgraphを作成
chain = prompt | model
# 4セル目


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def call_llm(state: GraphState):
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def create_langgraph():

    workflow = StateGraph(state_schema=GraphState)
    workflow.add_node("model", call_llm)

    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph


def run_chat():
    graph = create_langgraph()

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

        response = graph.invoke(
            {"messages": input_query}, config={"configurable": {"thread_id": "12345"}}
        )

        print("=================================")
        print("AIの回答", response["messages"][-1].content)


if __name__ == "__main__":
    run_chat()
