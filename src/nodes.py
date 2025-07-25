# nodes.py
# グラフの各ノード（関数）をまとめる

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_config import llm
from roles import ROLES
from state import Judgement, State


def selection_node(state: State) -> dict[str, Any]:
    """ユーザー質問から最適ロールを 1,2,3 で選択"""
    role_options = "\n".join(
        f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()
    )
    prompt = ChatPromptTemplate.from_template(
        """質問を分析し、最も適切な回答担当ロールを選択してください。

        選択肢:
        {role_options}

        回答は選択肢の番号（1、2、または3）のみを返してください。

        質問: {query}
        """
    )
    chain = (
        prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    )
    role_number = chain.invoke(
        {"role_options": role_options, "query": state.messages}
    ).strip()
    return {"current_role": ROLES[role_number]["name"]}


def answering_node(state: State) -> dict[str, Any]:
    """選択されたロールとして回答を生成"""
    role_details = "\n".join(f"- {v['name']}: {v['details']}" for v in ROLES.values())

    prompt = ChatPromptTemplate.from_template(
        """あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。

        役割の詳細:
        {role_details}

        質問: {query}

        回答:
        """
    )

    chain = prompt | llm | StrOutputParser()

    # ── 直近のユーザーメッセージを取得 ──
    last_user_msg: HumanMessage = next(
        msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)
    )
    question_text = last_user_msg.content

    answer_text: str = chain.invoke(
        {
            "role": state.current_role,
            "role_details": role_details,
            "query": question_text,
        }
    )

    # ── AIMessage で包んで返す ──
    return {"messages": [AIMessage(content=answer_text)]}


def check_node(state: State) -> dict[str, Any]:
    """回答品質をチェックし、合否と理由を返す"""
    prompt = ChatPromptTemplate.from_template(
        """以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。
        また、その判断理由も説明してください。

        ユーザーからの質問: {query}
        回答: {answer}
        """
    )
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke(
        {"query": state.messages, "answer": state.messages[-1]}
    )
    return {"current_judge": result.judge, "judgement_reason": result.reason}
