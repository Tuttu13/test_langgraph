import os
from typing import Any, Dict, List

import requests
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from models.state import ChatState, Restaurant, SearchParams

# --- 1) ユーザー発話を構造化して検索パラメータを作る ----------------------------
_llm_struct = ChatOpenAI(model="gpt-4o", temperature=0.0)
_param_parser = PydanticOutputParser(pydantic_object=SearchParams)

parse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは飲食店提案ボットの入力を解析します。"
            "次のユーザー発話から『地名や駅名(area)』だけを JSON で抽出してください。",
        ),
        ("human", "{user_query}"),
    ]
)
parse_chain = parse_prompt | _llm_struct | _param_parser


def parse_user(state: ChatState) -> Dict[str, Any]:
    params: SearchParams = parse_chain.invoke({"user_query": state.user_query})
    # 足りない情報があれば追加質問を生成
    missing = [f"『{f}』" for f, v in params.dict().items() if v is None]
    if missing:
        question = " ".join(missing) + " を教えてください。"
        return {"search_params": params, "pending_question": question}
    return {"search_params": params}


# --- 2) Hot Pepper API で飲食店検索 ------------------------------------------
_HOTPEPPER_ENDPOINT = "http://webservice.recruit.co.jp/hotpepper/gourmet/v1/"


def _call_hotpepper(
    params: SearchParams, *, lunch_flag: int, count: int = 10
) -> List[Restaurant]:
    key = os.getenv("HOTPEPPER_API_KEY")
    if key is None:
        raise RuntimeError("HOTPEPPER_API_KEY が .env に設定されていません")
    payload = {
        "key": key,
        "format": "json",
        "count": count,
        "lunch": lunch_flag,
    }
    if params.area:
        payload["keyword"] = params.area

    res = requests.get(_HOTPEPPER_ENDPOINT, params=payload, timeout=10)
    res.raise_for_status()
    data = res.json()
    shops = data.get("results", {}).get("shop", [])
    return [
        Restaurant(
            id=s["id"],
            name=s["name"],
            address=s["address"],
            genre=s["genre"]["name"],
            budget=s["budget"]["average"] or "",
            url=s["urls"]["pc"],
            catch=s["catch"],
        )
        for s in shops
    ]


def fetch_restaurants(state: ChatState) -> Dict[str, Any]:
    if state.pending_question:
        return {}

    # ★ 2 回呼び出してそれぞれ 3 件だけ使う
    lunch_shops = _call_hotpepper(state.search_params, lunch_flag=1)[:3]
    dinner_shops = _call_hotpepper(state.search_params, lunch_flag=0)[:3]

    return {
        "lunch_restaurants": lunch_shops,
        "dinner_restaurants": dinner_shops,
    }


# --- 3) レスポンス生成 ------------------------------------------------------
_llm_answer = ChatOpenAI(model="gpt-4o", temperature=0.3)
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは親切なグルメ案内ボットです。出力は必ず日本語。"),
        (
            "human",
            "【エリア】{area}\n\n"
            "▼ランチ候補(3件)\n{lunch_shops}\n\n"
            "▼ディナー候補(3件)\n{dinner_shops}\n\n"
            "それぞれをユーザーにわかりやすく紹介してください。\n"
            "Powered by ホットペッパーグルメ Webサービス",
        ),
    ]
)
answer_chain = answer_prompt | _llm_answer | StrOutputParser()


def generate_answer(state: ChatState) -> Dict[str, Any]:
    # ❶ 追加質問がある場合は先に返す
    if state.pending_question:
        return {"response_text": state.pending_question, "pending_question": None}

    # ❷ ヒット件数に応じて分岐
    lunch_hits = len(state.lunch_restaurants)
    dinner_hits = len(state.dinner_restaurants)

    if lunch_hits == 0 and dinner_hits == 0:
        # 両方ゼロ
        return {
            "response_text": "条件に合うお店が見つかりませんでした。他の条件で試しますか？"
        }

    if lunch_hits == 0:
        # ランチだけゼロ
        msg = (
            "ランチ条件では見つかりませんでした。ディナーのおすすめをご案内します。\n\n"
        )
    elif dinner_hits == 0:
        # ディナーだけゼロ
        msg = (
            "ディナー条件では見つかりませんでした。ランチのおすすめをご案内します。\n\n"
        )
    else:
        msg = ""  # 両方ヒットした場合はメッセージ不要

    # ❸ 要約作成
    def _fmt(r: Restaurant) -> str:
        return (
            f"- {r.name}｜{r.genre}｜{r.budget or '予算情報なし'}｜{r.address}｜{r.url}"
        )

    lunch_summary = "\n".join(_fmt(r) for r in state.lunch_restaurants)
    dinner_summary = "\n".join(_fmt(r) for r in state.dinner_restaurants)

    text = answer_chain.invoke(
        {
            "area": state.search_params.area or "（エリア未指定）",
            "lunch_shops": lunch_summary or "該当なし",
            "dinner_shops": dinner_summary or "該当なし",
        }
    )

    return {"response_text": msg + text}
