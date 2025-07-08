import os
from typing import Any, Dict, List

import requests
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import ChatState, Restaurant, SearchParams

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


def _call_hotpepper(params: SearchParams, count: int = 10) -> List[Restaurant]:
    key = os.getenv("HOTPEPPER_API_KEY")
    if key is None:
        raise RuntimeError("HOTPEPPER_API_KEY が .env に設定されていません")
    payload = {
        "key": key,
        "format": "json",
        "count": count,
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
        # まだ確認したいことが残っている
        return {}
    shops = _call_hotpepper(state.search_params)
    return {"restaurants": shops}


# --- 3) レスポンス生成 ------------------------------------------------------
_llm_answer = ChatOpenAI(model="gpt-4o", temperature=0.3)
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは親切なグルメ案内ボットです。出力は必ず日本語。"),
        (
            "human",
            "以下のエリアでおすすめできる飲食店を紹介してください。\n\n"
            "【エリア】{area}\n\n"
            "【検索結果 (最大10件)】\n{shops}\n\n"
            "上位5件をピックアップし、住所・ジャンル・予算・予約 URL を含めて要約してください。\n"
            "Powered by ホットペッパーグルメ Webサービス",
        ),
    ]
)
answer_chain = answer_prompt | _llm_answer | StrOutputParser()


def generate_answer(state: ChatState) -> Dict[str, Any]:
    if state.pending_question:
        # 足りない情報を質問
        return {"response_text": state.pending_question, "pending_question": None}

    if not state.restaurants:
        return {
            "response_text": "条件に合うお店が見つかりませんでした。他の条件で試しますか？"
        }

    shop_summary = "\n".join(
        f"- {r.name}｜{r.genre}｜{r.budget or '予算情報なし'}｜{r.address}｜{r.url or ''}"
        for r in state.restaurants
    )
    text = answer_chain.invoke(
        {
            "area": state.search_params.area or "（エリア未指定）",
            "shops": shop_summary,
        }
    )
    return {"response_text": text}
