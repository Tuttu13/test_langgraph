from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchParams(BaseModel):
    """ユーザー入力から抽出した検索条件（地名/駅名のみ）"""

    area: Optional[str] = Field(None, description="地名または駅名")


class Restaurant(BaseModel):
    id: str
    name: str
    address: str
    genre: str
    budget: str
    url: str
    catch: str


class ChatState(BaseModel):
    """LangGraph 全体で共有するステート"""

    user_query: str = Field(..., description="ユーザー発話の原文")
    search_params: SearchParams = Field(default_factory=SearchParams)
    restaurants: Annotated[List[Restaurant], operator.add] = Field(default_factory=list)
    pending_question: Optional[str] = Field(
        None, description="足りない情報を尋ねるための質問"
    )
    response_text: Optional[str] = Field(None, description="ユーザーへ返すメッセージ")
