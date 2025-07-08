import operator
from typing import Annotated

from pydantic import BaseModel, Field


class State(BaseModel):
    query: str = Field(..., description="ユーザーからの質問")
    current_role: str = Field(default="", description="選定された回答ロール")
    messages: Annotated[list[str], operator.add] = Field(
        default=[], description="回答履歴"
    )
    current_judge: bool = Field(default=False, description="品質チェックの結果")
    judgement_reason: str = Field(default="", description="品質チェックの判定理由")


class Judgement(BaseModel):
    judge: bool = Field(False, description="判定結果")
    reason: str = Field("", description="判定理由")
