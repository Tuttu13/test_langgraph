import operator
from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class State(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add]
    current_role: str = Field(default="", description="選定された回答ロール")
    current_judge: bool = Field(default=False, description="品質チェックの結果")
    judgement_reason: str = Field(default="", description="品質チェックの判定理由")


class Judgement(BaseModel):
    judge: bool = Field(False, description="判定結果")
    reason: str = Field("", description="判定理由")
