import operator
from typing import Annotated, List

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class State(BaseModel):
    """チャットスレッドの状態を保持。"""

    query: str
    messages: Annotated[List[BaseMessage], operator.add] = Field(default=[])
