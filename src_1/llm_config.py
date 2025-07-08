# llm_config.py
# LLM を 1 か所で初期化し、ノード側から import して使う

from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

# ここでモデル・温度など共通設定
_base_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# max_tokens だけはノード側でオンデマンド変更できるようにする
llm = _base_llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))
