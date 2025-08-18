from __future__ import annotations
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
# MultiQueryRetriever 生成多视角查询并取并集。:contentReference[oaicite:9]{index=9}

class BaseRetriever:
    def __init__(self, vectorstore, k: int = 2):
        self.vs, self.k = vectorstore, k
    def retrieve(self, query: str):
        try:
            return self.vs.as_retriever(search_kwargs={"k": self.k}).invoke(query)
        except Exception:
            return []

class MQRetriever(BaseRetriever):
    def __init__(self, vectorstore, llm_model: str, base_url: str, api_key: str, k: int = 2):
        super().__init__(vectorstore, k)
        self.llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0.0)
        self._mqr = MultiQueryRetriever.from_llm(retriever=self.vs.as_retriever(search_kwargs={"k": self.k}),
                                                 llm=self.llm, include_original=True)
    def retrieve(self, query: str):
        try: return self._mqr.invoke(query)
        except Exception: return super().retrieve(query)
