from __future__ import annotations
from typing import List, Tuple, Any
from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", devices=None, use_fp16=True):
        self.rr = FlagReranker(model_name, devices=devices or ["cuda:0"], use_fp16=use_fp16)

    def with_scores(self, query: str, docs: List[Any], top_n: int | None = 3) -> List[Tuple[Any, float]]:
        if not docs: return []
        pairs = [(query, getattr(d, "page_content", str(d))) for d in docs]
        scores = self.rr.compute_score(pairs, normalize=True)
        try: scores = list(scores)
        except Exception: pass
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        if top_n is not None: order = order[:max(0, int(top_n))]
        return [(docs[i], float(scores[i])) for i in order]

    def only(self, query: str, docs: List[Any], top_n: int | None = 3) -> List[Any]:
        return [d for (d, _) in self.with_scores(query, docs, top_n=top_n)]
