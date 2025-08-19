# main/core/providers/llm/rag/pipeline.py
from __future__ import annotations
from typing import List, Dict, Any, Callable, Iterable, Tuple
from config.logger import setup_logging
import os
import torch

from .timing import StageTimer, log_stage
from .vectorstore import load_embeddings, load_faiss
from .retriever import BaseRetriever, MQRetriever
from .reranker import Reranker
from .prompts import RAG_PROMPT_TEMPLATE

from .rerank_batcher import rerank_with_scores_batching  # 微批

logger = setup_logging()

class RAGPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg = cfg.get("rag", cfg)
        self.enabled = bool(cfg.get("enabled", True))
        self.k = int(cfg.get("k", 2))
        th = cfg.get("thresholds", {})
        self.mqr_low  = float(th.get("mqr_trigger_low", 0.0))
        self.mqr_high = float(th.get("mqr_trigger_high", 0.25))
        self.good_enough = float(th.get("good_enough", 0.25))
        paths = cfg.get("paths", {})
        self.emb_path = paths.get("embedding_model", "./models/bge-large-zh")
        self.faiss_dir = paths.get("faiss_index", "./data/faiss_index_DeepSeek")
        self.reranker_name = paths.get("reranker_model", "BAAI/bge-reranker-v2-m3")

        self.device = cfg.get("device", "auto")
        if self.device == "auto":
            self.runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.runtime_device = self.device

        logger.info(f"[Init_RAG] device selected: {self.runtime_device} "
                    f"(torch.cuda.is_available={torch.cuda.is_available()})")
        self.emb = load_embeddings(self.emb_path, device=self.device if self.device!="auto" else "cpu")
        self.vs  = load_faiss(self.faiss_dir, self.emb)
        use_cuda = self.runtime_device.startswith("cuda")
        self.reranker = Reranker(self.reranker_name, devices=[self.device] if self.device!="auto" else None)
        logger.info(f"[Init_RAG] reranker devices={self.device or 'cpu'} use_fp16={use_cuda}")

    def _avg(self, scores: List[float]) -> float | None:
        return sum(scores) / len(scores) if scores else None

    def route(self, query: str, mqr_llm: Dict[str, str] | None = None
             ) -> Tuple[str, List[Dict[str,str]] | None, Dict[str, float]]:
        timings: Dict[str, float] = {}

        with log_stage(logger, timings, "retrieve"):
            retr = BaseRetriever(self.vs, k=self.k)
            base_docs = retr.retrieve(query)

        with log_stage(logger, timings, "top1_probe"):
            probe = self.reranker.with_scores(query, base_docs, top_n=1)
            top1_probe = probe[0][1] if probe else float("-inf")
            logger.info(f"[RAG] top1(probe)={top1_probe:.3f} | band=[{self.mqr_low:.3f}, {self.mqr_high:.3f}]")

        if self.mqr_low <= top1_probe < self.mqr_high and mqr_llm:
            with log_stage(logger, timings, "multi_query"):
                mqr = MQRetriever(self.vs, llm_model=mqr_llm["model"],
                                  base_url=mqr_llm.get("base_url",""), api_key=mqr_llm.get("api_key",""), k=self.k)
                mqr_docs = mqr.retrieve(query)
                all_docs = list({id(d): d for d in list(base_docs)+list(mqr_docs)}.values())
        else:
            all_docs = base_docs

        with log_stage(logger, timings, "rerank"):
            use_mb = (os.getenv("RAG_RERANK_MICROBATCH", "1") == "1")
            ranked = None
            if use_mb:
                try:
                    ranked = rerank_with_scores_batching(
                        query=query,
                        docs=all_docs,
                        top_n=3,
                        model_name=self.reranker_name,
                        device=self.runtime_device,
                    )
                except Exception as e:
                    logger.exception(f"[RAG] micro-batch rerank failed, fallback to single-call: {e}")

            if ranked is None:
                ranked = self.reranker.with_scores(query, all_docs, top_n=3)

            scores = [s for _, s in ranked]
            top1 = scores[0] if scores else float("-inf")
            avg  = self._avg(scores) if scores else None
            avg_str = f"{avg:.3f}" if avg is not None else "n/a"
            logger.info(f"[RAG] scores after rerank: top1={top1:.3f}, avg={avg_str}, k={len(ranked)}")

        mode = "RAG" if (top1 >= self.good_enough) else "FUNCTION"
        logger.info(f"[RAG] thresholds: good_enough={self.good_enough:.3f} → mode={mode}")

        if mode != "RAG":
            return mode, None, timings

        with log_stage(logger, timings, "compose"):
            ctx = "\n\n".join([getattr(d, "page_content", "") for d, _ in ranked])
            sys = RAG_PROMPT_TEMPLATE.format(context=f"\n{ctx}\n", question=query, is_knowledge=True)
            msgs = [{"role": "system", "content": sys}]
        return mode, msgs, timings

    def stream(self, dialogue: List[Dict[str,str]],
               stream_fn: Callable[[List[Dict[str,str]]], Iterable[str]],
               mqr_llm: Dict[str,str] | None = None) -> Iterable[str]:
        query = (dialogue[-1].get("content") if dialogue else "").strip()
        mode, sys_msgs, timings = self.route(query, mqr_llm=mqr_llm)

        if mode != "RAG" or not sys_msgs:
            with log_stage(logger, timings, "generate_passthrough"):
                for s in stream_fn(dialogue):
                    yield s
            summary = " | ".join([f"{k}={v:.3f}s" for k, v in timings.items()])
            logger.info(f"[RAG] summary | {summary}")
            return

        messages = sys_msgs + dialogue
        with log_stage(logger, timings, "generate"):
            for s in stream_fn(messages):
                yield s

        summary = " | ".join([f"{k}={v:.3f}s" for k, v in timings.items()])
        logger.info(f"[RAG] summary | {summary}")
