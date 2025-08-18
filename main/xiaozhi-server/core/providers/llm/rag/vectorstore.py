from __future__ import annotations
import os, pathlib, logging
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# HuggingFaceEmbeddings 用法与本地路径传 model_name。:contentReference[oaicite:7]{index=7}
# FAISS load_local 的 allow_dangerous_deserialization 说明。:contentReference[oaicite:8]{index=8}

log = logging.getLogger("rag.vectorstore")

_EMB = None
_DB  = None

def load_embeddings(model_path: str, device: str = "cpu"):
    global _EMB
    if _EMB is None:
        _EMB = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"device": device})
        log.info("Embeddings loaded: %s", model_path)
    return _EMB

def load_faiss(index_dir: str, embeddings) -> Optional[FAISS]:
    global _DB
    if _DB is None:
        p = pathlib.Path(index_dir)
        if not p.exists():
            log.warning("FAISS index dir not found: %s", p); return None
        _DB = FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)
        log.info("FAISS loaded: %s", p)
    return _DB
