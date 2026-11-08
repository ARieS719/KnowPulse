# -*- coding: utf-8 -*-
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import get_settings
from src.generation.rag_chain import KnowPulseRAGChain
from src.indexing.dual_writer import AtomicDualIndexWriter, run_full_indexing
from src.indexing.embedder import CachedBgeM3Embedder
from src.llm.siliconflow_client import SiliconFlowService
from src.retrieval.hybrid_search import HybridRetrieverService

s = get_settings()

app = FastAPI(
    title="KnowPulse (EduGuard-RAG) 企业级中文教育知识库 API",
    version="1.2.0",
    description="支持入库质量门审计、Chroma+FTS5 原子双写、ACL 前置混合检索重排与 1ms 在线引用硬校验"
)

writer = AtomicDualIndexWriter()
embedder = CachedBgeM3Embedder(writer.sm.conn, mode=s.effective_embed_mode)
sf_service = SiliconFlowService()
retriever = HybridRetrieverService(writer, embedder, rerank_mode=s.effective_rerank_mode)
rag_chain = KnowPulseRAGChain(sf_service)


class QueryRequest(BaseModel):
    question: str
    user_role: str = "student"          # "student" | "teacher"
    subject: Optional[str] = None
    ablation_mode: str = "hybrid_rerank"


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str                          # "thumbs_up" | "thumbs_down"
    comment: Optional[str] = ""


class IngestRequest(BaseModel):
    """增量入库：对 raw 目录中新放入/已变更的文档执行 质量门 -> 解析 -> 双写。"""
    doc_path: Optional[str] = None       # 相对于 data/raw 的路径；不传则扫描 raw 全量
    run_quality_gate: bool = True


@app.get("/health", summary="健康检查")
def health():
    return {
        "status": "ok",
        "run_mode": s.run_mode,
        "embed_mode": embedder.mode,
        "rerank_mode": retriever.reranker.mode,
        "has_llm_api_key": s.has_siliconflow_key,
    }


@app.post("/query", summary="混合检索 + 重排 + 强制引用问答")
def query_knowledge_base(req: QueryRequest):
    ret = retriever.retrieve(
        query=req.question,
        user_role=req.user_role,
        subject=req.subject,
        ablation_mode=req.ablation_mode
    )
    gen = rag_chain.generate_and_verify(req.question, ret["chunks"])

    # 若回答有效且未拒答，写入热点语义缓存（容量上限由 SemanticCache 控制）
    if not ret.get("cache_hit") and not gen["refused"] and "query_vec" in ret:
        retriever.semantic_cache.add(
            ret["query_vec"],
            req.user_role,
            {**gen, "retrieval_latency_ms": ret["latency_ms"]},
        )

    return {
        "question": req.question,
        "user_role": req.user_role,
        "retrieval_latency_ms": ret["latency_ms"],
        "cache_hit": ret.get("cache_hit", False),
        **gen
    }


@app.post("/ingest", summary="增量入库（质量门 -> 解析 -> 原子双写）")
def ingest_documents(req: IngestRequest):
    from src.quality.quality_gate import run_quality_gate_pipeline

    if req.run_quality_gate:
        run_quality_gate_pipeline(
            s.data_root, s.manifest_csv, str(s.metadata_db_path),
            str(s.quality_gate_report_path), "data/quarantine/quarantine_list.csv"
        )
    run_full_indexing(s.data_root, embed_mode=embedder.mode)
    return {
        "status": "done",
        "indexed_docs": writer.sm.conn.cursor().execute(
            "SELECT COUNT(*) AS c FROM doc_versions WHERE status='INDEXED'").fetchone()["c"],
        "fts_chunks": writer.fts.count(),
    }


@app.get("/quality/summary", summary="查看语料质量门拦截与版本状态统计")
def get_quality_summary():
    cur = writer.sm.conn.cursor()
    status_rows = cur.execute("SELECT status, COUNT(*) as cnt FROM doc_versions GROUP BY status").fetchall()
    tag_rows = cur.execute("SELECT detected_tag, COUNT(*) as cnt FROM quality_audit_log GROUP BY detected_tag").fetchall()
    return {
        "doc_version_states": {r["status"]: r["cnt"] for r in status_rows},
        "detected_quality_tags": {r["detected_tag"]: r["cnt"] for r in tag_rows},
    }


@app.get("/cache/stats", summary="语义缓存命中统计")
def get_cache_stats():
    return retriever.cache_stats()


@app.post("/feedback", summary="用户点踩回流至 Badcase 库")
def submit_feedback(req: FeedbackRequest):
    if req.rating == "thumbs_down":
        badcase_file = s.badcases_path
        badcase_file.parent.mkdir(parents=True, exist_ok=True)
        with open(badcase_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({**req.model_dump(), "created_at": datetime.now().isoformat()}, ensure_ascii=False) + "\n")
    return {"status": "recorded"}
