# -*- coding: utf-8 -*-
import time
from typing import Dict, List, Optional
import numpy as np

from src.config import get_settings
from src.indexing.dual_writer import AtomicDualIndexWriter
from src.indexing.embedder import CachedBgeM3Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.semantic_cache import SemanticCache


class HybridRetrieverService:
    """
    四档消融模式 (ablation_mode: 'vec_only' | 'fts_only' | 'hybrid_rrf' | 'hybrid_rerank')
    + 双运行模式重排（本地 CrossEncoder 或 硅基流动 /v1/rerank API）
    + ACL 前置过滤（Chroma where 与 FTS5 access_roles 同步生效）
    + 语义缓存（热点查询 <20ms）
    """
    def __init__(self, writer: AtomicDualIndexWriter, embedder: CachedBgeM3Embedder,
                 reranker: Optional[Reranker] = None, rerank_mode: str | None = None):
        s = get_settings()
        self.writer = writer
        self.embedder = embedder
        self.reranker = reranker or Reranker(mode=rerank_mode or s.effective_rerank_mode)
        sc = s.retrieval.semantic_cache
        self.semantic_cache = SemanticCache(max_size=sc.max_size, threshold=sc.threshold, enabled=sc.enabled)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, user_role: str = "student", subject: Optional[str] = None,
                 ablation_mode: str = "hybrid_rerank", top_k_final: int | None = None) -> Dict:
        s = get_settings()
        top_k_final = top_k_final or s.retrieval.final_top_k
        t0 = time.perf_counter()
        q_vec = self.embedder.encode([query])[0]
        q_arr = np.asarray(q_vec, dtype=np.float32)

        # 0. 热点语义缓存（余弦相似度 >= 阈值直接命中）
        cached = self.semantic_cache.lookup(q_arr, user_role)
        if cached is not None:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            if "retrieval_latency_ms" in cached:
                cached["retrieval_latency_ms"] = ms
            else:
                cached["latency_ms"] = ms
            cached["cache_hit"] = True
            return cached

        # 1. ChromaDB ACL 前置过滤
        role_field = "role_teacher" if user_role == "teacher" else "role_student"
        where_filter: dict = {role_field: True}
        if subject:
            where_filter = {"$and": [{role_field: True}, {"subject": subject}]}

        vec_ids = []
        if ablation_mode in {"vec_only", "hybrid_rrf", "hybrid_rerank"}:
            res_vec = self.writer.collection.query(
                query_embeddings=[q_vec],
                n_results=s.retrieval.top_k_vector,
                where=where_filter
            )
            vec_ids = res_vec["ids"][0] if res_vec["ids"] else []

        # 2. SQLite FTS5 检索（ACL 前置过滤 + INDEXED 状态校验，bm25 排序）
        fts_ids = []
        if ablation_mode in {"fts_only", "hybrid_rrf", "hybrid_rerank"}:
            fts_ids = self.writer.fts.search(query, user_role=user_role, subject=subject, limit=s.retrieval.top_k_fts)

        # 3. 候选合并
        if ablation_mode == "vec_only":
            merged_ids = vec_ids[:top_k_final]
        elif ablation_mode == "fts_only":
            merged_ids = fts_ids[:top_k_final]
        else:
            # RRF 倒数秩融合。
            # 关键设计：合并池保留"两路召回的全部唯一候选"（最多两池之和），
            # 而不是截断到单池大小——否则只在单路出现的中后排候选会被双边
            # 候选挤出重排池（实测导致"讲义公式块"被题目块淹没、Recall 掉 20+pp）。
            k = s.retrieval.rrf_k
            scores: Dict[str, float] = {}
            for r_i, cid in enumerate(vec_ids, 1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r_i)
            for r_i, cid in enumerate(fts_ids, 1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r_i)
            pool_cap = s.retrieval.top_k_vector + s.retrieval.top_k_fts
            merged_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:pool_cap]]

        if not merged_ids:
            return {"chunks": [], "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "cache_hit": False}

        # 4. 回表：Chroma 取子块文本/元数据，SQLite 取父块完整上下文
        cur = self.writer.sm.conn.cursor()
        chroma_got = self.writer.collection.get(ids=merged_ids, include=["documents", "metadatas"])
        id_to_doc = dict(zip(chroma_got["ids"], chroma_got["documents"] or []))
        id_to_meta = dict(zip(chroma_got["ids"], chroma_got["metadatas"] or []))

        candidates = []
        for cid in merged_ids:
            meta = id_to_meta.get(cid)
            if not meta:
                continue
            p_row = cur.execute("SELECT * FROM parent_chunks WHERE parent_id = ?", (meta["parent_id"],)).fetchone()
            parent_text = p_row["parent_text"] if p_row else id_to_doc.get(cid, "")
            candidates.append({
                "chunk_id": cid.split("@")[0],
                "doc_id": meta["doc_id"],
                "version": meta["version"],
                "subject": meta["subject"],
                "section_path": meta["section_path"],
                "child_text": id_to_doc.get(cid, ""),
                "parent_text": parent_text,
                "rerank_score": 0.0
            })

        # 5. 重排 Top50 -> Top8
        if ablation_mode == "hybrid_rerank" and len(candidates) > 1:
            passages = [c["child_text"] for c in candidates]
            ranked = self.reranker.rerank(query, passages, top_n=min(top_k_final, len(candidates)))
            final_chunks = []
            for item in ranked:
                idx = item["index"]
                if idx < len(candidates):
                    c = candidates[idx]
                    c["rerank_score"] = round(float(item["relevance_score"]), 4)
                    final_chunks.append(c)
        else:
            final_chunks = candidates[:top_k_final]

        out = {
            "chunks": final_chunks,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "cache_hit": False,
            "query_vec": q_arr
        }
        return out

    def cache_stats(self) -> dict:
        return self.semantic_cache.stats()
