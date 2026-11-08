# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, List
import chromadb
from src.config import get_settings
from src.quality.state_machine import MetadataStateMachine
from src.ingestion.parser_router import MultiFormatParserRouter
from src.indexing.embedder import CachedBgeM3Embedder
from src.indexing.fts5_store import FTS5Store, tokenize_for_fts5


class AtomicDualIndexWriter:
    """
    负责 ChromaDB (稠密向量) + SQLite FTS5 (jieba 预分词 BM25) + parent_chunks 的原子双写与故障回滚
    """
    def __init__(self, chroma_dir: str | None = None, db_path: str | None = None):
        s = get_settings()
        self.sm = MetadataStateMachine(db_path or str(s.metadata_db_path))
        self.fts = FTS5Store(self.sm.conn)
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir or str(s.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowpulse_edu_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def tokenize_for_fts5(cls, text: str) -> str:
        """jieba 预分词 + 专有编号/公式正则保护（委托 fts5_store 保持口径统一）"""
        return tokenize_for_fts5(text)

    def atomic_upsert_document(
        self,
        base_doc_id: str,
        version: float,
        subject: str,
        title: str,
        access_roles: str,
        chunks: List[Dict],
        vectors: List[List[float]],
        simulate_fts_crash: bool = False
    ):
        """
        原子双写协议：
        1. 写入 ChromaDB 新版本子块向量（携带 role_teacher / role_student 布尔元数据，供 ACL 前置过滤）
        2. 写入 SQLite parent_chunks 与 chunks_fts
        3. 若 version > 1.0，在同一事务中将旧版本置为 DEPRECATED，并物理清除 ChromaDB 与 FTS5 中的旧版块
        4. 若中途发生任何异常（含故障注入测试），立即回滚 SQLite 并从 ChromaDB 删除刚写入的新版本 chunk_ids
        """
        if not chunks:
            return
        cur = self.sm.conn.cursor()
        child_ids = [f"{c['child_id']}@v{version:.1f}" for c in chunks]
        role_student = "student" in access_roles
        role_teacher = "teacher" in access_roles

        try:
            # Step 1: 写入 ChromaDB
            self.collection.upsert(
                ids=child_ids,
                embeddings=vectors,
                documents=[c["child_text"] for c in chunks],
                metadatas=[{
                    "doc_id": base_doc_id,
                    "parent_id": f"{c['parent_id']}@v{version:.1f}",
                    "version": float(version),
                    "subject": subject,
                    "section_path": c["section_path"],
                    "role_student": role_student,
                    "role_teacher": role_teacher
                } for c in chunks]
            )

            # 故障注入开关（用于 pytest 验证双写事务回滚无脏读）
            if simulate_fts_crash:
                raise RuntimeError("Injected Fault: Simulated crash after ChromaDB write before FTS5 commit!")

            # Step 2: 写入 SQLite 父块表与 FTS5 虚拟表
            seen_parents = set()
            for cid_ver, c in zip(child_ids, chunks):
                pid_ver = f"{c['parent_id']}@v{version:.1f}"
                if pid_ver not in seen_parents:
                    cur.execute("""
                    INSERT OR REPLACE INTO parent_chunks
                    (parent_id, doc_id, version, subject, title, section_path, parent_text, media_path, access_roles)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (pid_ver, base_doc_id, version, subject, title, c["section_path"], c["parent_text"], c["media_path"], access_roles))
                    seen_parents.add(pid_ver)

                self.fts.insert_chunk(
                    cur, cid_ver, pid_ver, base_doc_id, version, subject, access_roles, c["child_text"]
                )

            # Step 3: 原子切换版本状态并清理旧版本 (< version)
            cur.execute("""
            UPDATE doc_versions SET status = 'DEPRECATED', updated_at = CURRENT_TIMESTAMP
            WHERE base_doc_id = ? AND version < ? AND status = 'INDEXED'
            """, (base_doc_id, version))

            cur.execute("""
            UPDATE doc_versions SET status = 'INDEXED', updated_at = CURRENT_TIMESTAMP
            WHERE base_doc_id = ? AND version = ?
            """, (base_doc_id, version))

            self.fts.remove_document(base_doc_id, version)
            cur.execute("DELETE FROM parent_chunks WHERE doc_id = ? AND version < ?", (base_doc_id, version))
            self.sm.conn.commit()

            if version > 1.0:
                self.collection.delete(where={"$and": [{"doc_id": base_doc_id}, {"version": {"$lt": float(version)}}]})

        except Exception as e:
            self.sm.conn.rollback()
            try:
                self.collection.delete(ids=child_ids)
            except Exception:
                pass
            raise e


def run_full_indexing(data_root: Path | None = None, embed_mode: str | None = None, limit: int | None = None):
    s = get_settings()
    data_root = data_root or s.data_root
    writer = AtomicDualIndexWriter()
    router = MultiFormatParserRouter(parsed_cache_dir=str(s.parsed_dir))
    embedder = CachedBgeM3Embedder(writer.sm.conn, mode=embed_mode or s.effective_embed_mode)
    cur = writer.sm.conn.cursor()

    # 1. 先索引全部 PENDING 状态的干净文档
    sql_pending = "SELECT * FROM doc_versions WHERE status = 'PENDING' ORDER BY base_doc_id"
    if limit:
        sql_pending += f" LIMIT {int(limit)}"
    pending_docs = cur.execute(sql_pending).fetchall()
    print(f"[*] 开始解析并双写索引 {len(pending_docs)} 份 PENDING 干净文档 (embed_mode={embedder.mode})...")

    for idx, doc in enumerate(pending_docs, 1):
        full_path = data_root / doc["rel_path"]
        chunks = router.parse_document(dict(doc), full_path)
        if not chunks:
            cur.execute("UPDATE doc_versions SET status='QUARANTINED', quarantine_reason='LOW_OCR_CONFIDENCE (<0.75)' WHERE base_doc_id=? AND version=?",
                        (doc["base_doc_id"], doc["version"]))
            writer.sm.conn.commit()
            continue
        vecs = embedder.encode([c["child_text"] for c in chunks])
        writer.atomic_upsert_document(
            base_doc_id=doc["base_doc_id"],
            version=float(doc["version"]),
            subject=doc["subject"],
            title=doc["title"],
            access_roles=doc["access_roles"],
            chunks=chunks,
            vectors=vecs
        )
        if idx % 100 == 0 or idx == len(pending_docs):
            print(f"    已完成 {idx}/{len(pending_docs)} 份主库文档双写...")

    # 2. 处理 VERSION_UPGRADE_QUEUE（v2.0 冲突文档）的原子增量更新
    sql_v2 = "SELECT * FROM doc_versions WHERE status = 'VERSION_UPGRADE_QUEUE' ORDER BY base_doc_id"
    if limit:
        sql_v2 += f" LIMIT {int(limit)}"
    v2_docs = cur.execute(sql_v2).fetchall()
    print(f"[*] 开始执行 {len(v2_docs)} 份 v2.0 冲突文档的原子增量更新（自动废止 v1.0 旧 chunk）...")
    for doc in v2_docs:
        full_path = data_root / doc["rel_path"]
        chunks = router.parse_document(dict(doc), full_path)
        if not chunks:
            cur.execute("UPDATE doc_versions SET status='QUARANTINED', quarantine_reason='LOW_OCR_CONFIDENCE (<0.75)' WHERE base_doc_id=? AND version=?",
                        (doc["base_doc_id"], doc["version"]))
            writer.sm.conn.commit()
            continue
        vecs = embedder.encode([c["child_text"] for c in chunks])
        writer.atomic_upsert_document(
            base_doc_id=doc["base_doc_id"],
            version=float(doc["version"]),
            subject=doc["subject"],
            title=doc["title"],
            access_roles=doc["access_roles"],
            chunks=chunks,
            vectors=vecs
        )
    n_indexed = cur.execute("SELECT COUNT(*) AS c FROM doc_versions WHERE status='INDEXED'").fetchone()["c"]
    n_fts = writer.fts.count()
    print(f"[Step 2 & Step 3 完成] 双索引构建完成：INDEXED 文档 {n_indexed} 份，FTS5 块 {n_fts} 条，"
          f"v2.0 增量版本切换 {len(v2_docs)} 份。")


if __name__ == "__main__":
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(s.data_root))
    p.add_argument("--embed-mode", choices=["local", "siliconflow"], default=None)
    p.add_argument("--limit", type=int, default=None, help="只索引前 N 份 PENDING 文档（冒烟测试用）")
    args = p.parse_args()
    run_full_indexing(Path(args.data_root), args.embed_mode, args.limit)