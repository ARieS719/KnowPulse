# -*- coding: utf-8 -*-
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

class MetadataStateMachine:
    """
    管理 SQLite 核心库 (data/storage/metadata.db)：
    1. doc_versions: 文档版本状态机 (PENDING -> INDEXED -> DEPRECATED / QUARANTINED / VERSION_UPGRADE_QUEUE)
    2. quality_audit_log: 质量门全量拦截审计流水
    3. parent_chunks: 父子块架构中的父块完整原文存储（含完整大表格与图片元数据）
    4. embedding_cache: SHA256 向量缓存（相同文本绝不重复调用 bge-m3）
    5. chunks_fts: SQLite FTS5 全文检索表（jieba 预分词）
    """
    def __init__(self, db_path: str = "data/storage/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：FastAPI 线程池端点会跨线程复用同一连接（单写多读场景安全）
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_versions (
            base_doc_id TEXT,
            raw_doc_id TEXT,
            version REAL,
            rel_path TEXT,
            category TEXT,
            subject TEXT,
            title TEXT,
            file_sha256 TEXT,
            status TEXT CHECK(status IN ('PENDING', 'INDEXED', 'DEPRECATED', 'QUARANTINED', 'VERSION_UPGRADE_QUEUE')),
            access_roles TEXT DEFAULT 'teacher,student',
            quarantine_reason TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (base_doc_id, version)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS quality_audit_log (
            doc_id TEXT PRIMARY KEY,
            rel_path TEXT,
            category TEXT,
            expected_tag TEXT,
            detected_tag TEXT,
            action TEXT,
            reason TEXT,
            pii_masked_count INTEGER DEFAULT 0,
            audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS parent_chunks (
            parent_id TEXT PRIMARY KEY,
            doc_id TEXT,
            version REAL,
            subject TEXT,
            title TEXT,
            section_path TEXT,
            parent_text TEXT,
            media_path TEXT,
            access_roles TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_sha256 TEXT PRIMARY KEY,
            model_name TEXT,
            vector_blob BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            parent_id UNINDEXED,
            doc_id UNINDEXED,
            version UNINDEXED,
            subject UNINDEXED,
            access_roles UNINDEXED,
            tokenized_content
        );
        """)
        self.conn.commit()

    @staticmethod
    def get_base_doc_id(raw_doc_id: str) -> str:
        """剥离 -v2 等版本后缀，如 'QB-SYN-000501-v2' -> 'QB-SYN-000501'"""
        return re.sub(r"-v\d+$", "", raw_doc_id)

    def check_version_conflict(self, raw_doc_id: str, version: float, known_base_ids: Set[str]) -> Tuple[bool, str]:
        base_id = self.get_base_doc_id(raw_doc_id)
        if (base_id != raw_doc_id or version > 1.0) and base_id in known_base_ids:
            return True, f"VERSION_CONFLICT: 新版本 {raw_doc_id} (v{version:.1f}) 与存量基准版本 {base_id} (v1.0) 冲突，已路由至增量原子切换队列"
        return False, ""