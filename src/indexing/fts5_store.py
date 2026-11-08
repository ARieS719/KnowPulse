# -*- coding: utf-8 -*-
"""
SQLite FTS5 全文检索引擎（jieba 预分词 BM25）。

设计要点（README design decisions 之一）：
- unicode61 分词器会把中文按单字切碎，因此写入前必须用 jieba 预分词并以空格分隔；
- 课程号/文档编号/LaTeX 命令等专有 token 用正则"保护"，避免被 jieba 切碎（第一轮 badcase 修复点）；
- 检索时同样先对 query 做分词，token 加双引号避免 FTS5 运算符歧义，再按 bm25() 排序。
"""
from __future__ import annotations

import re
import sqlite3
from typing import List, Optional

import jieba

# 保护课程号、文档编号、LaTeX 公式不被 jieba 切碎
PROTECT_TOKEN_RE = re.compile(r"([A-Z]{2,6}-[A-Z0-9\-]+|\\[a-zA-Z]+|\d+\.\d+)")


def tokenize_for_fts5(text: str) -> str:
    """jieba 预分词 + 专有编号/公式正则保护，以空格分隔，供 FTS5 unicode61 索引。"""
    if not text:
        return ""
    protected = PROTECT_TOKEN_RE.findall(text)
    words = [w.strip() for w in jieba.cut_for_search(text) if w.strip()]
    return " ".join(words + protected)


def _safe_token(token: str) -> str:
    """把一个 token 安全地包进 FTS5 短语引号内。"""
    return token.replace('"', "").replace("'", "")


class FTS5Store:
    """chunks_fts 虚拟表的查询封装：ACL 前置过滤 + INDEXED 状态校验 + bm25 排序。"""

    def __init__(self, conn: sqlite3.Connection, max_query_tokens: int = 12):
        self.conn = conn
        self.max_query_tokens = max_query_tokens

    # ------------------------------------------------------------------
    # 写入辅助（供 AtomicDualIndexWriter 使用，保持双写口径一致）
    # ------------------------------------------------------------------
    @staticmethod
    def insert_chunk(cur: sqlite3.Cursor, chunk_id: str, parent_id: str, doc_id: str,
                     version: float, subject: str, access_roles: str, child_text: str) -> None:
        seg_txt = tokenize_for_fts5(child_text)
        cur.execute(
            """
            INSERT INTO chunks_fts (chunk_id, parent_id, doc_id, version, subject, access_roles, tokenized_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, parent_id, doc_id, float(version), subject, access_roles, seg_txt),
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def search(self, query: str, user_role: str = "student",
               subject: Optional[str] = None, limit: int = 50) -> List[str]:
        """
        BM25 全文召回。

        返回按 bm25 升序（相关性降序）排列的 chunk_id 列表。
        与向量召回对齐：role 走 access_roles 前置过滤，subject 可选过滤。
        """
        tokens = [t for t in tokenize_for_fts5(query).split() if len(t) > 1][: self.max_query_tokens]
        if not tokens:
            return []

        match_expr = " OR ".join(f'"{_safe_token(t)}"' for t in tokens)
        sql = """
        SELECT f.chunk_id
        FROM chunks_fts f
        JOIN doc_versions d ON f.doc_id = d.base_doc_id AND f.version = d.version
        WHERE chunks_fts MATCH ?
          AND d.status = 'INDEXED'
          AND instr(f.access_roles, ?) > 0
        """
        params: list = [match_expr, user_role]
        if subject:
            sql += " AND f.subject = ?"
            params.append(subject)
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(limit)

        try:
            rows = self.conn.cursor().execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # MATCH 语法异常（极端 token）时降级为空结果，不让整个检索链路崩溃
            return []
        return [r["chunk_id"] for r in rows]

    def remove_document(self, doc_id: str, below_version: float) -> None:
        """删除某文档所有低于指定版本的 FTS5 记录（增量更新用）。"""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks_fts WHERE doc_id = ? AND version < ?", (doc_id, below_version))

    def count(self) -> int:
        row = self.conn.cursor().execute("SELECT COUNT(*) AS c FROM chunks_fts").fetchone()
        return int(row["c"]) if row else 0
