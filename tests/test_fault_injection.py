# -*- coding: utf-8 -*-
import pytest
from src.indexing.dual_writer import AtomicDualIndexWriter

def test_atomic_dual_write_rollback_on_fts_crash(tmp_path):
    """
    验证当 ChromaDB 写入成功但 SQLite FTS5 写入前发生崩溃时：
    1. v1.0 旧版本数据在 ChromaDB 和 FTS5 中毫发无损；
    2. v2.0 半成品数据从 ChromaDB 中被干净回滚清除，绝不产生脏读。
    """
    chroma_dir = str(tmp_path / "chroma")
    db_path = str(tmp_path / "test_meta.db")
    writer = AtomicDualIndexWriter(chroma_dir=chroma_dir, db_path=db_path)
    cur = writer.sm.conn.cursor()

    # 1. 正常写入 v1.0
    cur.execute("INSERT INTO doc_versions (base_doc_id, raw_doc_id, version, status, access_roles) VALUES ('QB-SYN-000501', 'QB-SYN-000501', 1.0, 'PENDING', 'teacher,student')")
    writer.sm.conn.commit()
    v1_chunks = [{"parent_id": "QB-SYN-000501#P001", "child_id": "QB-SYN-000501#C001", "section_path": "第1题", "child_text": "旧版v1题干", "parent_text": "旧版v1完整解析", "media_path": ""}]
    writer.atomic_upsert_document("QB-SYN-000501", 1.0, "数学", "测试题组", "teacher,student", v1_chunks, [[0.1] * 1024])

    # 2. 模拟写入 v2.0 时在 FTS5 提交前断电抛异常
    cur.execute("INSERT INTO doc_versions (base_doc_id, raw_doc_id, version, status, access_roles) VALUES ('QB-SYN-000501', 'QB-SYN-000501-v2', 2.0, 'VERSION_UPGRADE_QUEUE', 'teacher,student')")
    writer.sm.conn.commit()
    v2_chunks = [{"parent_id": "QB-SYN-000501#P001", "child_id": "QB-SYN-000501#C001", "section_path": "第1题", "child_text": "新版v2题干", "parent_text": "新版v2完整解析", "media_path": ""}]

    with pytest.raises(RuntimeError, match="Injected Fault"):
        writer.atomic_upsert_document("QB-SYN-000501", 2.0, "数学", "测试题组v2", "teacher,student", v2_chunks, [[0.2] * 1024], simulate_fts_crash=True)

    # 3. 断言：ChromaDB 中不存在 v2.0 的残留块，且 v1.0 依然处于 INDEXED 激活状态
    chroma_v2 = writer.collection.get(ids=["QB-SYN-000501#C001@v2.0"])
    assert len(chroma_v2["ids"]) == 0
    v1_status = cur.execute("SELECT status FROM doc_versions WHERE base_doc_id='QB-SYN-000501' AND version=1.0").fetchone()["status"]
    assert v1_status == "INDEXED"