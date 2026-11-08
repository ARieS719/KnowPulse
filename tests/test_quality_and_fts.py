# -*- coding: utf-8 -*-
"""质量算子与 FTS5 的单元测试（不依赖模型与 API）。"""
import json

import pytest

from src.indexing.fts5_store import FTS5Store, tokenize_for_fts5
from src.quality.operators import CalibratedQualityOperators


def test_mask_pii():
    text = "学生张三电话13812345678身份证110101199003071234。"
    masked, n = CalibratedQualityOperators.mask_pii(text)
    assert "138****5678" in masked
    assert "19900307" not in masked
    assert n == 2


def test_garbled_json_field(tmp_path):
    ops = CalibratedQualityOperators()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"题干": "正常题干内容足够长", "答案": "锟斤拷烫烫烫硚硟硖" * 3}), encoding="utf-8")
    hit, reason = ops.check_garbled_and_format(bad, ".json")
    assert hit and reason.startswith("GARBLED_FIELD")


def test_math_symbols_not_garbled(tmp_path):
    """回归测试：× ° ± ² 等合法数学符号绝不能触发乱码拦截（实测曾误杀 1444 份干净数学文档）。"""
    ops = CalibratedQualityOperators()
    good = tmp_path / "math.json"
    good.write_text(
        json.dumps({"题干": "体积为 5 dm³ 的物体完全浸没在水中（ρ水=1.0×10³ kg/m³，g 取 10 N/kg），水温 25°C，误差 ±0.5%"}),
        encoding="utf-8",
    )
    hit, reason = ops.check_garbled_and_format(good, ".json")
    assert not hit, reason


def test_clean_json_passes(tmp_path):
    ops = CalibratedQualityOperators()
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"题干": "已知二次函数 y=ax^2+bx+c 的图像开口向上", "答案": "D"}, ensure_ascii=False), encoding="utf-8")
    hit, _ = ops.check_garbled_and_format(good, ".json")
    assert not hit


def test_blank_pdf_by_size(tmp_path):
    ops = CalibratedQualityOperators()
    p = tmp_path / "blank.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    hit, reason = ops.check_blank_pdf(p, size_kb=10.0)
    assert hit and "BLANK" in reason


def test_fts5_tokenize_protects_doc_id():
    text = "这道题来自 QB-REAL-000001 的二次函数部分"
    seg = tokenize_for_fts5(text)
    assert "QB-REAL-000001" in seg


def test_fts5_search_and_acl(tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED, parent_id UNINDEXED, doc_id UNINDEXED, version UNINDEXED,
            subject UNINDEXED, access_roles UNINDEXED, tokenized_content
        )
    """)
    conn.execute("""
        CREATE TABLE doc_versions (
            base_doc_id TEXT, version REAL, status TEXT,
            PRIMARY KEY (base_doc_id, version)
        )
    """)
    conn.execute("INSERT INTO doc_versions VALUES ('QB-000001', 1.0, 'INDEXED')")
    conn.commit()

    store = FTS5Store(conn)
    cur = conn.cursor()
    store.insert_chunk(cur, "QB-000001#C001@v1.0", "QB-000001#P001@v1.0",
                       "QB-000001", 1.0, "数学", "teacher,student", "二次函数的对称轴公式")
    store.insert_chunk(cur, "QB-000001#C002@v1.0", "QB-000001#P002@v1.0",
                       "QB-000001", 1.0, "数学", "teacher", "教研组内部备课笔记（仅教师可见）")
    conn.commit()

    ids_student = store.search("二次函数 对称轴", user_role="student", subject="数学")
    assert ids_student == ["QB-000001#C001@v1.0"]          # 学生看不到 teacher 专属块
    ids_teacher = store.search("二次函数 对称轴", user_role="teacher", subject="数学")
    assert set(ids_teacher) >= {"QB-000001#C001@v1.0"}

    # 状态机校验：DEPRECATED 文档不可被检索
    conn.execute("UPDATE doc_versions SET status='DEPRECATED' WHERE base_doc_id='QB-000001'")
    conn.commit()
    assert store.search("二次函数", user_role="teacher", subject="数学") == []
    conn.close()
