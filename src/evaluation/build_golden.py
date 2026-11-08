# -*- coding: utf-8 -*-
"""
Golden Set 构建脚本（Phase 0 / Phase 5 工具）。

流程（方案 v1.2 第 8.2 节）：
1. 读取已有 golden set（golden_set_v1.jsonl，人工复核产物）；
2. `candidates`：基于已入库语料，按"5 难度层 x 3 格式"分层，用 LLM 生成候选 QA，
   输出 candidates.jsonl —— 这一步的产出必须经过人工复核（防评测泄漏）；
3. `merge`：把复核通过的候选并入主评测集，按问题内容哈希去重，
   输出 golden_set_v2.jsonl。

证据绑定：evidence_doc_ids 既可以是相对路径（lectures/LEC-000001.docx），
也可以是裸 doc_id（TAB-000011）；两种形式都会被校验/规范化。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.config import get_settings
from src.indexing.dual_writer import AtomicDualIndexWriter
from src.llm.siliconflow_client import SiliconFlowService

CANDIDATE_SYSTEM = "你只输出合法的 JSON 对象，不要包含 Markdown 代码块。"

# 分层配额：难度层 -> 每类目标条数
STRATA_QUOTA = {
    "简单检索": 40,
    "跨文档聚合": 20,
    "表格数值": 20,
    "图片理解": 20,
    "应拒答": 10,
}


def _question_hash(question: str) -> str:
    core = re.sub(r"\s+", "", question.strip())
    return hashlib.sha1(core.encode("utf-8")).hexdigest()[:16]


def load_golden(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def normalize_evidence(ev: str) -> str:
    """把证据标识规范化为 base_doc_id（相对路径 -> 去目录去扩展名）。"""
    ev = ev.strip()
    if "/" in ev or "\\" in ev:
        ev = ev.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"\.(docx|xlsx|pdf|jpg|jpeg|png|json)$", "", ev)


def build_known_doc_ids(writer: AtomicDualIndexWriter) -> set:
    rows = writer.sm.conn.cursor().execute("SELECT base_doc_id, rel_path FROM doc_versions").fetchall()
    known = {r["base_doc_id"] for r in rows}
    for r in rows:
        p = r["rel_path"].replace("\\", "/")
        known.add(re.sub(r"\.(docx|xlsx|pdf|jpg|jpeg|png|json)$", "", p.rsplit("/", 1)[-1]))
    return known


def validate_golden(records: List[dict], writer: AtomicDualIndexWriter) -> List[str]:
    """校验证据是否能在索引库中解析，返回警告列表。"""
    known = build_known_doc_ids(writer)
    warnings = []
    for rec in records:
        for ev in rec.get("evidence_doc_ids", []):
            if normalize_evidence(ev) not in known:
                warnings.append(f"{rec.get('id', '?')}: 证据 {ev} 无法解析到已入库文档")
    return warnings


# ----------------------------------------------------------------------
def generate_candidates(out_jsonl: Path, per_stratum: Optional[dict] = None) -> int:
    s = get_settings()
    if not s.has_siliconflow_key:
        raise SystemExit(
            "[!] 未配置 SILICONFLOW_API_KEY（configs/.env）。候选 QA 生成需要 LLM API；"
            "请先填写密钥，或直接使用已有 golden_set_v1.jsonl 做人工扩充。"
        )
    writer = AtomicDualIndexWriter()
    sf = SiliconFlowService()
    quota = per_stratum or STRATA_QUOTA
    cur = writer.sm.conn.cursor()

    # 简单检索：从讲义/题库父块出题
    simple_src = cur.execute("""
        SELECT p.doc_id, p.title, p.parent_text FROM parent_chunks p
        JOIN doc_versions d ON p.doc_id = d.base_doc_id AND p.version = d.version
        WHERE d.status='INDEXED' AND d.category IN ('question_bank','lectures')
        ORDER BY RANDOM() LIMIT ?
    """, (quota["简单检索"] * 3,)).fetchall()

    # 表格数值：从 tables 出题
    table_src = cur.execute("""
        SELECT p.doc_id, p.title, p.parent_text FROM parent_chunks p
        JOIN doc_versions d ON p.doc_id = d.base_doc_id AND p.version = d.version
        WHERE d.status='INDEXED' AND d.category='tables'
        ORDER BY RANDOM() LIMIT ?
    """, (quota["表格数值"] * 3,)).fetchall()

    # 图片理解：从 scanned/images 出题（父块含标题/元数据）
    image_src = cur.execute("""
        SELECT p.doc_id, p.title, p.parent_text FROM parent_chunks p
        JOIN doc_versions d ON p.doc_id = d.base_doc_id AND p.version = d.version
        WHERE d.status='INDEXED' AND d.category IN ('scanned','images')
        ORDER BY RANDOM() LIMIT ?
    """, (quota["图片理解"] * 3,)).fetchall()

    def llm_qa(prompt_template: str, row, difficulty: str) -> Optional[dict]:
        prompt = prompt_template.format(title=row["title"], text=row["parent_text"][:700])
        try:
            res = sf.chat_with_cost(CANDIDATE_SYSTEM, prompt, temperature=0.3)
            txt = res["answer"].strip().strip("`").removeprefix("json").strip()
            qa = json.loads(txt)
            return {
                "question": qa["question"],
                "expected_answer": qa.get("ground_truth") or qa.get("answer") or "",
                "evidence_doc_ids": [row["doc_id"]],
                "difficulty": difficulty,
                "subject": "",
                "eval_note": f"LLM 候选生成自 {row['title']}（待人工复核）",
                "reviewed": False,
            }
        except Exception:
            return None

    out: List[dict] = []
    for row in simple_src:
        cand = llm_qa(
            "请基于以下教育文档片段，设计 1 个具体的检索测试问题和简短标准答案（JSON：{\"question\": \"...\", \"ground_truth\": \"...\"}）。"
            "问题必须能被该片段唯一回答：\n\n文档标题：{title}\n片段：{text}", row, "简单检索")
        if cand:
            out.append(cand)
    for row in table_src:
        cand = llm_qa(
            "请基于以下表格文档，设计 1 个「必须读取表格数据才能回答」的数值型问题与标准答案（JSON：{\"question\": \"...\", \"ground_truth\": \"...\"}）：\n\n文档标题：{title}\n内容：{text}",
            row, "表格数值")
        if cand:
            out.append(cand)
    for row in image_src:
        cand = llm_qa(
            "请基于以下扫描件/图片文档的元数据与文字内容，设计 1 个关于该文档标题/内容的识别型问题与标准答案（JSON：{\"question\": \"...\", \"ground_truth\": \"...\"}）：\n\n文档标题：{title}\n内容：{text}",
            row, "图片理解")
        if cand:
            out.append(cand)

    # 应拒答：从领域外模板生成
    refused_templates = [
        "2025 年国庆节 A 股市场的休市安排是什么？",
        "如何用 PyTorch 从零实现 Transformer 模型？",
        "北京地铁 4 号线的末班车时间是几点？",
        "某互联网公司 2024 年第四季度财报净利润是多少？",
        "量子计算机的拓扑量子纠错最新进展如何？",
    ]
    for q in refused_templates[: quota["应拒答"]]:
        out.append({
            "question": q,
            "expected_answer": "应拒答：知识库中无此内容",
            "evidence_doc_ids": [],
            "difficulty": "应拒答",
            "subject": "综合",
            "eval_note": "领域外问题模板（待人工复核）",
            "reviewed": False,
        })

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[+] 已生成 {len(out)} 条候选 QA -> {out_jsonl}（请人工复核后再 merge）")
    return len(out)


# ----------------------------------------------------------------------
def merge_golden(existing_path: Path, candidates_path: Path, out_path: Path) -> int:
    existing = load_golden(existing_path)
    candidates = load_golden(candidates_path)

    seen = {_question_hash(r["question"]) for r in existing}
    added = []
    for c in candidates:
        if not c.get("reviewed"):
            continue                        # 只合并人工复核通过的候选
        h = _question_hash(c["question"])
        if h in seen:
            continue
        seen.add(h)
        added.append(c)

    merged = existing + added
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[+] 合并完成：原 {len(existing)} 条 + 新增 {len(added)} 条 = {len(merged)} 条 -> {out_path}")
    return len(added)


if __name__ == "__main__":
    s = get_settings()
    p = argparse.ArgumentParser(description="Golden Set 构建工具")
    p.add_argument("--action", choices=["candidates", "merge", "validate"], required=True)
    p.add_argument("--golden", default=str(s.golden_set_path))
    p.add_argument("--candidates", default="data/golden_set/candidates.jsonl")
    p.add_argument("--out", default="data/golden_set/golden_set_v2.jsonl")
    args = p.parse_args()

    if args.action == "candidates":
        generate_candidates(Path(args.candidates))
    elif args.action == "merge":
        merge_golden(Path(args.golden), Path(args.candidates), Path(args.out))
    else:
        writer = AtomicDualIndexWriter()
        warnings = validate_golden(load_golden(Path(args.golden)), writer)
        if warnings:
            print("[!] 以下证据无法解析：")
            for w in warnings:
                print("   ", w)
        else:
            print("[+] 全部证据均可解析到已入库文档。")
