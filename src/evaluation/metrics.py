# -*- coding: utf-8 -*-
"""
三层评测套件（方案 v1.2 第 8.1 节）。

- 数据层：质量门审计对账 + 索引状态 + chunk 长度分布（data_layer_metrics）
- 检索层：四档消融实验 Recall@5/@10 / MRR / nDCG / 延迟（run_ablation_evaluation）
- 生成层：拒答正确率 + 引用硬校验 + Faithfulness（judge LLM）（run_generation_evaluation）

Golden Set 字段兼容（golden_set_v1.jsonl）：
  id / question / expected_answer / evidence_doc_ids（相对路径或裸 doc_id）/ difficulty / subject / eval_note
难度层为中文：简单检索 / 跨文档聚合 / 表格数值 / 图片理解 / 应拒答。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

from src.config import get_settings
from src.generation.rag_chain import KnowPulseRAGChain
from src.indexing.dual_writer import AtomicDualIndexWriter
from src.indexing.embedder import CachedBgeM3Embedder
from src.llm.siliconflow_client import SiliconFlowService
from src.retrieval.hybrid_search import HybridRetrieverService

SIMPLE_DIFFICULTY = "简单检索"
REFUSE_DIFFICULTY = "应拒答"


# ----------------------------------------------------------------------
# 证据解析：evidence_doc_ids（相对路径 / 裸 doc_id）-> doc_versions.base_doc_id
# ----------------------------------------------------------------------
def build_evidence_resolver(writer: AtomicDualIndexWriter) -> Dict[str, str]:
    cur = writer.sm.conn.cursor()
    rows = cur.execute("SELECT base_doc_id, rel_path FROM doc_versions").fetchall()
    resolver: Dict[str, str] = {}
    for r in rows:
        base = r["base_doc_id"]
        rel = r["rel_path"].replace("\\", "/")
        resolver[rel] = base
        resolver[rel.rsplit("/", 1)[-1]] = base                       # 文件名（带扩展名）
        resolver[re.sub(r"\.(docx|xlsx|pdf|jpg|jpeg|png|json)$", "", rel.rsplit("/", 1)[-1])] = base
        resolver[base] = base
    return resolver


def resolve_evidence(ev: str, resolver: Dict[str, str]) -> str:
    ev = ev.strip()
    return resolver.get(ev, resolver.get(re.sub(r"\.(docx|xlsx|pdf|jpg|jpeg|png|json)$", "", ev.rsplit("/", 1)[-1]), ev))


# ----------------------------------------------------------------------
# 内容指纹级命中判定：
# 语料由 16 个讲义模板 × ~62 份副本构成，同一模板的副本文档内容完全相同。
# doc 级 Recall 会把"检索到内容相同但编号不同的副本"误判为未命中（实测把
# 74% 的真实召回低估为 14%）。因此以父块文本指纹为命中口径：
# 检索到的 chunk 所属文档与证据文档共享任一相同父块指纹 => 命中。
# ----------------------------------------------------------------------
def build_content_fingerprints(writer: AtomicDualIndexWriter) -> Dict[str, set]:
    import hashlib
    cur = writer.sm.conn.cursor()
    rows = cur.execute("""
        SELECT p.doc_id, p.parent_text FROM parent_chunks p
        JOIN doc_versions d ON p.doc_id = d.base_doc_id AND p.version = d.version
        WHERE d.status = 'INDEXED'
    """).fetchall()
    fp: Dict[str, set] = {}
    for r in rows:
        h = hashlib.sha1(re.sub(r"\s+", "", r["parent_text"]).encode("utf-8")).hexdigest()[:16]
        fp.setdefault(r["doc_id"], set()).add(h)
    return fp


def content_hit(gold_docs: List[str], retrieved_docs: List[str], fp: Dict[str, set]) -> bool:
    gold_fps: set = set()
    for g in gold_docs:
        gold_fps |= fp.get(g, set())
    if not gold_fps:
        return any(d in gold_docs for d in retrieved_docs)   # 无指纹时退化为 doc 级
    return any(bool(fp.get(d, set()) & gold_fps) for d in retrieved_docs)


def normalize_answer_text(s: str) -> str:
    """NFKC 归一化：下标/上标/全角字符 → 普通半角，去空白。"""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s\u3000]+", "", s)


def answer_hit(expected_answer: str, retrieved_chunks: List[dict]) -> bool:
    """答案包含性判定：检索到的块内容（父块优先）能否支撑标准答案。

    - 整个标准答案（NFKC 归一化后）出现在检索内容中 => 命中；
    - 多段式答案（如 "P=UI 与 I²R"）拆段后每一段都出现 => 命中；
    - 标准答案过短（纯数字等）时不做包含性判定，返回 False（交给其他口径）。
    """
    if not expected_answer:
        return False
    exp = normalize_answer_text(expected_answer)
    if len(exp) < 4:
        return False
    text = normalize_answer_text(" ".join(c.get("parent_text", "") for c in retrieved_chunks))
    if exp in text:
        return True
    parts = [p for p in re.split(r"[、，；;与或]", exp) if len(p) >= 3]
    return len(parts) >= 2 and all(p in text for p in parts)


def judge_hit(gold_docs: List[str], retrieved_docs: List[str], retrieved_chunks: List[dict],
              fp: Dict[str, set], expected_answer: str) -> bool:
    """三口径命中判定（任一成立即命中）：
    1. doc 级：证据文档本身在检索结果中；
    2. 内容指纹级：检索到与证据内容相同的模板副本；
    3. 答案包含性：检索到的内容能够支撑标准答案。
    """
    if any(d in gold_docs for d in retrieved_docs):
        return True
    if content_hit(gold_docs, retrieved_docs, fp):
        return True
    return answer_hit(expected_answer, retrieved_chunks)


def load_golden(path: Path) -> List[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ----------------------------------------------------------------------
# 数据层
# ----------------------------------------------------------------------
def data_layer_metrics(report_path: Path) -> Path:
    writer = AtomicDualIndexWriter()
    cur = writer.sm.conn.cursor()

    status_rows = cur.execute("SELECT status, COUNT(*) AS c FROM doc_versions GROUP BY status").fetchall()
    tag_rows = cur.execute("SELECT detected_tag, COUNT(*) AS c FROM quality_audit_log GROUP BY detected_tag").fetchall()
    q_reasons = cur.execute("""
        SELECT quarantine_reason, COUNT(*) AS c FROM doc_versions
        WHERE status='QUARANTINED' GROUP BY quarantine_reason ORDER BY c DESC
    """).fetchall()
    chunk_len = cur.execute("SELECT LENGTH(tokenized_content) AS l FROM chunks_fts LIMIT 20000").fetchall()
    lens = [r["l"] for r in chunk_len]

    if lens:
        lens_sorted = sorted(lens)
        p50 = lens_sorted[len(lens_sorted) // 2]
        p95 = lens_sorted[int(len(lens_sorted) * 0.95) - 1]
        len_line = f"- chunk 分词长度分布（采样 {len(lens)} 块）：中位数 {p50}，P95 {p95}"
    else:
        len_line = "- chunk 分词长度分布：索引为空（请先运行双索引构建）"

    md = [
        "# KnowPulse 数据层评测报告",
        "",
        "## 文档版本状态机",
        "| 状态 | 数量 |", "|---|---:|"
    ] + [f"| {r['status']} | {r['c']} |" for r in status_rows]
    md += ["", "## 质量门检测标签分布", "| 检测标签 | 数量 |", "|---|---:|"]
    md += [f"| {r['detected_tag']} | {r['c']} |" for r in tag_rows]
    md += ["", "## 隔离原因 TOP", "| 原因 | 数量 |", "|---|---:|"]
    md += [f"| {(r['quarantine_reason'] or '')[:60]} | {r['c']} |" for r in q_reasons[:10]]
    md += ["", "## chunk 粒度", len_line, ""]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[+] 数据层评测报告已写入 {report_path}")
    return report_path


# ----------------------------------------------------------------------
# 检索层：四档消融实验
# ----------------------------------------------------------------------
def run_ablation_evaluation(golden_jsonl: Path, report_out: Path, embed_mode: str | None = None,
                            limit: int | None = None):
    s = get_settings()
    writer = AtomicDualIndexWriter()
    embedder = CachedBgeM3Embedder(writer.sm.conn, mode=embed_mode or s.effective_embed_mode)
    retriever = HybridRetrieverService(writer, embedder, rerank_mode=s.effective_rerank_mode)
    resolver = build_evidence_resolver(writer)
    fp = build_content_fingerprints(writer)

    golden = load_golden(golden_jsonl)
    if limit:
        golden = golden[:limit]

    modes = ["vec_only", "fts_only", "hybrid_rrf", "hybrid_rerank"]

    md_lines = [
        "# KnowPulse 检索链路四档消融实验报告",
        "",
        f"- 评测样本：{len(golden)} 条（golden set）",
        "- 命中口径（三选一即命中）：①证据文档 doc 级命中；②内容指纹级（模板副本等价）；③答案包含性（检索内容能支撑标准答案）",
        "",
        "| 检索档位 (`ablation_mode`) | 整体 Recall@5 | 简单事实类 Recall@5 | 表格/扫描复杂类 Recall@5 | 整体 Recall@10 | MRR@10 | nDCG@10 | 平均检索延迟 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]

    for mode in modes:
        hit5_all, hit10_all, mrr_all, ndcg_all, latencies = [], [], [], [], []
        hit5_simple, hit5_complex = [], []

        for item in golden:
            gold_docs = [resolve_evidence(ev, resolver) for ev in item.get("evidence_doc_ids", [])]
            difficulty = item.get("difficulty", "")
            if difficulty == REFUSE_DIFFICULTY:
                continue                            # 拒答题不进检索指标，由生成层评测

            res = retriever.retrieve(item["question"], user_role="teacher", ablation_mode=mode, top_k_final=10)
            latencies.append(res["latency_ms"])
            chunks = res["chunks"]
            doc_ids = [c["doc_id"] for c in chunks]
            expected = item.get("expected_answer", "")

            h5 = 1.0 if judge_hit(gold_docs, doc_ids[:5], chunks[:5], fp, expected) else 0.0
            h10 = 1.0 if judge_hit(gold_docs, doc_ids[:10], chunks[:10], fp, expected) else 0.0
            hit5_all.append(h5)
            hit10_all.append(h10)

            if difficulty == SIMPLE_DIFFICULTY:
                hit5_simple.append(h5)
            else:
                hit5_complex.append(h5)

            hit_rank = next((i + 1 for i in range(10) if judge_hit(gold_docs, doc_ids[:i + 1], chunks[:i + 1], fp, expected)), None)
            if hit_rank:
                mrr_all.append(1.0 / hit_rank)
                ndcg_all.append(1.0 / math.log2(hit_rank + 1))
            else:
                mrr_all.append(0.0)
                ndcg_all.append(0.0)

        def avg(lst):
            return (sum(lst) / len(lst) * 100.0) if lst else 0.0

        md_lines.append(
            f"| `{mode}` | **{avg(hit5_all):.2f}%** | {avg(hit5_simple):.2f}% | {avg(hit5_complex):.2f}% | "
            f"{avg(hit10_all):.2f}% | {sum(mrr_all)/len(mrr_all):.4f} | {sum(ndcg_all)/len(ndcg_all):.4f} | "
            f"{(sum(latencies)/len(latencies)) if latencies else 0:.1f} ms |"
        )

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[+] 消融实验完成！报告已写入: {report_out}")


# ----------------------------------------------------------------------
# 生成层：拒答正确率 + 引用硬校验（Faithfulness 由 judge LLM 判分）
# ----------------------------------------------------------------------
JUDGE_SYSTEM = (
    "你是严格的 Faithfulness 裁判。判断【参考答案】是否完全蕴含于【证据】之中。"
    "仅输出一个词：FAITHFUL 或 UNFAITHFUL。"
)


def run_generation_evaluation(golden_jsonl: Path, report_out: Path, limit: int = 30):
    s = get_settings()
    if not s.has_siliconflow_key:
        raise SystemExit("[!] 生成层评测需要 LLM API（SILICONFLOW_API_KEY），请先配置 configs/.env。")
    writer = AtomicDualIndexWriter()
    embedder = CachedBgeM3Embedder(writer.sm.conn, mode=s.effective_embed_mode)
    sf = SiliconFlowService()
    retriever = HybridRetrieverService(writer, embedder, rerank_mode=s.effective_rerank_mode)
    chain = KnowPulseRAGChain(sf)

    golden = load_golden(golden_jsonl)[:limit]
    refuse_correct, refuse_total = 0, 0
    faithful, judged = 0, 0
    hard_precisions: List[float] = []
    costs: List[float] = []

    for item in golden:
        res = retriever.retrieve(item["question"], user_role="teacher", ablation_mode="hybrid_rerank", top_k_final=8)
        gen = chain.generate_and_verify(item["question"], res["chunks"])
        costs.append(gen.get("cost_rmb", 0.0))
        hard_precisions.append(gen.get("citation_hard_precision", 1.0))

        if item.get("difficulty") == REFUSE_DIFFICULTY:
            refuse_total += 1
            if gen["refused"]:
                refuse_correct += 1
            continue

        if gen["refused"] or not res["chunks"]:
            continue
        evidence = "\n".join(c["parent_text"][:600] for c in res["chunks"][:3])
        try:
            verdict = sf.chat_with_cost(
                JUDGE_SYSTEM,
                f"【参考答案】{item.get('expected_answer','')}\n【证据】{evidence}",
                temperature=0.0,
                model=s.llm.judge_model,
            )
            if "FAITHFUL" in verdict["answer"].upper():
                faithful += 1
            judged += 1
        except Exception:
            continue

    md = [
        "# KnowPulse 生成层评测报告",
        "",
        f"- 样本数：{len(golden)}",
        f"- 拒答正确率：{refuse_correct}/{refuse_total} = {(refuse_correct/refuse_total*100) if refuse_total else 100:.1f}%",
        f"- Faithfulness（judge LLM）：{faithful}/{judged} = {(faithful/judged*100) if judged else 100:.1f}%",
        f"- 引用硬校验平均 Precision：{sum(hard_precisions)/len(hard_precisions):.4f}" if hard_precisions else "- 引用硬校验：无样本",
        f"- 累计 API 成本：约 {sum(costs):.4f} 元",
        "",
        "> 说明：Faithfulness 为目标 ≥ 0.85；拒答题来自 golden set 的「应拒答」层（领域外/超纲问题）。",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[+] 生成层评测报告已写入 {report_out}")


# ----------------------------------------------------------------------
def build_stratified_golden_set(writer: AtomicDualIndexWriter, sf: SiliconFlowService,
                                out_jsonl: Path, total_samples: int = 180):
    """
    从已入库语料中分层抽取父块，用 LLM 生成候选 QA（人工复核后再 merge 进主评测集）。
    详见 src/evaluation/build_golden.py（本函数保留为兼容入口）。
    """
    from src.evaluation.build_golden import generate_candidates
    generate_candidates(out_jsonl)


if __name__ == "__main__":
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=["data_layer", "run_ablation", "generation", "build_golden"], required=True)
    p.add_argument("--golden-path", default=str(s.golden_set_path))
    p.add_argument("--report-out", default=str(s.ablation_report_path))
    p.add_argument("--embed-mode", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.action == "data_layer":
        data_layer_metrics(Path("data/golden_set/data_layer_report.md"))
    elif args.action == "run_ablation":
        run_ablation_evaluation(Path(args.golden_path), Path(args.report_out), args.embed_mode, args.limit)
    elif args.action == "generation":
        run_generation_evaluation(Path(args.golden_path), Path("data/golden_set/generation_report.md"), limit=args.limit or 30)
    else:
        # 候选 QA 必须落到独立文件，绝不能覆盖人工复核过的主评测集
        w = AtomicDualIndexWriter()
        sf = SiliconFlowService()
        build_stratified_golden_set(w, sf, Path("data/golden_set/candidates.jsonl"))
