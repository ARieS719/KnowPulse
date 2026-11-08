# -*- coding: utf-8 -*-
"""
Badcase 归因与回归闭环（Phase 5，核心差异化 2）。

`analyze`   : 读取 /feedback 沉淀的 badcases.jsonl，逐条重跑检索，按"解析/切片/检索/生成"
              四层做启发式归因，输出 badcase_report.md；
`regression`: 对比两份消融报告（改动前/改动后），量化每个指标的涨跌，输出 regression_report.md。

闭环留痕：记录"badcase -> 归因 -> 修复 -> 指标回升"的完整迭代过程。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from src.config import get_settings
from src.indexing.dual_writer import AtomicDualIndexWriter
from src.indexing.embedder import CachedBgeM3Embedder
from src.llm.siliconflow_client import SiliconFlowService
from src.retrieval.hybrid_search import HybridRetrieverService

ROOT_CAUSES = ("检索层", "生成层", "拒答误伤", "未分类")


def _attribution(query: str, retriever: HybridRetrieverService) -> Dict:
    res = retriever.retrieve(query, user_role="teacher", ablation_mode="hybrid_rerank", top_k_final=8)
    chunks = res["chunks"]
    if not chunks:
        return {"root_cause": "检索层", "detail": "候选集为空：解析/切片/检索环节未产出候选", "max_score": 0.0, "n_chunks": 0}

    scores = [c.get("rerank_score", 0.0) for c in chunks]
    max_score = max(scores) if scores else 0.0
    n_chunks = len(chunks)

    if max_score < 0.05:
        # 有候选但重排置信度极低 -> 检索层召回不相关；或质量门拦截过头
        return {"root_cause": "检索层", "detail": "Top8 最高重排分 < 0.05：召回内容与问题不相关（检索层）", "max_score": round(max_score, 4), "n_chunks": n_chunks}
    return {
        "root_cause": "生成层",
        "detail": "证据充分但用户点踩：大概率生成层问题（漏引/错引/回答不忠实），建议跑引用软校验",
        "max_score": round(max_score, 4),
        "n_chunks": n_chunks,
    }


def analyze_badcases(badcases_path: Path, report_path: Path, limit: int = 200) -> Path:
    s = get_settings()
    writer = AtomicDualIndexWriter()
    embedder = CachedBgeM3Embedder(writer.sm.conn, mode=s.effective_embed_mode)
    sf = SiliconFlowService()
    retriever = HybridRetrieverService(writer, embedder, rerank_mode=s.effective_rerank_mode)

    records: List[dict] = []
    if badcases_path.exists():
        for line in badcases_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records = records[:limit]

    stats = {k: 0 for k in ROOT_CAUSES}
    details: List[str] = []
    for i, rec in enumerate(records, 1):
        q = rec.get("question", "")
        if not q:
            continue
        attr = _attribution(q, retriever)
        stats[attr["root_cause"]] += 1
        details.append(f"| {i} | {q[:40]} | {attr['root_cause']} | {attr['detail']} | {attr['max_score']} | {attr['n_chunks']} |")

    md = [
        "# KnowPulse Badcase 归因报告",
        "",
        f"- 分析条目：{sum(stats.values())}",
        f"- 归因分布：{ {k: stats[k] for k in ROOT_CAUSES} }",
        "",
        "| # | 问题 | 归因层 | 判定依据 | Top1 重排分 | 候选数 |",
        "|---|---|---|---:|---:|---:|",
    ] + details
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[+] badcase 归因报告已写入 {report_path}")
    return report_path


# ----------------------------------------------------------------------
_TABLE_ROW_RE = re.compile(r"^\|\s*`?(\w+)`?\s*\|")
_METRIC_RE = re.compile(r"^\|\s*`(\w+)`\s*\|")


def _parse_ablation(path: Path) -> Dict[str, Dict[str, float]]:
    """解析消融报告表格：mode -> {metric: value}。数值列为百分数/小数/毫秒混合。"""
    out: Dict[str, Dict[str, float]] = {}
    headers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| 检索档位"):
            headers = [h.strip().strip("`") for h in line.strip("|").split("|")]
            continue
        m = _TABLE_ROW_RE.match(line)
        if not m or not headers:
            continue
        cells = [c.strip().strip("**") for c in line.strip("|").split("|")]
        mode = cells[0].strip("`")
        out[mode] = {}
        for h, v in zip(headers[1:], cells[1:]):
            vm = re.search(r"([\d.]+)", v)
            if vm:
                out[mode][h] = float(vm.group(1))
    return out


def regression_compare(old_path: Path, new_path: Path, report_path: Path) -> Path:
    old = _parse_ablation(old_path)
    new = _parse_ablation(new_path)

    md = ["# KnowPulse 回归对比报告", "", f"- 基线报告：{old_path}", f"- 当前报告：{new_path}", ""]
    for mode in sorted(set(old) | set(new)):
        md.append(f"## `{mode}`")
        md.append("")
        md.append("| 指标 | 改动前 | 改动后 | 变化 |")
        md.append("|---|---:|---:|---:|")
        metrics = sorted(set(old.get(mode, {})) | set(new.get(mode, {})))
        for metric in metrics:
            a = old.get(mode, {}).get(metric)
            b = new.get(mode, {}).get(metric)
            if a is None or b is None:
                delta = "—"
            else:
                delta = f"{b - a:+.2f}"
            fa = f"{a:.2f}" if a is not None else "—"
            fb = f"{b:.2f}" if b is not None else "—"
            md.append(f"| {metric} | {fa} | {fb} | {delta} |")
        md.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[+] 回归对比报告已写入 {report_path}")
    return report_path


if __name__ == "__main__":
    s = get_settings()
    p = argparse.ArgumentParser(description="Badcase 归因与回归对比")
    p.add_argument("--action", choices=["analyze", "regression"], required=True)
    p.add_argument("--badcases", default=str(s.badcases_path))
    p.add_argument("--old", help="改动前消融报告路径")
    p.add_argument("--new", help="改动后消融报告路径")
    p.add_argument("--report-out", default="data/golden_set/regression_report.md")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    if args.action == "analyze":
        analyze_badcases(Path(args.badcases), Path("data/golden_set/badcase_report.md"), args.limit)
    else:
        if not args.old or not args.new:
            raise SystemExit("regression 需要 --old 与 --new 两份消融报告路径")
        regression_compare(Path(args.old), Path(args.new), Path(args.report_out))
