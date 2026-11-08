# -*- coding: utf-8 -*-
"""
引用双层校验（方案 v1.2 第 8.3 节）：

1. 在线硬校验（1ms 级）：正则提取回答中的 [doc_id:chunk_id]，校验其真实存在于
   本次召回结果 —— 拦截"编造编号"型幻觉；
2. 离线软校验（NLI 蕴含裁判）：将带引用的回答句作为 Hypothesis，对应 chunk 原文
   作为 Premise，用裁判 LLM 判断蕴含关系，计算 Citation Precision / Recall。
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from src.llm.siliconflow_client import SiliconFlowService

CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+):([A-Za-z0-9_\-#]+)\]")

NLI_SYSTEM_PROMPT = (
    "你是严格的 NLI 自然语言推断裁判。判断【前提Premise】是否蕴含【假设Hypothesis】中的事实。"
    "仅输出一个词：ENTAILMENT 或 CONTRADICTION。"
)


def hard_check(answer: str, retrieved_chunks: List[Dict]) -> Tuple[str, List[str], float]:
    """
    在线硬校验。

    返回 (清洗后答案[剔除编造引用标签], 编造引用标签列表, Citation Precision)。
    当答案没有任何引用时：若属于无证据拒答 -> Precision=1.0，否则 0.0。
    """
    valid_tags: Set[Tuple[str, str]] = {(c["doc_id"], c["chunk_id"]) for c in retrieved_chunks}
    cited = CITATION_RE.findall(answer)
    hallucinated = [f"[{d}:{cid}]" for d, cid in cited if (d, cid) not in valid_tags]

    cleaned = answer
    for bad in hallucinated:
        cleaned = cleaned.replace(bad, "")

    if cited:
        precision = (len(cited) - len(hallucinated)) / len(cited)
    else:
        precision = 1.0 if "无证据拒答" in answer else 0.0
    return cleaned, hallucinated, round(precision, 4)


def soft_check_nli(answer: str, retrieved_chunks: List[Dict], sf: SiliconFlowService,
                   judge_model: str) -> dict:
    """
    离线软校验（Citation Precision / Recall）。

    需要可用的 LLM API；仅对"带引用标签的句子"逐句判定蕴含关系。
    返回 {precision, recall, checked, entailed, total_citations}。
    """
    chunk_map = {f"{c['doc_id']}:{c['chunk_id']}": c["parent_text"] for c in retrieved_chunks}
    sentences = [s.strip() for s in re.split(r"[。！？\n]", answer) if len(s.strip()) > 5]

    total_citations = 0
    checked = 0
    entailed = 0
    for sent in sentences:
        matches = CITATION_RE.findall(sent)
        for d, cid in matches:
            total_citations += 1
            premise = chunk_map.get(f"{d}:{cid}")
            if not premise:
                continue                          # 引用不在本次召回中（硬校验的漏网）-> 判为不蕴含
            checked += 1
            try:
                res = sf.chat_with_cost(
                    NLI_SYSTEM_PROMPT,
                    f"【前提Premise】：{premise}\n【假设Hypothesis】：{sent}",
                    temperature=0.0,
                    model=judge_model,
                )
                if "ENTAILMENT" in res["answer"].upper():
                    entailed += 1
            except Exception:
                continue

    precision = round(entailed / checked, 4) if checked else 1.0
    recall = round(entailed / total_citations, 4) if total_citations else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "checked": checked,
        "entailed": entailed,
        "total_citations": total_citations,
    }
