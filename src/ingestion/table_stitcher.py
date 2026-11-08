# -*- coding: utf-8 -*-
"""
跨页表格拼接修复算子（方案 v1.2 Phase 1 关键组件）。

背景：MinerU / 版面解析对"跨页表格"常输出为两段独立的 Markdown 表格，
第二段要么重复表头、要么直接是断裂的表体行。若不加修复：
1. 检索时"表头与表体分离"，表格数值类问题召回失败（断头块，见 chunker）；
2. 重复表头混入表体，LLM 拿到错误行列语义。

本模块提供纯文本层的拼接修复：
- 按连续表格行聚块（段落级分割）；
- 相邻两块列数一致时合并；
- 合并后清除块内重复出现的表头行与分隔行。
"""
from __future__ import annotations

import re
from typing import List, Tuple

TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEP_ROW_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


# ----------------------------------------------------------------------
def split_table_blocks(markdown_text: str) -> List[Tuple[int, List[str]]]:
    """
    把 Markdown 按段落切块，返回 [(起始行号, 行列表), ...]。
    表格行（|...|）单独聚成表格块；普通文本聚成文本块。
    """
    lines = markdown_text.splitlines()
    blocks: List[Tuple[int, List[str]]] = []
    cur_start, cur_lines, cur_is_table = 0, [], None

    def flush():
        nonlocal cur_start, cur_lines, cur_is_table
        if cur_lines:
            blocks.append((cur_start, cur_lines))
        cur_start, cur_lines, cur_is_table = 0, [], None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        is_table = bool(TABLE_ROW_RE.match(stripped))
        if cur_is_table is not None and is_table != cur_is_table:
            flush()
        if cur_is_table is None:
            cur_is_table = is_table
            cur_start = idx
        cur_lines.append(line)
    flush()
    return blocks


def _row_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_sep_row(line: str) -> bool:
    cells = _row_cells(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _is_header_dup(row: List[str], header: List[str]) -> bool:
    """判定一行是否为表头的重复出现。"""
    if len(row) != len(header):
        return False
    # 至少一半单元格与表头一致，且不全是数字（排除数据行恰好长得像表头）
    match = sum(1 for a, b in zip(row, header) if a == b)
    numeric = sum(1 for c in row if re.fullmatch(r"-?\d+(\.\d+)?%?", c))
    return match >= max(1, len(header) // 2) and numeric < len(header) // 2


def stitch_table_block(lines: List[str]) -> List[str]:
    """单个表格块内部的去重修复：只保留首个表头 + 分隔行，剔除跨页重复表头与多余分隔行。"""
    if not lines:
        return []
    sep_positions = [i for i, ln in enumerate(lines) if _is_sep_row(ln)]
    if not sep_positions:
        return lines

    first_sep = sep_positions[0]
    header = _row_cells(lines[first_sep - 1]) if first_sep > 0 else _row_cells(lines[0])
    out: List[str] = []
    if first_sep > 0:
        out.append(lines[first_sep - 1])   # 表头
        out.append(lines[first_sep])       # 分隔行
    for ln in lines[first_sep + 1:]:
        if _is_sep_row(ln):
            continue                       # 多余分隔行（跨页重复）丢弃
        if _is_header_dup(_row_cells(ln), header):
            continue                       # 跨页重复表头丢弃
        out.append(ln)
    return out


def stitch_markdown_tables(markdown_text: str) -> Tuple[str, int]:
    """
    主入口：跨页表格拼接。

    返回 (拼接后的 Markdown 全文, 拼接发生次数)。
    """
    blocks = split_table_blocks(markdown_text)
    merged: List[Tuple[int, List[str]]] = []
    repair_count = 0

    i = 0
    while i < len(blocks):
        start, lines = blocks[i]
        if TABLE_ROW_RE.match(lines[0].strip()):
            # 向后吞并相邻的表格块（列数一致）
            ncols = len(_row_cells(lines[0]))
            j = i + 1
            while j < len(blocks):
                nstart, nlines = blocks[j]
                if not (nlines and TABLE_ROW_RE.match(nlines[0].strip())):
                    break
                if len(_row_cells(nlines[0])) != ncols:
                    break
                lines = lines + nlines
                repair_count += 1
                j += 1
            i = j
        merged.append((start, lines))
        i += 1

    rebuilt: List[str] = []
    cursor = 0
    for start, lines in merged:
        rebuilt.extend(markdown_text.splitlines()[cursor:start])
        if TABLE_ROW_RE.match(lines[0].strip()):
            rebuilt.extend(stitch_table_block(lines))
        else:
            rebuilt.extend(lines)
        cursor = start + len(lines)
    rebuilt.extend(markdown_text.splitlines()[cursor:])
    return "\n".join(rebuilt), repair_count
