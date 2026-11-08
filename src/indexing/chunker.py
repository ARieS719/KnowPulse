# -*- coding: utf-8 -*-
"""
切片层：Markdown 标题树切分 + 父子块构建 + 断头块语义完整度修复。

设计要点（README design decisions 之二）：
- 父子块（parent-child）：检索用"子块"（短、精准），喂给 LLM 用"父块"（完整章节上下文）；
- 标题树切分：按 #/##/### 层级保留 section_path，切分边界与文档结构对齐；
- 滑动窗口子块带 40 字重叠，避免关键句被拦腰切断；
- 语义完整度校验（方案 v1.2 缺陷四修复项）：检出"表头与表格内容分离"
  "题目与选项分离"两类断头块并自动合并修复。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
OPTION_LINE_RE = re.compile(r"^\s*([A-DＡ-Ｄ])\s*[\.、．:：]\s*\S")


class MarkdownHeadingChunker:
    """把（MinerU 产出的）Markdown 长文档按标题树切分为父子块。"""

    def __init__(self, window: int = 300, stride: int = 260, min_section_len: int = 15):
        self.window = window
        self.stride = stride
        self.min_section_len = min_section_len
        self._seq_counter = 0

    # ------------------------------------------------------------------
    def split_sections(self, markdown_text: str) -> List[Tuple[str, str]]:
        """
        按标题树切分，返回 [(section_path, section_body), ...]。
        无标题的纯文本整体作为一个 section（section_path 为空串）。
        """
        lines = markdown_text.splitlines()
        sections: List[Tuple[str, str]] = []
        stack: List[Tuple[int, str]] = []          # [(level, title), ...]
        buf: List[str] = []

        def current_path() -> str:
            return " > ".join(t for _, t in stack) if stack else ""

        def flush() -> None:
            if buf:
                body = "\n".join(buf).strip()
                buf.clear()
                if len(body) >= self.min_section_len:
                    sections.append((current_path(), body))

        for line in lines:
            m = HEADING_RE.match(line.strip())
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
            else:
                buf.append(line)
        flush()
        if not sections:
            body = markdown_text.strip()
            if body:
                sections.append(("", body))
        return sections

    # ------------------------------------------------------------------
    @staticmethod
    def sliding_windows(text: str, window: int = 300, stride: int = 260) -> List[str]:
        """带重叠的滑动窗口子块。"""
        text = text.strip()
        if not text:
            return []
        if len(text) <= window:
            return [text]
        return [text[i:i + window] for i in range(0, max(1, len(text) - window + 1), stride)]

    # ------------------------------------------------------------------
    def build_parent_child(self, doc_id: str, title: str, section_path: str,
                           section_body: str) -> List[Dict]:
        """
        一个 section -> 1 个父块 + N 个子块（共享 parent_id）。
        """
        self._seq_counter += 1
        pid = f"{doc_id}#P{self._seq_counter:03d}"
        children = self.sliding_windows(section_body, self.window, self.stride)
        out: List[Dict] = []
        full_path = f"{title} > {section_path}".strip(" >")
        for i, piece in enumerate(children, 1):
            out.append({
                "parent_id": pid,
                "child_id": f"{pid}_C{i:02d}",
                "section_path": full_path,
                "child_text": piece,
                "parent_text": section_body,
                "media_path": "",
            })
        return out

# ----------------------------------------------------------------------
# 语义完整度修复（chunk 级）
# ----------------------------------------------------------------------
def detect_broken_chunks(chunks: List[Dict]) -> Tuple[List[Dict], int]:
    """
    检出两类断头块并合并修复，返回 (修复后的 chunks, 修复次数)：

    1. 表头断离：子块内容是表头行（| 列名 | 列名 |...）但下一块不是表体 ->
       将该块与下一块合并（表头 + 表体必须在同一检索单元内）；
    2. 选项断离：子块以 "A. xxx / B. xxx" 选项开头却缺少题干（无"题目"上下文的选项块）
       -> 与上一块合并，保证"题目+选项"不分离。
    """
    if len(chunks) < 2:
        return chunks, 0

    fixed: List[Dict] = []
    repairs = 0
    i = 0
    n = len(chunks)
    while i < n:
        cur = chunks[i]
        nxt = chunks[i + 1] if i + 1 < n else None

        # 规则 1：表头断离
        if nxt and _looks_like_table_header(cur["child_text"]) and not _looks_like_table_row(nxt["child_text"]):
            merged = _merge(cur, nxt)
            fixed.append(merged)
            repairs += 1
            i += 2
            continue

        # 规则 2：选项断离（选项块出现在题干块之后但独立成块）
        if nxt and _looks_like_option_fragment(nxt["child_text"]) and not _looks_like_option_fragment(cur["child_text"]):
            merged = _merge(cur, nxt)
            fixed.append(merged)
            repairs += 1
            i += 2
            continue

        fixed.append(cur)
        i += 1
    return fixed, repairs


def _looks_like_table_header(text: str) -> bool:
    """检测"仅表头 + 分隔行、无表体"的表格片段。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("|"):
        return False
    if len(lines) == 1 and "---" not in lines[0]:
        return True                     # 只有一行表头
    if len(lines) >= 2 and set(lines[1].replace("|", "").replace("-", "").replace(":", "").strip()) == set():
        return True                     # 表头 + 分隔行，但没有数据行
    return False


def _looks_like_table_row(text: str) -> bool:
    return bool(text) and text.strip().startswith("|")


def _looks_like_option_fragment(text: str) -> bool:
    """检测是否以选项 A./B. 开头且整块以选项为主。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return bool(OPTION_LINE_RE.match(lines[0])) and len(text) < 160


def _merge(a: Dict, b: Dict) -> Dict:
    return {
        "parent_id": a["parent_id"],
        "child_id": a["child_id"],
        "section_path": a["section_path"] or b["section_path"],
        "child_text": (a["child_text"] + "\n" + b["child_text"]).strip(),
        "parent_text": (a["parent_text"] + "\n" + b["parent_text"]).strip(),
        "media_path": a.get("media_path", "") or b.get("media_path", ""),
    }
