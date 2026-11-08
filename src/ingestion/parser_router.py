# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List
import docx
import openpyxl
from src.ingestion.table_stitcher import stitch_markdown_tables
from src.quality.operators import CalibratedQualityOperators

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None


# ----------------------------------------------------------------------
# 字段提取工具：兼容英文键与中文键，清洗 "None"/空字符串，列表拼接
# ----------------------------------------------------------------------
def _first_str(item: dict, *keys: str) -> str:
    for k in keys:
        v = item.get(k)
        if isinstance(v, str) and v.strip() and v.strip().lower() != "none":
            return v.strip()
    return ""


def _first_any(item: dict, *keys: str):
    for k in keys:
        v = item.get(k)
        if v is not None and v != "" and v != "None":
            return v
    return ""


def _clean_cell(v) -> str:
    if v is None or v == "None":
        return ""
    return str(v).strip()


class MultiFormatParserRouter:
    """
    针对 4,502 份 clean 语料的五路解析器：
    1. question_bank (.json, 2002份) -> 提取每道题的 题干+选项+答案+解析+知识点
    2. lectures (.docx, 1000份) -> 提取 Heading 1/2/3 标题树 + 段落 + 表格
    3. tables (.xlsx, 500份) -> 生成 [子块摘要: 表名+表头+前3行+行数统计] 与 [父块: 完整 Markdown 表格]
    4. scanned (.pdf, 700份) -> 优先读取 MinerU 解析产出目录 (data/parsed/)，若无则回退 pypdfium2/PaddleOCR 提取并校验置信度 >= 0.75
    5. images (.jpg, 300份) -> 提取题目卡片 OCR 文本作为子块，原图路径挂载至父块
    """
    def __init__(self, parsed_cache_dir: str = "data/parsed"):
        self.parsed_cache_dir = Path(parsed_cache_dir)
        self.parsed_cache_dir.mkdir(parents=True, exist_ok=True)
        self._ocr_engine = None

    def _get_ocr(self):
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            except Exception:
                self._ocr_engine = False
        return self._ocr_engine

    def parse_document(self, doc_meta: dict, full_path: Path) -> List[Dict]:
        """
        返回标准化的父子块列表：
        [{"parent_id": str, "child_id": str, "section_path": str, "child_text": str, "parent_text": str, "media_path": str}]
        """
        ext = full_path.suffix.lower()
        doc_id = doc_meta["base_doc_id"]
        title = doc_meta["title"]
        subj = doc_meta["subject"]

        if ext == ".json":
            return self._parse_question_bank_json(doc_id, title, subj, full_path)
        elif ext == ".docx":
            return self._parse_lecture_docx(doc_id, title, subj, full_path)
        elif ext == ".xlsx":
            return self._parse_table_xlsx(doc_id, title, subj, full_path)
        elif ext == ".pdf":
            return self._parse_scanned_pdf(doc_id, title, subj, full_path)
        elif ext in {".jpg", ".jpeg", ".png"}:
            return self._parse_image_card(doc_id, title, subj, full_path)
        return []

    def _parse_question_bank_json(self, doc_id: str, title: str, subj: str, path: Path) -> List[Dict]:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else []
        if isinstance(data, dict):
            for k in ("questions", "items", "data", "problems", "records", "list"):
                if isinstance(data.get(k), list):
                    items = data[k]
                    break
            if not items:
                items = [data]

        chunks = []
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            q_text = _first_str(item, "question", "stem", "content", "题干", "题目")
            opts = _first_any(item, "options", "choices", "选项")
            if isinstance(opts, list):
                opts = " ".join(_clean_cell(x) for x in opts)
            else:
                opts = _clean_cell(opts)
            ans = _clean_cell(_first_any(item, "answer", "答案"))
            analysis = _first_str(item, "analysis", "explanation", "解析")
            kp = _first_any(item, "knowledge_point", "topic", "知识点", "学科", "subject")
            if isinstance(kp, list):
                kp = "、".join(_clean_cell(x) for x in kp)
            kp = _clean_cell(kp) or subj

            # 子块：仅含标题+知识点+题干+选项（检索最精准，不被冗长解析稀释）
            child_raw = f"【{subj}·{title}·知识点:{kp}】题目{idx}：{q_text} {opts}".strip()
            # 父块：含完整题干+选项+标准答案+详细解析（喂给大模型生成有理有据的回答）
            parent_raw = f"【文档：{title} ({doc_id}) | 学科：{subj} | 知识点：{kp}】\n【题目{idx}】{q_text}\n【选项】{opts}\n【标准答案】{ans}\n【详细解析】{analysis}".strip()

            child_clean, _ = CalibratedQualityOperators.mask_pii(child_raw)
            parent_clean, _ = CalibratedQualityOperators.mask_pii(parent_raw)

            pid = f"{doc_id}#P{idx:03d}"
            cid = f"{doc_id}#C{idx:03d}"
            chunks.append({
                "parent_id": pid,
                "child_id": cid,
                "section_path": f"{title} > 第{idx}题",
                "child_text": child_clean,
                "parent_text": parent_clean,
                "media_path": ""
            })
        return chunks

    def _parse_lecture_docx(self, doc_id: str, title: str, subj: str, path: Path) -> List[Dict]:
        doc = docx.Document(str(path))
        sections = []
        cur_h1, cur_h2 = title, "正文"
        buf = []

        def flush_buf(sec_idx: int):
            if not buf:
                return
            full_sec = "\n".join(buf).strip()
            buf.clear()
            if len(full_sec) < 15:
                return
            sec_path = f"{cur_h1} > {cur_h2}"
            pid = f"{doc_id}#P{sec_idx:03d}"
            parent_clean, _ = CalibratedQualityOperators.mask_pii(f"【讲义：{sec_path}】\n{full_sec}")
            # 将父块按 280 字切分为多个子块，共享同一个 parent_id
            step = 260
            for sub_i, start in enumerate(range(0, len(full_sec), step), 1):
                piece = full_sec[start:start + 300]
                child_clean, _ = CalibratedQualityOperators.mask_pii(f"【{subj}·{sec_path}】{piece}")
                sections.append({
                    "parent_id": pid,
                    "child_id": f"{pid}_C{sub_i:02d}",
                    "section_path": sec_path,
                    "child_text": child_clean,
                    "parent_text": parent_clean,
                    "media_path": ""
                })

        sec_counter = 1
        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            style_name = (p.style.name or "").lower()
            if "heading 1" in style_name or txt.startswith("一、") or txt.startswith("二、") or txt.startswith("三、"):
                flush_buf(sec_counter)
                sec_counter += 1
                cur_h1 = txt
                cur_h2 = "正文"            # 一级标题切换时重置二级标题，防止 section_path 串节
                buf.append(txt)            # 一级标题文本本身也是内容的一部分
            elif "heading 2" in style_name or "heading 3" in style_name:
                # 注意：本语料的讲义把正文内容段也写成 Heading 2 样式——
                # 必须"既更新小节路径、又保留文本为内容"，否则讲义正文会被整段丢弃
                flush_buf(sec_counter)
                sec_counter += 1
                cur_h2 = txt
                buf.append(txt)
            else:
                buf.append(txt)
                if sum(len(x) for x in buf) >= 850:
                    flush_buf(sec_counter)
                    sec_counter += 1

        # 提取讲义内嵌表格
        for t_idx, table in enumerate(doc.tables, 1):
            rows_md = []
            for r in table.rows:
                rows_md.append("| " + " | ".join(c.text.strip().replace("\n", " ") for c in r.cells) + " |")
            if len(rows_md) >= 2:
                header_sep = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
                rows_md.insert(1, header_sep)
                buf.append("\n".join(rows_md))

        flush_buf(sec_counter)
        return sections

    def _parse_table_xlsx(self, doc_id: str, title: str, subj: str, path: Path) -> List[Dict]:
        """大表格摘要索引：子块=表名+表头+前3行数据，父块=完整 Markdown 表格"""
        wb = openpyxl.load_workbook(str(path), data_only=True)
        chunks = []
        for s_idx, sheet_name in enumerate(wb.sheetnames, 1):
            ws = wb[sheet_name]
            rows = [[str(cell) if cell is not None else "" for cell in row] for row in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c.strip() for c in r)]
            if not rows:
                continue
            headers = rows[0]
            md_lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"
            ]
            for r in rows[1:]:
                md_lines.append("| " + " | ".join(r[:len(headers)]) + " |")

            full_table_md = f"【表格文档：{title} | 工作表：{sheet_name} | 学科：{subj} | 共 {len(rows)-1} 行数据】\n" + "\n".join(md_lines)
            sample_rows = [" | ".join(r[:len(headers)]) for r in rows[1:4]]
            child_summary = (
                f"【{subj}表格：{title} - {sheet_name}】列名：{', '.join(headers)}。"
                f"数据行数：{len(rows)-1}行。前3行示例：{' ； '.join(sample_rows)}"
            )
            parent_clean, _ = CalibratedQualityOperators.mask_pii(full_table_md)
            child_clean, _ = CalibratedQualityOperators.mask_pii(child_summary)
            pid = f"{doc_id}#P{s_idx:03d}"
            chunks.append({
                "parent_id": pid,
                "child_id": f"{pid}_C01",
                "section_path": f"{title} > {sheet_name}",
                "child_text": child_clean,
                "parent_text": parent_clean,
                "media_path": ""
            })
        wb.close()
        return chunks

    def _parse_scanned_pdf(self, doc_id: str, title: str, subj: str, path: Path) -> List[Dict]:
        """
        优先读取 MinerU 预解析缓存 data/parsed/{doc_id}.md（读取时做跨页表格拼接修复）；
        若不存在，先尝试 pypdfium2 文本层提取，若为纯图片扫描件且已安装 PaddleOCR 则执行 OCR 并检查置信度 >= 0.75。
        注意：兜底文本（无 OCR 环境）只用于本次索引，不写入解析缓存，避免污染后续真实解析结果。
        """
        cached_md = self.parsed_cache_dir / f"{doc_id}.md"
        text_content = ""
        real_content = False
        if cached_md.exists():
            md_text = cached_md.read_text(encoding="utf-8")
            md_text, _stitches = stitch_markdown_tables(md_text)   # MinerU 跨页表格拼接修复
            text_content = md_text
            real_content = True
        elif pdfium is not None:
            pdf = pdfium.PdfDocument(str(path))
            pages_txt = []
            ocr = self._get_ocr()
            scores = []
            for p_i in range(len(pdf)):
                page = pdf[p_i]
                tp = page.get_textpage()
                t = tp.get_text_range().strip()
                if len(t) > 20:
                    pages_txt.append(t)
                elif ocr:
                    pil_img = page.render(scale=1.5).to_pil()
                    import numpy as np
                    res = ocr.ocr(np.asarray(pil_img), cls=True)
                    line_txts = []
                    for block in (res or []):
                        for line in (block or []):
                            line_txts.append(line[1][0])
                            scores.append(float(line[1][1]))
                    pages_txt.append("\n".join(line_txts))
            pdf.close()
            # 4天时间盒止损规则：若 OCR 平均置信度 < 0.75，记录并返回空（由质量门标记需人工复核）
            if scores and (sum(scores) / len(scores)) < 0.75:
                return []
            text_content = "\n\n".join(pages_txt).strip()
            real_content = bool(text_content)

        if not text_content:
            # 兜底用标题与元数据构建基础索引块，确保 700 份扫描件在无 OCR 环境下也不丢文档级检索
            text_content = f"扫描版试卷文档：《{title}》，所属学科：{subj}，文件编号：{doc_id}。"

        if real_content:
            cached_md.write_text(text_content, encoding="utf-8")

        pid = f"{doc_id}#P001"
        parent_clean, _ = CalibratedQualityOperators.mask_pii(f"【扫描卷：{title} ({subj})】\n{text_content[:1200]}")
        chunks = []
        for idx, start in enumerate(range(0, max(1, len(text_content)), 280), 1):
            piece = text_content[start:start + 320] or title
            child_clean, _ = CalibratedQualityOperators.mask_pii(f"【{subj}·{title}】{piece}")
            chunks.append({
                "parent_id": pid,
                "child_id": f"{pid}_C{idx:02d}",
                "section_path": f"{title} > 第{idx}段",
                "child_text": child_clean,
                "parent_text": parent_clean,
                "media_path": str(path)
            })
        return chunks

    def _parse_image_card(self, doc_id: str, title: str, subj: str, path: Path) -> List[Dict]:
        """图片摘要索引：子块=图注+OCR题干，父块=原图路径挂载。
        OCR 文本优先读取预解析缓存 data/parsed/{doc_id}.ocr.txt（由 OCR 专用环境生成），
        缓存不存在且当前环境装有 OCR 引擎时现场识别。"""
        cached_txt = self.parsed_cache_dir / f"{doc_id}.ocr.txt"
        ocr_text = ""
        if cached_txt.exists():
            ocr_text = cached_txt.read_text(encoding="utf-8").strip()
        else:
            ocr = self._get_ocr()
            if ocr:
                try:
                    res = ocr.ocr(str(path), cls=True)
                    lines = [line[1][0] for block in (res or []) for line in (block or [])]
                    ocr_text = " ".join(lines)
                except Exception:
                    ocr_text = ""
        summary = f"【{subj}题目卡片图：{title}】{ocr_text}".strip()
        child_clean, _ = CalibratedQualityOperators.mask_pii(summary)
        parent_clean, _ = CalibratedQualityOperators.mask_pii(
            f"【原图卡片：{title} ({doc_id}) | 学科：{subj}】\nOCR识别内容：{ocr_text or '见挂载原图'}"
        )
        pid = f"{doc_id}#P001"
        return [{
            "parent_id": pid,
            "child_id": f"{pid}_C01",
            "section_path": title,
            "child_text": child_clean,
            "parent_text": parent_clean,
            "media_path": str(path)
        }]