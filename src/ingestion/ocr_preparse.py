# -*- coding: utf-8 -*-
"""
扫描件/图片 OCR 预解析（在 OCR 专用虚拟环境 .venv-ocr 中运行）。

背景：paddlepaddle 2.6 要求 protobuf<=3.20.2，与主环境 chromadb 链路
（protobuf>=6.33.5）不兼容，因此 OCR 在独立虚拟环境中执行，识别结果落盘
为解析缓存，由主流水线（parser_router）消费：

- 扫描件 PDF -> data/parsed/{doc_id}.md（逐页文本，页间空行分隔）
- 题目图片   -> data/parsed/{doc_id}.ocr.txt

止损规则：扫描件的平均 OCR 置信度低于 0.75 时不写缓存，保持标题级兜底
（对应方案"低置信度路由质量门"的时间盒止损设计）。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import fitz  # PyMuPDF（随 paddleocr 安装）
import numpy as np
from paddleocr import PaddleOCR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST = PROJECT_ROOT / "data" / "meta" / "manifest.csv"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"

MIN_CONFIDENCE = 0.75


def ocr_scanned_pdf(ocr, pdf_path: Path):
    """返回 (全文, 平均置信度)。无文字行时平均置信度为 None。"""
    doc = fitz.open(str(pdf_path))
    pages, scores = [], []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            res = ocr.ocr(arr, cls=True)
            lines = []
            for block in (res or []):
                for line in (block or []):
                    lines.append(line[1][0])
                    scores.append(float(line[1][1]))
            pages.append("\n".join(lines))
    finally:
        doc.close()
    text = "\n\n".join(p for p in pages if p.strip()).strip()
    avg = (sum(scores) / len(scores)) if scores else None
    return text, avg


def ocr_image(ocr, img_path: Path) -> str:
    try:
        res = ocr.ocr(str(img_path), cls=True)
        return " ".join(line[1][0] for block in (res or []) for line in (block or []))
    except Exception:
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=["scanned", "images"],
                   choices=["scanned", "images"])
    p.add_argument("--limit", type=int, default=None, help="每类只处理前 N 份（冒烟用）")
    args = p.parse_args()

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    targets = [r for r in rows if r["category"] in args.categories]

    done = skipped = 0
    for i, r in enumerate(targets):
        if args.limit and i >= args.limit:
            break
        doc_id = r["doc_id"]
        src = DATA_ROOT / r["path"]

        if r["category"] == "scanned":
            text, avg = ocr_scanned_pdf(ocr, src)
            if not text or (avg is not None and avg < MIN_CONFIDENCE):
                skipped += 1
                if (i + 1) % 50 == 0 or i + 1 == len(targets):
                    print(f"[{i + 1}/{len(targets)}] {doc_id} 跳过（置信度 {avg if avg is not None else 'N/A'}）", flush=True)
                continue
            (PARSED_DIR / f"{doc_id}.md").write_text(text, encoding="utf-8")
        else:
            text = ocr_image(ocr, src)
            if not text.strip():
                skipped += 1
                continue
            (PARSED_DIR / f"{doc_id}.ocr.txt").write_text(text.strip(), encoding="utf-8")

        done += 1
        if (i + 1) % 50 == 0 or i + 1 == len(targets):
            print(f"[{i + 1}/{len(targets)}] 完成 {doc_id}（累计成功 {done}，跳过 {skipped}）", flush=True)

    print(f"预解析结束：成功 {done}，跳过 {skipped}")


if __name__ == "__main__":
    main()
