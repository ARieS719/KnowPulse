# -*- coding: utf-8 -*-
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from PIL import Image

try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False


class CalibratedQualityOperators:
    """
    严格对齐 dirty_audit_report.md 实测校准阈值的六大算子 + PII 脱敏。

    乱码检测口径（实测校准结论）：
    - 中文文本天然高熵，熵不能做阈值；
    - 只认"混乱字符"（GBK/Latin-1 解码崩坏产生的 MOJIBAKE 字符 + 私有区字符），
      绝不能把 × ° ± ² 等合法数学符号算作乱码（否则数学语料会被 100% 误杀）；
    - 检测粒度必须是字段级（半损坏文档的损坏集中在个别字段，整文件占比会被稀释）。
    """
    # 实测校准的混乱字符集：解码崩坏典型产物（锟斤拷/烫烫烫...）+ 私有区 PUA
    MOJIBAKE_CHARS = set("锟斤拷烫烫烫硚硟硖胬鹌齉龘氅丏丗亜")

    def __init__(self):
        self.seen_sha256: Dict[str, str] = {}             # file_sha256 -> doc_id
        self.seen_question_sets: Dict[str, Set[str]] = {}  # doc_id -> set(question_fingerprints)
        self.known_clean_base_ids: Set[str] = set()

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_text_sha256(text: str) -> str:
        normalized = re.sub(r"\s+", "", text.strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def check_garbled_and_format(self, file_path: Path, ext: str,
                                 field_max: float = 0.05, file_max: float = 0.02) -> Tuple[bool, str]:
        """算子4：JSON 可解析性 + 字段级混乱字符占比 (>5%) + 整文件占比 (>2%)"""
        if ext not in {".json", ".txt", ".md"}:
            return False, ""
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return True, "GARBLED_ENCODING: UTF-8 字节流解码失败"

        if not raw_text.strip():
            return True, "GARBLED_EMPTY: 文本内容为空"

        if ext == ".json":
            try:
                data = json.loads(raw_text)
            except Exception as e:
                return True, f"GARBLED_JSON_BROKEN: JSON 未闭合或截断 ({str(e)[:50]})"

            for field_path, val in self._extract_str_fields(data):
                clean_val = re.sub(r"\s+", "", val)
                if clean_val:
                    ratio = self._count_garbled_chars(clean_val) / len(clean_val)
                    if ratio > field_max:
                        return True, f"GARBLED_FIELD: 字段 {field_path} 混乱字符占比 {ratio:.2%} > {field_max:.0%}"

        clean_all = re.sub(r"\s+", "", raw_text)
        if clean_all:
            file_ratio = self._count_garbled_chars(clean_all) / len(clean_all)
            if file_ratio > file_max:
                return True, f"GARBLED_FILE: 整文件混乱字符占比 {file_ratio:.2%} > {file_max:.0%}"
        return False, ""

    def _count_garbled_chars(self, text: str) -> int:
        """只统计 MOJIBAKE 字符与私有区字符（实测校准口径，绝不含数学符号）。"""
        return sum(1 for ch in text if ch in self.MOJIBAKE_CHARS or 0xE000 <= ord(ch) <= 0xF8FF)

    def _extract_str_fields(self, obj, prefix: str = "root") -> List[Tuple[str, str]]:
        out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.extend(self._extract_str_fields(v, f"{prefix}.{k}"))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                out.extend(self._extract_str_fields(v, f"{prefix}[{i}]"))
        elif isinstance(obj, str):
            out.append((prefix, obj))
        return out

    def check_near_duplicate_json(self, doc_id: str, file_path: Path) -> Tuple[bool, str]:
        """算子2：题目级指纹重合度 >= max(3, ceil(唯一题干数 * 0.80))"""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return False, ""
        fps = self._extract_question_fingerprints(data)
        if not fps:
            return False, ""
        threshold = max(3, math.ceil(len(fps) * 0.80))
        for exist_id, exist_fps in self.seen_question_sets.items():
            overlap = len(fps & exist_fps)
            if overlap >= threshold:
                return True, f"NEAR_DUPLICATE: 与 {exist_id} 题干指纹重合 {overlap}/{len(fps)} (阈值>={threshold})"
        self.seen_question_sets[doc_id] = fps
        return False, ""

    @staticmethod
    def _extract_question_fingerprints(data) -> Set[str]:
        items = data if isinstance(data, list) else []
        if isinstance(data, dict):
            for k in ("questions", "items", "data", "problems", "records", "list"):
                if isinstance(data.get(k), list):
                    items = data[k]
                    break
            if not items:
                items = [data]
        fps = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            stem = ""
            for k in ("question", "stem", "content", "problem", "title", "prompt", "题干", "题目"):
                if isinstance(item.get(k), str) and len(item[k].strip()) > 4:
                    stem = item[k]
                    break
            if not stem:
                strs = [v for v in item.values() if isinstance(v, str)]
                stem = max(strs, key=len) if strs else ""
            if stem:
                # 指纹必须取"完整题干"：前缀截断会导致模板题（前半句相同、数字在后）
                # 大量撞车误伤（实测 153 份干净文档被误判近重复）
                core = re.sub(r"\s+", "", stem)
                if len(core) >= 6:
                    fps.add(hashlib.md5(core.encode("utf-8")).hexdigest()[:16])
        return fps

    @staticmethod
    def check_blank_pdf(file_path: Path, size_kb: float) -> Tuple[bool, str]:
        """算子5：整份空白 (<30KB) + 页面级像素检测局部空白页（第1页有内容、第2页空白）"""
        if size_kb < 30.0:
            return True, f"BLANK_PDF_WHOLE: 文件大小 {size_kb:.1f}KB < 30.0KB"
        if HAS_PDFIUM:
            try:
                pdf = pdfium.PdfDocument(str(file_path))
                blank_pages = []
                for idx in range(len(pdf)):
                    pil_img = pdf[idx].render(scale=0.5).to_pil().convert("L")
                    arr = np.asarray(pil_img, dtype=np.float32)
                    if float(np.mean(arr > 245.0)) >= 0.992 or float(np.std(arr)) < 5.0:
                        blank_pages.append(idx + 1)
                pdf.close()
                if blank_pages:
                    return True, f"BLANK_PDF_PARTIAL: 页面级像素检测发现空白页 {blank_pages}"
            except Exception as e:
                return True, f"CORRUPT_PDF: {str(e)[:40]}"
        return False, ""

    @staticmethod
    def check_low_res_image(file_path: Path, min_energy: float = 5.0) -> Tuple[bool, str]:
        """
        算子6：图像梯度能量清晰度。
        实测校准（dirty_audit_report.md）：先 resize(400,270) 再算 gx.mean()+gy.mean()（求和标度），
        低清图 2.14~4.47、干净图 5.36~13.30，阈值 5.0 可 100% 分离。
        """
        try:
            with Image.open(file_path) as img:
                gray = img.convert("L").resize((400, 270))
                arr = np.asarray(gray, dtype=np.float32)
            gx = np.abs(np.diff(arr, axis=1))
            gy = np.abs(np.diff(arr, axis=0))
            energy = float(np.mean(gx) + np.mean(gy))
            if energy < min_energy:
                return True, f"LOW_RES_IMAGE: 梯度能量 {energy:.2f} < {min_energy:.2f}"
            return False, ""
        except Exception as e:
            return True, f"CORRUPT_IMAGE: {str(e)[:40]}"

    @staticmethod
    def mask_pii(text: str) -> Tuple[str, int]:
        """PII 敏感信息脱敏（手机号与18位身份证）"""
        text, n1 = re.subn(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2", text)
        text, n2 = re.subn(r"(?<!\d)(\d{6})\d{8}(\d{3}[0-9Xx])(?!\d)", r"\1********\2", text)
        return text, n1 + n2