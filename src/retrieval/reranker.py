# -*- coding: utf-8 -*-
"""
重排器：bge-reranker-v2-m3，支持双模式切换。

- local       : 本地 CrossEncoder（GPU/CPU 自动选择；显存紧张时配合 batch_size 控制峰值）
- siliconflow : 硅基流动 /v1/rerank 云端 API（MODE_CLOUD 或无本地算力时兜底）

召回 Top50 -> 重排 Top8，是性价比最高的精度提升手段（README design decisions 之四）。
"""
from __future__ import annotations

from typing import List, Optional

import httpx

from src.config import get_settings


class Reranker:
    def __init__(self, mode: Optional[str] = None, model: Optional[str] = None, batch_size: Optional[int] = None):
        s = get_settings()
        self.mode = mode or s.effective_rerank_mode
        self.model = model or s.rerank.model
        self.batch_size = batch_size or s.rerank.batch_size
        self.top_n = s.rerank.top_n
        self.base_url = s.siliconflow_base_url
        self.api_key = s.siliconflow_api_key
        self._local_model = None

    def _get_local(self):
        if self._local_model is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._local_model = CrossEncoder(self.model, device=device)
        return self._local_model

    def rerank(self, query: str, passages: List[str], top_n: Optional[int] = None) -> List[dict]:
        """
        返回按相关性降序的 [{index, relevance_score}, ...]，index 为 passages 中的原始下标。
        """
        top_n = top_n or self.top_n
        if not passages:
            return []
        if len(passages) == 1:
            return [{"index": 0, "relevance_score": 1.0}]

        if self.mode == "siliconflow":
            return self._remote(query, passages, top_n)

        model = self._get_local()
        pairs = [[query, p] for p in passages]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            ({"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)),
            key=lambda x: x["relevance_score"],
            reverse=True,
        )[:top_n]
        return ranked

    def _remote(self, query: str, passages: List[str], top_n: int) -> List[dict]:
        if not self.api_key:
            raise RuntimeError(
                "重排模式为 siliconflow 但未配置 SILICONFLOW_API_KEY。"
                "请在 configs/.env 填入密钥，或把 configs/settings.yaml 中 rerank.mode 改为 local。"
            )
        url = f"{self.base_url}/rerank"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "query": query,
            "documents": passages,
            "top_n": min(top_n, len(passages)),
            "return_documents": False,
        }
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["results"]
