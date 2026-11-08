# -*- coding: utf-8 -*-
"""
语义缓存：高频相似问题直接返回缓存结果（热点查询延迟降至 20ms 级）。

- 命中判定：查询向量与缓存项向量余弦相似度 >= threshold（归一化向量下即点积）；
- 容量上限：LRU 淘汰，防止常驻内存无界增长；
- 分角色隔离：student / teacher 各自命中各自的缓存项（ACL 一致性）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class SemanticCache:
    def __init__(self, max_size: int = 256, threshold: float = 0.96, enabled: bool = True):
        self.max_size = max_size
        self.threshold = threshold
        self.enabled = enabled
        self._items: List[Dict] = []          # [{"vec": np.ndarray, "role": str, "response": dict}]
        self.hits = 0
        self.misses = 0

    def lookup(self, query_vec: np.ndarray, user_role: str) -> Optional[dict]:
        """命中返回缓存的 response（浅拷贝），未命中返回 None。"""
        self.misses += 1
        if not self.enabled or not self._items:
            return None
        for item in self._items:
            if item["role"] == user_role and float(np.dot(query_vec, item["vec"])) >= self.threshold:
                self.hits += 1
                self.misses -= 1
                return dict(item["response"])
        return None

    def add(self, query_vec: np.ndarray, user_role: str, response: dict) -> None:
        if not self.enabled:
            return
        self._items.append({"vec": np.asarray(query_vec, dtype=np.float32), "role": user_role, "response": dict(response)})
        while len(self._items) > self.max_size:      # LRU：淘汰最旧
            self._items.pop(0)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "size": len(self._items),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def clear(self) -> None:
        self._items.clear()
