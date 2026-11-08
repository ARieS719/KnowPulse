# -*- coding: utf-8 -*-
import os
import sqlite3
from typing import List
import httpx
import numpy as np
from dotenv import load_dotenv

from src.config import get_settings, apply_hf_endpoint
from src.quality.operators import CalibratedQualityOperators

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", ".env"))

class CachedBgeM3Embedder:
    """
    带 SQLite SHA256 永久缓存的 bge-m3 嵌入器：
    - mode='local': 本机推理（GPU 可用时 FP16；无 GPU 自动降级 CPU，batch_size 建议调小）
    - mode='siliconflow': 调用硅基流动 https://api.siliconflow.com/v1/embeddings (BAAI/bge-m3)
    """
    def __init__(self, sqlite_conn: sqlite3.Connection, mode: str | None = None,
                 model_name: str | None = None, batch_size: int | None = None):
        s = get_settings()
        self.conn = sqlite_conn
        self.mode = mode or s.effective_embed_mode
        self.model_name = model_name or s.embed.model
        self.batch_size = batch_size or s.embed.batch_size
        self.api_key = s.siliconflow_api_key
        self.base_url = s.siliconflow_base_url
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            apply_hf_endpoint()          # 必须在加载模型前设置 HF_ENDPOINT（国内镜像加速下载）
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._local_model = SentenceTransformer(
                self.model_name,
                device=device,
                model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
            )
        return self._local_model

    def encode(self, texts: List[str], batch_size: int | None = None) -> List[List[float]]:
        batch_size = batch_size or self.batch_size
        cur = self.conn.cursor()
        hashes = [CalibratedQualityOperators.compute_text_sha256(t) for t in texts]
        vectors: List[List[float]] = [[] for _ in texts]
        miss_idx, miss_texts, miss_hashes = [], [], []

        for i, h in enumerate(hashes):
            row = cur.execute(
                "SELECT vector_blob FROM embedding_cache WHERE text_sha256 = ? AND model_name = ?",
                (h, self.model_name)
            ).fetchone()
            if row:
                vectors[i] = np.frombuffer(row[0], dtype=np.float32).tolist()
            else:
                miss_idx.append(i)
                miss_texts.append(texts[i])
                miss_hashes.append(h)

        if miss_texts:
            if self.mode == "siliconflow":
                computed = self._encode_siliconflow(miss_texts, batch_size=batch_size)
            else:
                model = self._get_local_model()
                computed = model.encode(
                    miss_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
                )

            for i, h, vec in zip(miss_idx, miss_hashes, computed):
                arr = np.asarray(vec, dtype=np.float32)
                vectors[i] = arr.tolist()
                cur.execute(
                    "INSERT OR REPLACE INTO embedding_cache (text_sha256, model_name, vector_blob) VALUES (?, ?, ?)",
                    (h, self.model_name, arr.tobytes())
                )
            self.conn.commit()
        return vectors

    def _encode_siliconflow(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError(
                "嵌入模式为 siliconflow 但未配置 SILICONFLOW_API_KEY（configs/.env）。"
                "请填写密钥，或把 configs/settings.yaml 中 embed.mode 改为 local。"
            )
        out = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=60.0) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                r = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model_name, "input": batch, "encoding_format": "float"}
                )
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda x: x["index"])
                out.extend([item["embedding"] for item in data])
        return out