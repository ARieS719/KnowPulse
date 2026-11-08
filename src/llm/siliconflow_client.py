import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv

from src.config import get_settings

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", ".env"))

class SiliconFlowService:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.siliconflow_api_key
        self.base_url = self.settings.siliconflow_base_url
        self.llm_model = self.settings.llm.model
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _require_client(self):
        if self.client is None:
            raise RuntimeError(
                "未配置 SILICONFLOW_API_KEY（configs/.env）。"
                "请到硅基流动平台（https://siliconflow.cn）创建 API Key 后填入该文件再运行。"
            )

    def chat_with_cost(self, system_prompt: str, user_prompt: str, temperature: float = 0.1,
                       model: str | None = None) -> dict:
        """调用硅基流动 LLM 并精确统计耗时与人民币成本。"""
        self._require_client()
        use_model = model or self.llm_model
        # 裁判模型与生成模型单价不同（settings.llm.price_per_mtok / judge_price_per_mtok）
        price = self.settings.llm.judge_price_per_mtok if use_model == self.settings.llm.judge_model else self.settings.llm.price_per_mtok
        resp = self.client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        usage = resp.usage
        cost_rmb = (usage.prompt_tokens + usage.completion_tokens) / 1_000_000 * price
        return {
            "answer": resp.choices[0].message.content,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_rmb": round(cost_rmb, 6)
        }

    def remote_rerank(self, query: str, documents: list[str], top_n: int = 8) -> list[dict]:
        """当本地显存不足时的云端兜底：调用硅基流动 /v1/rerank 接口"""
        self._require_client()
        url = f"{self.base_url}/rerank"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.settings.rerank.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False
        }
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["results"]  # 返回 [{"index": int, "relevance_score": float}, ...]