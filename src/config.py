# -*- coding: utf-8 -*-
"""
KnowPulse 全局配置加载器。

优先级：configs/settings.yaml（静态配置） < 环境变量 / configs/.env（密钥与运行模式）。
所有相对路径均解析为"项目根目录"下的绝对路径，因此任意工作目录下运行 CLI 都能正确定位数据。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


class PathsConfig(BaseModel):
    data_root: str = "data/raw"
    manifest_csv: str = "data/meta/manifest.csv"
    parsed_dir: str = "data/parsed"
    quarantine_dir: str = "data/quarantine"
    golden_set: str = "data/golden_set/golden_set_v1.jsonl"
    badcases: str = "data/golden_set/badcases.jsonl"
    chroma_dir: str = "data/storage/chroma_db"
    metadata_db: str = "data/storage/metadata.db"
    ablation_report: str = "data/golden_set/ablation_report.md"
    quality_gate_report: str = "data/quarantine/quality_gate_report.md"


class EmbedConfig(BaseModel):
    mode: str = "local"                     # local | siliconflow
    model: str = "BAAI/bge-m3"
    batch_size: int = 32
    normalize: bool = True
    hf_endpoint: str = "https://hf-mirror.com"


class RerankConfig(BaseModel):
    mode: str = "local"                     # local | siliconflow
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    top_n: int = 8


class SemanticCacheConfig(BaseModel):
    enabled: bool = True
    max_size: int = 256
    threshold: float = 0.96


class RetrievalConfig(BaseModel):
    top_k_vector: int = 50
    top_k_fts: int = 50
    final_top_k: int = 8
    rrf_k: int = 60
    min_rerank_score: float = 0.05
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)


class LLMConfig(BaseModel):
    provider: str = "siliconflow"
    model: str = "Qwen/Qwen2.5-72B-Instruct"
    judge_model: str = "deepseek-ai/DeepSeek-V3"
    temperature: float = 0.1
    price_per_mtok: float = 4.13
    judge_price_per_mtok: float = 2.0


class QualityConfig(BaseModel):
    garbled_field_max: float = 0.05
    garbled_file_max: float = 0.02
    blank_pdf_size_kb: float = 30.0
    low_res_energy: float = 5.0
    near_dup_overlap_ratio: float = 0.80
    ocr_confidence_min: float = 0.75


class LangfuseConfig(BaseModel):
    enabled: bool = False


class Settings(BaseModel):
    """聚合配置。密钥字段来自 configs/.env。"""
    run_mode: str = "MODE_A"                # MODE_A | MODE_B | MODE_CLOUD

    paths: PathsConfig = Field(default_factory=PathsConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)

    # ---- secrets（configs/.env）----
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"   # 国内主站；国际站为 .com（账号不互通）
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    ollama_base_url: str = "http://127.0.0.1:11434"
    hf_token: str = ""

    # ------------------------------------------------------------
    # 路径解析
    # ------------------------------------------------------------
    def resolve(self, rel: str) -> Path:
        """把配置中的相对路径解析为项目根目录下的绝对路径。"""
        p = Path(rel)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def data_root(self) -> Path:
        return self.resolve(self.paths.data_root)

    @property
    def manifest_csv(self) -> Path:
        return self.resolve(self.paths.manifest_csv)

    @property
    def parsed_dir(self) -> Path:
        return self.resolve(self.paths.parsed_dir)

    @property
    def quarantine_dir(self) -> Path:
        return self.resolve(self.paths.quarantine_dir)

    @property
    def golden_set_path(self) -> Path:
        return self.resolve(self.paths.golden_set)

    @property
    def badcases_path(self) -> Path:
        return self.resolve(self.paths.badcases)

    @property
    def chroma_dir(self) -> Path:
        return self.resolve(self.paths.chroma_dir)

    @property
    def metadata_db_path(self) -> Path:
        return self.resolve(self.paths.metadata_db)

    @property
    def ablation_report_path(self) -> Path:
        return self.resolve(self.paths.ablation_report)

    @property
    def quality_gate_report_path(self) -> Path:
        return self.resolve(self.paths.quality_gate_report)

    # ------------------------------------------------------------
    # 运行模式 -> 组件模式推导（v1.2 双运行模式的落地）
    # ------------------------------------------------------------
    @property
    def effective_embed_mode(self) -> str:
        if self.run_mode == "MODE_CLOUD":
            return "siliconflow"
        if self.run_mode == "MODE_B":
            return "local"
        return self.embed.mode

    @property
    def effective_rerank_mode(self) -> str:
        if self.run_mode == "MODE_CLOUD":
            return "siliconflow"
        return self.rerank.mode

    @property
    def use_ollama_llm(self) -> bool:
        return self.run_mode == "MODE_B"

    @property
    def has_siliconflow_key(self) -> bool:
        k = self.siliconflow_api_key.strip()
        return bool(k) and "你的" not in k and k.lower() not in {"sk-xxx", "sk-your"}


def _load_env_file() -> dict:
    """读取 configs/.env。已存在于环境变量的键以环境变量值为准（不丢值）。"""
    env_path = CONFIG_DIR / ".env"
    out: dict = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key:
            out[key] = os.environ.get(key, val)   # 环境变量优先，但绝不丢值
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存全局配置（进程内单例）。"""
    yaml_path = CONFIG_DIR / "settings.yaml"
    yaml_data: dict = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        yaml_data = {k: v for k, v in loaded.items() if isinstance(v, dict)}

    env = _load_env_file()

    settings = Settings(
        run_mode=env.get("RUN_MODE", yaml_data.pop("run_mode", "MODE_A")).strip(),
        paths=PathsConfig(**{**yaml_data.get("paths", {}), "data_root": env.get("DATA_ROOT") or yaml_data.get("paths", {}).get("data_root", "data/raw")}),
        embed=EmbedConfig(**yaml_data.get("embed", {})),
        rerank=RerankConfig(**yaml_data.get("rerank", {})),
        retrieval=RetrievalConfig(**yaml_data.get("retrieval", {})),
        llm=LLMConfig(**yaml_data.get("llm", {})),
        quality=QualityConfig(**yaml_data.get("quality", {})),
        langfuse=LangfuseConfig(**yaml_data.get("langfuse", {})),
        siliconflow_api_key=env.get("SILICONFLOW_API_KEY", ""),
        siliconflow_base_url=env.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1"),
        langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY", ""),
        langfuse_host=env.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        hf_token=env.get("HF_TOKEN", ""),
    )
    # 模型覆盖（.env 中可覆盖 yaml 的模型名）
    if env.get("LLM_MODEL"):
        settings.llm.model = env["LLM_MODEL"]
    if env.get("JUDGE_MODEL"):
        settings.llm.judge_model = env["JUDGE_MODEL"]
    if env.get("LOCAL_EMBED_MODEL"):
        settings.embed.model = env["LOCAL_EMBED_MODEL"]
    if env.get("LOCAL_RERANK_MODEL"):
        settings.rerank.model = env["LOCAL_RERANK_MODEL"]
    return settings


def apply_hf_endpoint() -> None:
    """把 settings 里的镜像端点写入环境变量（必须在导入 huggingface/torch 前调用）。"""
    s = get_settings()
    ep = (s.embed.hf_endpoint or "").strip()
    if ep and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = ep
    if s.hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = s.hf_token
