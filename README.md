# KnowPulse (EduGuard-RAG)

[![CI](https://github.com/ARieS719/KnowPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/ARieS719/KnowPulse/actions)

一个围绕检索增强生成（RAG）数据质量问题的实验项目。实验以一份混合格式的中文教育语料（5,179 份文档，其中 677 份为人为注入的脏数据）为对象，实现并验证「入库质量门 → 双索引混合检索 → 三层评测」的完整流程。项目关心的两个具体问题是：非结构化文档进入索引之前如何进行质量拦截，以及检索与生成的效果如何被量化评测。

## 实验结果摘要

| 项目 | 结果 | 证据 |
|---|---|---|
| 质量门：677 份脏数据拦截率 | 677/677，干净数据零误伤 | [quality_gate_report.md](data/quarantine/quality_gate_report.md) |
| 检索 Recall@5（hybrid_rerank） | 81.08% | [ablation_report.md](data/golden_set/ablation_report.md) |
| 检索 Recall@10 | 86.49% | 同上 |
| 相对向量基线提升 | +25.67 个百分点 | 同上 |
| 生成忠实度（裁判模型判分） | 68/68 判为忠实 | [generation_report.md](data/golden_set/generation_report.md) |
| 无证据拒答正确率 | 5/5 | 同上 |
| 引用硬校验 Precision | 0.8924 | 同上 |
| 单元测试 | 8 项通过（CI 门禁） | `.github/workflows/ci.yml` |

索引规模：4,542 份文档、19,211 个切片（ChromaDB 向量 + SQLite FTS5 双索引）。全量评测的 LLM API 累计成本约 0.61 元。

## 1. 背景与目标

### 1.1 背景

RAG 知识库的检索效果很大程度上取决于进入索引的文档质量。实际收集到的文档往往混杂重复文件、低清扫描件、乱码块以及新旧版本冲突，这些问题若不在入库前拦截，会在检索与生成阶段造成难以定位的错误。实验希望回答两个问题：

1. 能否用一套可配置的检测规则，在文档入库前拦截上述问题文档；
2. 能否建立一套可复现的量化评测流程，衡量检索与生成效果，并让每次修改的影响可度量。

### 1.2 评测指标

| 指标 | 设定目标 | 实测 |
|---|---|---|
| 脏数据入库拦截率 | 100% | 100%（677/677，误伤 0） |
| 文档解析成功率 | ≥ 95% | 未做全量抽检（解析失败自动转入质量门） |
| 检索 Recall@5 | ≥ 82% | 81.08% |
| Recall@5 分层 | 简单类 ≥ 92%；表格/扫描类 ≥ 72% | 88.57% / 74.36% |
| Recall@10 | ≥ 90% | 86.49% |
| 相对朴素基线提升 | ≥ 18–25 个百分点 | +25.67pp |
| 生成忠实度 | ≥ 0.85 | 68/68 |
| 引用硬校验 Precision | ≥ 0.90 | 0.8924 |
| 单次问答成本 | < 0.05 元 | 实测 0.005 元量级 |

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│ 接入层   FastAPI（/ingest /query /feedback /quality/summary）│
├─────────────────────────────────────────────────────────────┤
│ 采集层   本地目录导入 → 元数据抽取（标题/学科/版本/access_roles）│
├─────────────────────────────────────────────────────────────┤
│ 解析层   MinerU（预留）/ pypdfium2 / PaddleOCR（预留）        │
│          python-docx / openpyxl / JSON 题库 → 父子块          │
│          跨页表格拼接修复（table_stitcher）                    │
├─────────────────────────────────────────────────────────────┤
│ 数据质量层                                                    │
│  哈希去重 / 题目级近重复检测 / 版本一致性状态机 / 字段级乱码    │
│  检测 / 空白页与低分辨率检测 / PII 脱敏 / 质量门 + 核查报告     │
├─────────────────────────────────────────────────────────────┤
│ 切片层   标题树切分 + 父子块 + 表格/图片摘要索引               │
│          断头块语义完整度修复（表头断离/题目选项分离）         │
├─────────────────────────────────────────────────────────────┤
│ 索引层   bge-m3 → 1024 维向量 → ChromaDB                      │
│          BM25 全文 → SQLite FTS5（jieba 预分词）              │
│          SHA256 嵌入缓存（文本未变不重算）                     │
│          双索引由同一 chunk_id 关联，原子双写 + 故障回滚       │
├─────────────────────────────────────────────────────────────┤
│ 检索层   向量 Top100 + FTS5 Top100 → RRF 融合 → bge-reranker  │
│          重排 Top8 → ACL 前置过滤 → 语义缓存（热点 <20ms）     │
├─────────────────────────────────────────────────────────────┤
│ 生成层   硅基流动 API（主）/ Ollama 本地（兜底）               │
│          强制引用 prompt + 无证据拒答 + 引用双层校验           │
├─────────────────────────────────────────────────────────────┤
│ 评测层                                                        │
│  数据层/检索层/生成层三层评测 + 四档消融实验                   │
│  badcase 归因（解析/切片/检索/生成）→ 修复 → 回归对比          │
└─────────────────────────────────────────────────────────────┘
```

## 3. 目录结构

```
knowpulse-rag/
├── .github/            CI 质量门禁（pytest 全量测试）
├── configs/            settings.yaml（全局配置）+ .env.example（密钥模板）
├── data/
│   ├── raw/            5,179 份语料（含 677 份注入脏数据；不入库，生成脚本另行托管）
│   ├── parsed/         解析缓存（运行时生成）
│   ├── quarantine/     质量门拦截清单与核查报告（报告入库）
│   ├── golden_set/     评测集（79 条，5 难度层）与消融/评测报告（入库）
│   ├── meta/           manifest.csv + 脏数据审计报告 + 来源声明（入库）
│   └── storage/        chroma_db/ + metadata.db（运行时生成，不入库）
├── src/
│   ├── config.py       全局配置加载器（settings.yaml + .env）
│   ├── llm/            siliconflow_client.py
│   ├── ingestion/      parser_router.py + table_stitcher.py
│   ├── quality/        state_machine.py + operators.py + quality_gate.py
│   ├── indexing/       chunker.py + embedder.py + fts5_store.py + dual_writer.py
│   ├── retrieval/      hybrid_search.py + reranker.py + semantic_cache.py
│   ├── generation/     rag_chain.py + citation_check.py
│   ├── evaluation/     metrics.py + build_golden.py + badcase_loop.py
│   └── api/            main.py
└── tests/              test_fault_injection.py + test_quality_and_fts.py
```

## 4. 快速开始

```powershell
# 环境准备（一次性）
cd D:\knowpulse-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # GPU 环境另见 §8

# 配置密钥
# 复制 configs/.env.example 为 configs/.env，填入 SILICONFLOW_API_KEY
# （硅基流动 https://cloud.siliconflow.cn 注册后创建 API 密钥）
# 注意：中国区账号密钥只能使用 https://api.siliconflow.cn/v1 端点（.com 国际站不互通）

# Step 1 质量门全量审计（5,179 份语料 → 拦截 677 份脏/冲突文档）
python -m src.quality.quality_gate

# Step 2-3 双索引构建（先小样本验证，再全量）
python -m src.indexing.dual_writer --limit 50
python -m src.indexing.dual_writer

# Step 4 四档消融实验（Recall@5 / MRR / nDCG / 延迟）
python -m src.evaluation.metrics --action run_ablation

# Step 5 三层评测
python -m src.evaluation.metrics --action data_layer
python -m src.evaluation.metrics --action generation

# Step 6 启动 API 服务
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000/docs 直接试 /query

# 测试
python -m pytest tests
```

## 5. 评测方法与结果

### 5.1 评测集

评测集 [golden_set_v1.jsonl](data/golden_set/golden_set_v1.jsonl) 共 79 条，按难度分为五层：简单检索 35、跨文档聚合 14、表格数值 9、图片理解 16、应拒答 5。评测题与语料模板解耦，避免字面重合造成的评测泄漏。每条包含问题、标准答案、证据文档与难度标签。

命中判定采用三口径（任一成立即命中）：证据文档本身被检索到；检索到与证据内容相同的模板副本（内容指纹级）；检索内容能够支撑标准答案（答案包含性）。语料中存在大量同模板副本，仅按文档编号判定会系统性低估真实检索质量。

### 5.2 检索层：四档消融实验

对 74 条可检索评测题（排除 5 条应拒答），分别在四种检索配置下测量：

| 检索配置 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| 仅向量 | 55.41% | 64.86% | 0.5087 | 0.5416 | 62.8 ms |
| 仅 BM25 | 70.27% | 78.38% | 0.6706 | 0.6968 | 18.0 ms |
| 混合 + RRF | 68.92% | 81.08% | 0.5805 | 0.6348 | 78.3 ms |
| 混合 + RRF + 重排 | 81.08% | 86.49% | 0.7392 | 0.7693 | ~1.4–2.1 s |

重排档的延迟主要来自云端重排 API 往返（详见 §8）。完整报告见 [ablation_report.md](data/golden_set/ablation_report.md)。

### 5.3 生成层：忠实度与引用校验

- 忠实度：以裁判模型（DeepSeek-V3）判断"标准答案是否蕴含于检索证据"，74 条可回答题目中 68 条参与判定、全部判为忠实；
- 无证据拒答：5 条应拒答题全部正确拒答；
- 引用硬校验：在线正则校验回答中的引用编号是否真实存在于本次召回，平均 Precision 0.8924。

完整报告见 [generation_report.md](data/golden_set/generation_report.md)。

### 5.4 数据层与质量门

质量门对 5,179 份语料全量审计：4,502 份干净文档全部放行；677 份脏/冲突文档全部拦截（重复 200、近重复 100、版本冲突 97、乱码 160、空白 70、低清 50，其中 6 份为低清+重复的复合注入，按优先级归入重复）。报告见 [quality_gate_report.md](data/quarantine/quality_gate_report.md)，检测器阈值全部经过语料实测校准，校准过程记录在 [dirty_audit_report.md](data/meta/dirty_audit_report.md)。

### 5.5 迭代记录

实验过程中三次较大的修改及其影响：

1. 讲义解析器将 Heading 2 样式的正文段误判为标题，导致讲义正文丢失；修复后 FTS 索引块 14,710 → 19,211，简单类 Recall 提升约 20 个百分点；
2. RRF 合并池截断导致只在单路出现的候选被挤出重排池；保留两路全部唯一候选后，Recall@5 提升 13.5 个百分点；
3. 评测命中口径从文档级升级为三口径判定，修正模板副本语料下的系统性低估。

## 6. 语料说明

- 构成（5,179 份）：题库 2,002（562 组真实高考真题 GAOKAO-Bench，Apache-2.0 + 1,440 组合成题）/ 讲义 1,000 / 表格 500 / 模拟扫描件 700 / 题目图片 300 / 注入脏数据 677。
- 脏数据按故障类型反向设计：每一类对应一种真实故障模式与一个可指定检测器（重复、近重复、版本冲突、乱码、空白页、低清）。
- 版权：所有数据仅用于学习与技术演示，来源与许可详见 [SOURCES.md](data/meta/SOURCES.md)。
- 语料生成脚本与评测集构建脚本位于仓库之外（体积与版权考虑），需要复现语料时可另行提供。

## 7. 运行模式（settings.yaml `run_mode`）

| 模式 | 嵌入/重排 | LLM | 适用 |
|---|---|---|---|
| MODE_A（默认） | 本地嵌入 + 云端重排 | 硅基流动 API | 本机算力 + API 组合 |
| MODE_B | 本地 | Ollama 本地 | 离线演示 |
| MODE_CLOUD | 硅基流动 API | 硅基流动 API | 未装本地模型时最快跑通 |

## 8. 运行环境与硬件适配

- 实验机器：RTX 3050 Laptop 4GB / 16GB RAM / Windows 11。
- bge-m3 本地嵌入实测约 31 chunks/s、峰值显存 1.1GB；Ollama 3B Q4（约 2.2GB）不应与嵌入/重排模型同时常驻。
- 重排默认走云端（`rerank.mode: siliconflow`）：本机 4GB 显存与桌面 GUI 共享，本地 CrossEncoder（FP32 约 2.2GB）实测触发 WDDM 显存分页抖动（CPU 空闲、GPU 100%、速度极慢）；云端使用同款 bge-reranker-v2-m3，单次约 0.3s。关闭大应用后可切回 `local` 验证本地路径。
- CUDA 版 torch（当前 venv 若是 CPU 版，需重装）：
  ```powershell
  pip uninstall -y torch
  pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
  ```
- 无 GPU 也可运行：嵌入/重排自动降级 CPU（settings.yaml 中 embed.batch_size 建议调至 8–16），或切换 MODE_CLOUD。
- 模型下载镜像：settings.yaml `embed.hf_endpoint` 默认 `https://hf-mirror.com`。

## 9. 局限性与后续方向

- 语料与评测集的规模较小（79 条评测题），后续可用 `build_golden.py` 的候选生成 + 人工复核流程扩充；
- 真实扫描件 OCR（paddleocr/mineru）尚未安装，当前以标题+元数据兜底索引；
- 引用硬校验 Precision 0.8924 略低于设定目标，可尝试在 prompt 中加入引用格式的少样本示例；
- 未做模型预训练/微调、多租户权限（仅文档级 ACL 预留）、移动端；文档规模按万级设计；
- 检索延迟在云端重排配置下约 1.4–2.1s（API 往返）；本地 GPU 重排可达 150–280ms，受 4GB 共享显存限制。

## 参考资料

- 真实题库数据：[GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench)（OpenLMLab，Apache-2.0）
- 试卷标题元数据：[Exam-Question-Bank-Dataset-zh_mnbvc](https://github.com/UnstoppableCurry/Exam-Question-Bank-Dataset-zh_mnbvc)（MIT）
- 嵌入与重排模型：[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)、[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- 向量库：[ChromaDB](https://docs.trychroma.com/)；全文检索：[SQLite FTS5](https://www.sqlite.org/fts5.html)（jieba 预分词）
- 文档解析：[MinerU](https://github.com/opendatalab/MinerU)（预留）、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（预留）
- LLM 服务：[硅基流动 API](https://siliconflow.cn)（Qwen2.5-72B-Instruct / DeepSeek-V3）
- 评测方法参考：RAGAS 的忠实度与引用评测思路
