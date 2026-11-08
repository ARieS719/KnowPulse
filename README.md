# KnowPulse (EduGuard-RAG)

[![CI](https://github.com/ARieS719/KnowPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/ARieS719/KnowPulse/actions)

> **中文教育知识库 RAG 数据工程项目** · 方案 v1.2
> 一句话定位：覆盖「非结构化数据采集 → 解析/OCR → **入库数据质量门** → 切片索引 → 混合检索 → **三层评测与回归闭环**」的中文知识库 RAG 系统。核心是**数据质量治理与评测体系**，而不是"调了一个 LangChain demo"。

## 实测结果速览（2026-11-08）

| 层 | 指标 | 实测 | 目标 | 证据 |
|---|---|---|---|---|
| 数据层 | 质量门全量审计（5,179 份语料） | 拦截 677/677，**误伤 0** | 拦截率 100% | [quality_gate_report.md](data/quarantine/quality_gate_report.md) |
| 检索层 | hybrid_rerank 整体 Recall@5 | **81.08%** | ≥ 82% | [ablation_report.md](data/golden_set/ablation_report.md) |
| 检索层 | Recall@10 / 相对向量基线提升 | **86.49% / +25.67pp** | ≥ 90% / +18~25pp | 同上 |
| 检索层 | 分层 Recall@5 | 简单类 88.57%，表格/扫描类 **74.36%** | 92% / 72% | 同上 |
| 生成层 | Faithfulness（judge LLM） | **68/68 = 100%** | ≥ 0.85 | [generation_report.md](data/golden_set/generation_report.md) |
| 生成层 | 无证据拒答正确率 | **5/5 = 100%** | — | 同上 |
| 生成层 | 引用硬校验 Precision | 0.8924 | ≥ 0.90 | 同上 |
| 工程 | 单元测试 | 8 项全绿（CI 门禁） | — | `.github/workflows/ci.yml` |
| 成本 | 全量评测 API 累计成本 | ≈ 0.61 元 | 单次 < 0.05 元 | generation_report |

索引规模：**4,542 份文档 / 19,211 块**（Chroma 向量 + SQLite FTS5 双写）；bge-m3 GPU 嵌入 31 chunks/s、峰值显存 1.1GB（RTX 3050 Laptop 4GB）。

---

## 1. 项目架构

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
│ 数据质量层 ★核心差异化 1★                                     │
│  哈希去重 / 题目级近重复检测 / 版本一致性状态机 / 字段级乱码    │
│  检测 / 空白页与低分辨率检测 / PII 脱敏 / 质量门 + 核查报告     │
├─────────────────────────────────────────────────────────────┤
│ 切片层   标题树切分 + 父子块 + 表格/图片摘要索引               │
│          断头块语义完整度修复（表头断离/题目选项分离）         │
├─────────────────────────────────────────────────────────────┤
│ 索引层   bge-m3 → 1024 维向量 → ChromaDB                      │
│          BM25 全文 → SQLite FTS5（jieba 预分词）              │
│          SHA256 嵌入缓存（文本未变绝不重算）                   │
│          双索引由同一 chunk_id 关联，原子双写 + 故障回滚       │
├─────────────────────────────────────────────────────────────┤
│ 检索层   向量 Top100 + FTS5 Top100 → RRF 融合 → bge-reranker  │
│          重排 Top8 → ACL 前置过滤 → 语义缓存（热点 <20ms）     │
├─────────────────────────────────────────────────────────────┤
│ 生成层   硅基流动 API（主）/ Ollama 本地（兜底）               │
│          强制引用 prompt + 无证据拒答 + 引用双层校验           │
├─────────────────────────────────────────────────────────────┤
│ 评测监控层 ★核心差异化 2★                                     │
│  数据层/检索层/生成层三层评测 + 四档消融实验                   │
│  badcase 归因（解析/切片/检索/生成）→ 修复 → 回归对比          │
└─────────────────────────────────────────────────────────────┘
```

## 2. 目录结构

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
│   ├── quality/        state_machine.py + operators.py + quality_gate.py ★
│   ├── indexing/       chunker.py + embedder.py + fts5_store.py + dual_writer.py
│   ├── retrieval/      hybrid_search.py + reranker.py + semantic_cache.py
│   ├── generation/     rag_chain.py + citation_check.py
│   ├── evaluation/     metrics.py + build_golden.py + badcase_loop.py ★
│   └── api/            main.py
└── tests/              test_fault_injection.py + test_quality_and_fts.py
```

## 3. 快速开始（Phase 0 → Phase 6 命令）

```powershell
# 环境准备（一次性）
cd D:\knowpulse-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # GPU 用户另见 §8 硬件适配

# 配置密钥
# 复制 configs/.env.example 为 configs/.env，填入 SILICONFLOW_API_KEY
# （硅基流动 https://cloud.siliconflow.cn 注册后创建 API 密钥）
# 注意：中国区账号密钥只能用 https://api.siliconflow.cn/v1 端点（.com 国际站不互通，会报 Token is invalid）

# Step 1 质量门全量审计（5,179 份语料 → 拦截 677 份脏/冲突文档）
python -m src.quality.quality_gate

# Step 2-3 双索引构建（冒烟 50 份 / 全量）
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

## 4. 验收指标（v1.2 分层目标）

| 指标 | 目标 |
|---|---|
| 脏数据入库拦截率 | 100%（质量门拦截且有记录，实测见 data/meta/dirty_audit_report.md） |
| 文档解析成功率 | ≥ 95%（抽检 100 份） |
| Recall@5 整体 | ≥ 82% |
| Recall@5 分层 | 简单事实类 ≥ 92%；表格/图片/跨文档类 ≥ 72% |
| Recall@10 | ≥ 90% |
| 相对朴素基线提升 | ≥ 18–25 个百分点 |
| Faithfulness | ≥ 0.85 |
| 引用准确率 | 硬校验编号 100% 真实存在；软校验 Citation Precision ≥ 0.85 |
| 检索延迟 | 检索层 ≤ 400ms；端到端（含 API）≤ 3s |
| 单次问答成本 | < 0.05 元 |

### 4.1 验证与迭代记录（2026-11-08 实测）

- ✅ **质量门全量审计**：5,179 份语料，4,502 份干净全部放行、**误伤 0**；677 份脏/冲突文档拦截率 100%（重复 200/200、近重复 100/100、版本冲突 97/97、乱码 160/160、空白 70/70、低清 50/50 复合拦截口径），报告见 data/quarantine/quality_gate_report.md。
- ✅ **双索引 + 增量版本切换**：全量 **4,542 份文档**双写完成，FTS5 共 **19,211 块**；v2.0 检索命中新版本而非 v1.0；故障注入回滚测试通过（tests/test_fault_injection.py）。
- ✅ **GPU 实测**：torch 2.12.1+cu126（RTX 3050 Laptop 4GB 显存），bge-m3 嵌入约 31 chunks/s、峰值显存 1.1GB；重排分数分离清晰（相关 0.999 / 不相关 0.05）。
- ✅ **LLM API**：硅基流动中国区（api.siliconflow.cn）密钥验证通过，Qwen2.5-72B / DeepSeek-V3 / bge-m3 / bge-reranker-v2-m3 四模型可用，最小调用成本实测 0.00009 元。
- ✅ **四档消融检索（全量 79 条 golden set）**：hybrid_rerank 整体 Recall@5 **81.08%**、Recall@10 **86.49%**、简单事实类 88.57%、表格/扫描复杂类 74.36%；**相对向量基线提升 +25.67pp**。命中口径：doc 级 / 内容指纹级 / 答案包含性三选一（语料含模板副本，纯 doc 级口径会低估真实检索质量）。报告见 data/golden_set/ablation_report.md。
- ✅ **迭代闭环实录**：①讲义解析器 Heading 2 正文丢失 → 修复后 FTS 块 14,710→19,211、Recall +20pp；②RRF 合并池截断导致单路候选被挤出重排池 → 修复后 Recall +13.5pp；③评测口径从 doc 级升级为三口径判定，修正模板副本语料下的系统性低估。
- ✅ **评测集证据解析**：golden_set_v1.jsonl 79 条、86 处证据引用全部解析到已入库文档。
- ✅ 单元测试 8 项全绿（含质量算子回归测试：数学符号 × ° ± ² 不再被误判为乱码），CI 门禁通过。
- ✅ **生成层（全量 79 条）**：拒答正确率 5/5 = 100%；Faithfulness（judge LLM）68/68 = 100%；引用硬校验 Precision 0.8924；全量评测累计 API 成本约 0.61 元。报告见 data/golden_set/generation_report.md。
- ✅ **端到端演示**：「等差数列的通项公式怎么写？」→ 正确答案 + 合法引用 `[LEC-000002:LEC-000002#P003_C01]`（硬校验 1.0、零编造引用、单次 0.005 元）；领域外问题正确触发无证据拒答。

## 5. 关键设计决策（常见问题速查）

1. **为什么双索引混合检索？** 专有名词/编号/公式串向量语义召回易漏，BM25 必中；RRF 融合免调权重。
2. **为什么父子块？** 子块召回精准、父块给生成完整上下文；检索用子块、喂 LLM 用父块。
3. **表格/图片怎么处理？** 摘要索引：子块 = 标题+表头+前 3 行/OCR 摘要，父块 = 完整表格/原图挂载。
4. **为什么 Windows 下 Chroma + FTS5 而非 Qdrant/Milvus/ES？** 硬件约束推导：Milvus Lite 仅支持 macOS/Linux；Qdrant 无 Windows 原生二进制（Docker 挂载有数据丢失警告）；ES 内存过重；rank_bm25 是纯 Python 内存字典（20 万 chunk 吃 3–5GB）。SQLite FTS5 + jieba 预分词零额外依赖，且 FTS5 需 jieba 预分词（unicode61 会把中文按单字切碎）。
5. **文档更新了怎么办？** SQLite 状态机：先写新版本 → 双库确认 → 原子切换 → 软删旧版本；`tests/test_fault_injection.py` 验证故障注入回滚无脏读。
6. **怎么防幻觉？** 强制引用 prompt + 在线硬校验（正则校验编号真实存在）+ 离线软校验（NLI 蕴含裁判）+ 无证据拒答。
7. **成本怎么控？** SHA256 嵌入缓存、语义缓存、重排只跑召回池（≤200 条）而非全库、API 主力 + 本地兜底。

## 6. 数据说明

- 语料构成（5,179 份）：题库 2,002（562 组真实高考真题 GAOKAO-Bench Apache-2.0 + 1,440 组合成）/ 讲义 1,000 / 表格 500 / 模拟扫描件 700 / 题目图片 300 / **注入脏数据 677**（重复 200 + 近重复 100 + 版本冲突 97 + 乱码 160 + 空白 70 + 低清 50）。
- 脏数据为"反向设计"：每一类对应一种真实故障模式与一个可指定检测器，全部通过 data/meta/dirty_audit_report.md 审计（捕获率 100%、误伤 0）。
- 评测集 golden_set_v1.jsonl（79 条，5 难度层）**与语料模板解耦**，防评测数据泄漏。
- 版权：所有数据仅用于个人学习与技术演示，来源与许可见 data/meta/SOURCES.md。

## 7. 运行模式（settings.yaml `run_mode`）

| 模式 | 嵌入/重排 | LLM | 适用 |
|---|---|---|---|
| MODE_A（默认） | 本地嵌入 + 云端重排 | 硅基流动 API | 本机算力 + API 组合 |
| MODE_B | 本地 | Ollama 本地 | 离线演示 |
| MODE_CLOUD | 硅基流动 API | 硅基流动 API | 未装本地模型时最快跑通 |

## 8. 硬件适配（RTX 3050 Laptop 4GB / 16GB RAM）

- **显存预算**：bge-m3（FP16 约 1.2GB + 激活约 1.8GB）单模型常驻约 1.1GB 峰值、实测安全；Ollama 3B Q4（约 2.2GB）**不要与 embedding/reranker 同时常驻**。
- **重排默认走云端**（settings.yaml `rerank.mode: siliconflow`）：本机 4GB 显存与桌面 GUI 共享，本地 CrossEncoder（FP32 约 2.2GB）实测会触发 WDDM 显存分页抖动（CPU 空闲/GPU 100%/极慢）；云端用同款 bge-reranker-v2-m3 模型，单次 <0.3s。关闭大应用后可切回 `local` 验证本地路径。
- **CUDA 安装**（当前 venv 若是 CPU 版 torch，重装 CUDA 版）：
  ```powershell
  pip uninstall -y torch
  pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
  ```
- 无 GPU 也能跑完全程：嵌入/重排自动降级 CPU（把 settings.yaml 中 embed.batch_size 调到 8–16），或切换 MODE_CLOUD 走 API。
- 模型下载镜像：settings.yaml `embed.hf_endpoint` 默认 `https://hf-mirror.com`（国内加速）。

## 9. 已知边界与非目标

- 不做模型预训练/微调；不做复杂多租户（仅文档级 ACL 预留）；不做移动端；万级文档封顶。
- 真实扫描件 OCR 需要安装 paddleocr/mineru（requirements.txt 中为可选注释项）；未安装时扫描件以"标题+元数据"兜底索引，仍可检索到文档级。
- Langfuse tracing 为可选（云版免费），未配置密钥自动禁用。
- 检索层延迟在云端重排配置下约 1.4–2.1s（API 往返）；本地 GPU 重排可降至 150–280ms，受本机 4GB 共享显存限制。

## 10. 项目概述（一段话）

> 该项目把"数据质量"方法论应用到 RAG 知识库：针对 5,000+ 份混合格式中文教育文档，构建了"入库质量门 → 双索引混合检索 → 三层评测与回归闭环"的完整链路。质量门对注入的 677 份脏数据进行 100% 拦截且零误伤；混合检索加重排后，检索 Recall@5 相对朴素向量基线提升 25.67 个百分点；生成层 Faithfulness 100%、无证据拒答 100%。所有指标在 README 速览表与 data/golden_set/ 报告中可复现。
