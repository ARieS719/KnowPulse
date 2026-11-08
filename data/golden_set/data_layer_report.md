# KnowPulse 数据层评测报告

## 文档版本状态机
| 状态 | 数量 |
|---|---:|
| DEPRECATED | 57 |
| INDEXED | 4542 |
| QUARANTINED | 580 |

## 质量门检测标签分布
| 检测标签 | 数量 |
|---|---:|
| blank | 70 |
| clean | 4502 |
| duplicate | 206 |
| duplicate_near | 100 |
| garbled | 160 |
| low_res | 44 |
| version_conflict | 97 |

## 隔离原因 TOP
| 原因 | 数量 |
|---|---:|
| BLANK_PDF_WHOLE: 文件大小 25.5KB < 30.0KB | 50 |
| GARBLED_JSON_BROKEN: JSON 未闭合或截断 (Unterminated string starti | 30 |
| GARBLED_JSON_BROKEN: JSON 未闭合或截断 (Expecting value: line 1 co | 28 |
| BLANK_PDF_PARTIAL: 页面级像素检测发现空白页 [2] | 20 |
| GARBLED_FIELD: 字段 root.questions[0].答案 混乱字符占比 100.00% > 5% | 10 |
| GARBLED_FIELD: 字段 root.questions[0].题干 混乱字符占比 10.87% > 5% | 4 |
| NEAR_DUPLICATE: 与 QB-SYN-000876 题干指纹重合 5/5 (阈值>=4) | 2 |
| NEAR_DUPLICATE: 与 QB-SYN-000615 题干指纹重合 5/5 (阈值>=4) | 2 |
| NEAR_DUPLICATE: 与 QB-SYN-000312 题干指纹重合 5/5 (阈值>=4) | 2 |
| LOW_RES_IMAGE: 梯度能量 4.23 < 5.00 | 2 |

## chunk 粒度
- chunk 分词长度分布（采样 19211 块）：中位数 217，P95 555
