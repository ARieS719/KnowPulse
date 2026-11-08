# -*- coding: utf-8 -*-
from typing import Dict, List

from src.config import get_settings
from src.generation.citation_check import hard_check, soft_check_nli
from src.llm.siliconflow_client import SiliconFlowService

SYSTEM_PROMPT = """你是企业级教育知识库问答助手 KnowPulse。请严格基于提供的【参考资料】回答用户问题：
1. 你的每一句核心事实、公式、定理或答案结论末尾，必须严格附带形如 [doc_id:chunk_id] 的引用标签（例如 [QB-REAL-000001:QB-REAL-000001#C001]）。
2. 严禁编造【参考资料】中不存在的编号或知识。
3. 如果【参考资料】中的信息不足以回答该问题，请直接输出："抱歉，当前知识库中没有找到足以回答该问题的可靠证据（无证据拒答）。\""""


class KnowPulseRAGChain:
    def __init__(self, sf_service: SiliconFlowService):
        self.sf = sf_service

    def generate_and_verify(self, question: str, retrieved_chunks: List[Dict],
                            min_rerank_score: float | None = None) -> Dict:
        s = get_settings()
        if min_rerank_score is None:
            min_rerank_score = s.retrieval.min_rerank_score

        # 1. 检索层前置无证据拒答判定
        if not retrieved_chunks or (
            retrieved_chunks[0]["rerank_score"] > 0 and retrieved_chunks[0]["rerank_score"] < min_rerank_score
        ):
            return {
                "answer": "抱歉，当前知识库中没有找到足以回答该问题的可靠证据（无证据拒答）。",
                "refused": True,
                "citation_hard_precision": 1.0,
                "hallucinated_citations": [],
                "cost_rmb": 0.0,
                "sources": []
            }

        # 2. 组装带明确编号的父块上下文
        ctx_blocks = []
        for idx, c in enumerate(retrieved_chunks, 1):
            tag = f"[{c['doc_id']}:{c['chunk_id']}]"
            ctx_blocks.append(f"资料 {idx} 引用标签 {tag} (版本 v{float(c['version']):.1f}):\n{c['parent_text']}")
        user_prompt = "【参考资料】\n" + "\n\n".join(ctx_blocks) + f"\n\n【用户问题】\n{question}"

        llm_res = self.sf.chat_with_cost(SYSTEM_PROMPT, user_prompt, temperature=s.llm.temperature)
        raw_answer = llm_res["answer"] or ""

        # 3. 1ms 级在线引用硬校验（拦截编造编号幻觉）
        cleaned_answer, hallucinated, hard_precision = hard_check(raw_answer, retrieved_chunks)

        return {
            "answer": cleaned_answer,
            "refused": "无证据拒答" in cleaned_answer,
            "citation_hard_precision": hard_precision,
            "hallucinated_citations": hallucinated,
            "cost_rmb": llm_res["cost_rmb"],
            "sources": [{
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "version": c["version"],
                "section_path": c["section_path"],
                "rerank_score": c["rerank_score"]
            } for c in retrieved_chunks]
        }

    def verify_citation_soft_nli(self, answer: str, retrieved_chunks: List[Dict]) -> dict:
        """离线软校验（NLI 蕴含裁判）：判断每个带引用标签的句子是否被对应父块原文所蕴含"""
        s = get_settings()
        return soft_check_nli(answer, retrieved_chunks, self.sf, s.llm.judge_model)
