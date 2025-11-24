#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动 RAG Demo（CPU + GPU，无 dummy CUDA）：

- CPU：
    * 从网络抓取中文维基百科页面作为知识库
    * 文本切块
    * SentenceTransformer 在 CPU 上做文档 & query embedding
    * 暴力向量检索 + rerank

- GPU：
    * 中文 GPT2 模型在 GPU 上做生成回答（完整 RAG）

特点：
    * 自动跑一组预定义问题，程序结束后自动退出
    * 不需要用户输入
    * 不做 dummy CUDA 计算，所有 CUDA kernel 都来自 LLM 推理
"""

import re
import requests
import numpy as np
import torch

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 配置参数
# =========================

# 向量模型（CPU）
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM（GPU）：公开中文 GPT2，小模型，免权限
LLM_MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"

# 请求头，避免 Wikipedia 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# 外部知识库 URL
URL_LIST = [
    "https://zh.wikipedia.org/zh-cn/深度学习",
    "https://zh.wikipedia.org/zh-cn/卷积神经网络",
    "https://zh.wikipedia.org/zh-cn/自然语言处理",
]

# 文本切块设置：稍微保守一点，避免 prompt 过长
CHUNK_SIZE = 600           # 每块最大字符数
MAX_CHUNKS_PER_URL = 6     # 每个 URL 最多切多少块

TOP_K = 5                  # 检索返回多少块文档
RERANK_LOOPS = 2           # rerank 重复计算次数（调大更吃 CPU）

MAX_NEW_TOKENS = 64        # LLM 生成长度（适当短一点，避免超长）

# 自动测试问题列表（无需输入）
TEST_QUERIES = [
    "什么是深度学习？",
    "卷积神经网络的典型应用是什么？",
    "自然语言处理主要研究什么？",
]


# =========================
# 1. 从网络加载知识库（CPU）
# =========================

def fetch_url_text(url: str) -> str:
    print(f"[Web] Fetching: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, max_chunks: int):
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks


def load_web_corpus():
    docs = []
    meta = []
    for url in URL_LIST:
        try:
            full_text = fetch_url_text(url)
        except Exception as e:
            print(f"[Web][Error] 无法获取 {url}: {e}")
            continue

        chunks = chunk_text(full_text, CHUNK_SIZE, MAX_CHUNKS_PER_URL)
        for i, c in enumerate(chunks):
            if len(c.strip()) < 100:
                continue
            docs.append(c)
            meta.append(f"{url}#chunk{i}")

    print(f"[Web] 成功加载文档块数量: {len(docs)}\n")
    return docs, meta


# =========================
# 2. Embedding（CPU）
# =========================

def load_embedding_model():
    print(f"[Embedding] Loading '{EMBEDDING_MODEL_NAME}' on CPU...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def encode_corpus(emb_model, docs):
    print(f"[Embedding] Encoding {len(docs)} docs on CPU...")
    emb = emb_model.encode(
        docs,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=True,
    )
    print("[Embedding] 文档向量编码完成。\n")
    return emb


# =========================
# 3. 检索 + rerank（CPU）
# =========================

def brute_force_search(query_emb, corpus_embs, top_k):
    sims = np.dot(corpus_embs, query_emb)  # (N, dim) · (dim,) -> (N,)
    idx = np.argpartition(-sims, top_k)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def heavy_rerank(query_emb, candidate_embs, loops):
    scores = None
    for _ in range(loops):
        tmp = [float(np.dot(emb, query_emb)) for emb in candidate_embs]
        scores = np.array(tmp)
    rerank_idx = np.argsort(-scores)
    return rerank_idx, scores[rerank_idx]


# =========================
# 4. LLM 生成（GPU）
# =========================

def load_llm():
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 GPU，无法运行 GPU LLM。")

    print(f"[LLM] Loading tokenizer & model '{LLM_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)

    print("[LLM] 模型已加载到 GPU：", device, "\n")
    return tokenizer, model


def generate_answer(tokenizer, model, query, retrieved_docs):
    """
    使用 GPU 上的 LLM，根据检索到的文档生成回答。
    控制输入长度不超过模型最大长度，避免 CUDA 报错。
    只要 input_ids 在 cuda 上，forward/generate 过程必然触发 CUDA kernel launch。
    """
    context = "\n\n".join(f"[Doc {i}]\n{d}" for i, d in enumerate(retrieved_docs))

    system_prompt = (
        "你是一个中文技术助手。下面给出的是从百科文档中检索到的内容，"
        "请尽量基于这些内容，用中文回答用户的问题。如果文档信息不足，可以适当补充。"
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"[文档内容]\n{context}\n\n"
        f"[用户问题]\n{query}\n\n"
        f"[回答]\n"
    )

    # 先在 CPU 上 tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # 控制最大输入长度，避免超过 max_position_embeddings
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None:
        max_input_len = max_positions - MAX_NEW_TOKENS - 16
        if max_input_len < 0:
            max_input_len = max_positions
        seq_len = input_ids.shape[1]
        if seq_len > max_input_len:
            input_ids = input_ids[:, -max_input_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -max_input_len:]

    # 移到 GPU
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 这里的 generate 在 GPU 上运行，会触发一堆 CUDA kernels（matmul/attention 等）
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[回答]" in text:
        answer = text.split("[回答]")[-1].strip()
    else:
        answer = text
    return answer


# =========================
# 5. 主流程（自动执行，自动结束）
# =========================

def main():
    # 1) 加载网络知识库（CPU）
    docs, meta = load_web_corpus()
    if not docs:
        print("[Error] 没有成功加载任何文档，退出。")
        return

    # 2) 文档 embedding（CPU）
    emb_model = load_embedding_model()
    corpus_embs = encode_corpus(emb_model, docs)

    # 3) 加载 LLM（GPU）
    tokenizer, llm = load_llm()

    print("=========== 自动 RAG 测试开始（CPU 检索 + GPU LLM） ===========\n")

    for query in TEST_QUERIES:
        print(f"=== 问题：{query} ===")

        # A) query embedding（CPU）
        query_emb = emb_model.encode(query, convert_to_numpy=True)

        # B) 检索（CPU）
        topk_idx, sims = brute_force_search(query_emb, corpus_embs, TOP_K)

        # C) rerank（CPU）
        candidate_embs = corpus_embs[topk_idx]
        rerank_idx, rerank_scores = heavy_rerank(query_emb, candidate_embs, RERANK_LOOPS)
        final_idx = topk_idx[rerank_idx]
        retrieved_docs = [docs[i] for i in final_idx]
        retrieved_meta = [meta[i] for i in final_idx]

        print("\n[RAG] 检索到的文档块来源：")
        for m, s in zip(retrieved_meta, rerank_scores):
            print(f"  - {m}  (score={s:.4f})")

        # D) LLM 生成（GPU）
        answer = generate_answer(tokenizer, llm, query, retrieved_docs)

        print("\n[回答]:")
        print(answer)
        print("\n" + "=" * 70 + "\n")

    print("=========== 所有自动 RAG 测试完成，程序结束 ===========\n")


if __name__ == "__main__":
    main()
