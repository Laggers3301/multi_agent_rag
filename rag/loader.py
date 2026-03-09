"""
文档加载与文本分块
职责：将本地文件或原始文本转换为可检索的文本块。
"""

from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP


def recursive_split(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    递归字符分割 (RecursiveCharacterTextSplitter 简化版)。
    按优先级依次尝试：双换行 → 单换行 → 句号 → 空格 → 硬截断。
    面试话术：分块策略直接影响召回质量，overlap 保证上下文不在关键语句处断裂。
    """
    separators = ["\n\n", "\n", "。", "！", "？", ".", " "]

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            current_chunk = ""
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            break
    else:
        # 硬截断 fallback
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

    return chunks


def load_text_file(filepath: str) -> List[str]:
    """加载本地文本文件并分块。"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return recursive_split(text)


def load_texts(texts: List[str]) -> List[str]:
    """将多段原始文本分块。"""
    all_chunks = []
    for text in texts:
        all_chunks.extend(recursive_split(text))
    return all_chunks
