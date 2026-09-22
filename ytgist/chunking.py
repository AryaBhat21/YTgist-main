import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from .data_loader import TranscriptItem

@dataclass
class Chunk:
    text: str
    chunk_index: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    token_count: int = 0
    char_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_count:
            # Approximate 1 token ~= 0.75 words, or simple word count based estimation
            self.token_count = max(1, int(len(self.text.split()) * 1.3))


class FixedCharChunker:
    """Fixed-size character chunking with optional overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap
        idx = 0
        for start in range(0, len(text), step):
            sub = text[start:start + self.chunk_size].strip()
            if sub:
                chunks.append(Chunk(text=sub, chunk_index=idx))
                idx += 1
            if start + self.chunk_size >= len(text):
                break
        return chunks


class TokenChunker:
    """
    Token/Word based chunking with percentage or absolute token overlap.
    Supports preserving timestamp intervals when items are provided.
    """

    def __init__(self, target_tokens: int = 512, overlap_ratio: float = 0.1):
        self.target_tokens = target_tokens
        self.overlap_ratio = max(0.0, min(overlap_ratio, 0.5))
        self.overlap_tokens = int(self.target_tokens * self.overlap_ratio)

    def chunk_text(self, text: str) -> List[Chunk]:
        words = text.split()
        target_words = max(1, int(self.target_tokens * 0.75))
        overlap_words = int(target_words * self.overlap_ratio)
        step = max(1, target_words - overlap_words)

        chunks: List[Chunk] = []
        idx = 0
        for i in range(0, len(words), step):
            window = words[i:i + target_words]
            if not window:
                break
            chunk_str = " ".join(window)
            chunks.append(Chunk(text=chunk_str, chunk_index=idx))
            idx += 1
            if i + target_words >= len(words):
                break
        return chunks

    def chunk_transcript_items(self, items: List[TranscriptItem]) -> List[Chunk]:
        """Groups timestamped transcript items into token chunks while tracking timestamps."""
        if not items:
            return []

        chunks: List[Chunk] = []
        curr_words: List[str] = []
        curr_start: Optional[float] = None
        curr_end: Optional[float] = None
        target_words = max(1, int(self.target_tokens * 0.75))
        idx = 0

        for item in items:
            words = item.text.split()
            if not words:
                continue

            if curr_start is None:
                curr_start = item.start
            curr_end = item.start + item.duration

            curr_words.extend(words)

            if len(curr_words) >= target_words:
                chunk_str = " ".join(curr_words)
                chunks.append(Chunk(
                    text=chunk_str,
                    chunk_index=idx,
                    start_time=curr_start,
                    end_time=curr_end,
                    token_count=int(len(curr_words) * 1.3)
                ))
                idx += 1
                # Overlap step
                overlap_count = int(target_words * self.overlap_ratio)
                curr_words = curr_words[-overlap_count:] if overlap_count > 0 else []
                curr_start = item.start if curr_words else None

        if curr_words:
            chunks.append(Chunk(
                text=" ".join(curr_words),
                chunk_index=idx,
                start_time=curr_start,
                end_time=curr_end,
                token_count=int(len(curr_words) * 1.3)
            ))

        return chunks


class SentenceBoundaryChunker:
    """
    Sentence-aware chunker that never cuts sentences in half.
    Accumulates full sentences until the token limit is reached.
    """

    def __init__(self, target_tokens: int = 512, overlap_sentences: int = 1):
        self.target_tokens = target_tokens
        self.target_words = max(1, int(target_tokens * 0.75))
        self.overlap_sentences = overlap_sentences

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        # Split on sentence terminals while keeping cohesion
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[Chunk]:
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sents: List[str] = []
        current_word_count = 0
        idx = 0

        for s in sentences:
            s_words = len(s.split())
            if current_word_count + s_words > self.target_words and current_sents:
                chunk_text = " ".join(current_sents)
                chunks.append(Chunk(text=chunk_text, chunk_index=idx))
                idx += 1
                # Retain overlap sentences
                if self.overlap_sentences > 0:
                    current_sents = current_sents[-self.overlap_sentences:]
                    current_word_count = sum(len(x.split()) for x in current_sents)
                else:
                    current_sents = []
                    current_word_count = 0

            current_sents.append(s)
            current_word_count += s_words

        if current_sents:
            chunks.append(Chunk(text=" ".join(current_sents), chunk_index=idx))

        return chunks


class RecursiveHierarchicalChunker:
    """
    Builds a hierarchical tree of chunks for long transcripts (>1hr).
    Level 0: Micro-chunks (e.g. 512 tokens)
    Level 1: Macro-chunks (aggregations of level 0)
    """

    def __init__(self, micro_tokens: int = 512, macro_factor: int = 4):
        self.micro_chunker = SentenceBoundaryChunker(target_tokens=micro_tokens, overlap_sentences=1)
        self.macro_factor = macro_factor

    def chunk_text(self, text: str) -> Dict[str, List[Chunk]]:
        micro_chunks = self.micro_chunker.chunk_text(text)
        macro_chunks: List[Chunk] = []

        for i in range(0, len(micro_chunks), self.macro_factor):
            group = micro_chunks[i:i + self.macro_factor]
            combined_text = "\n\n".join(c.text for c in group)
            start_t = group[0].start_time if group[0].start_time is not None else None
            end_t = group[-1].end_time if group[-1].end_time is not None else None
            macro_chunks.append(Chunk(
                text=combined_text,
                chunk_index=len(macro_chunks),
                start_time=start_t,
                end_time=end_t
            ))

        return {
            "micro": micro_chunks,
            "macro": macro_chunks
        }
