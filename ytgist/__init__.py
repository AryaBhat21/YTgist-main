"""
YTgist: Research-Grade Pipeline for YouTube Transcript Summarization & Extraction.

Architecture:
    YouTube Video → Transcript Acquisition → Text Preprocessing & Normalization
                                           ↓
                          Dynamic Chunking & Windowing
                                           ↓
                          Multi-Paradigm Summarization
                                           ↓
                        Key Information & Timeline Extraction
                                           ↓
                        Evaluation & Benchmark Metrics
                                           ↓
                                  Final Structured Gist
"""

__version__ = "1.0.0"
__author__ = "YTgist Research Team"

from .data_loader import TranscriptLoader, TranscriptItem
from .preprocessing import clean_transcript, restore_punctuation_heuristic, normalize_asr_text
from .chunking import (
    FixedCharChunker,
    TokenChunker,
    SentenceBoundaryChunker,
    RecursiveHierarchicalChunker,
    Chunk
)
from .summarizers import (
    TextRankSummarizer,
    TFIDFSalienceSummarizer,
    MapReduceSummarizer,
    RefineSummarizer,
    HierarchicalTreeSummarizer
)
from .extraction import (
    KeyPhraseExtractor,
    TimelineExtractor,
    KeyMoment
)
from .metrics import (
    compute_rouge,
    compute_compression_ratio,
    compute_redundancy_score,
    compute_lexical_diversity
)

__all__ = [
    "TranscriptLoader",
    "TranscriptItem",
    "clean_transcript",
    "restore_punctuation_heuristic",
    "normalize_asr_text",
    "FixedCharChunker",
    "TokenChunker",
    "SentenceBoundaryChunker",
    "RecursiveHierarchicalChunker",
    "Chunk",
    "TextRankSummarizer",
    "TFIDFSalienceSummarizer",
    "MapReduceSummarizer",
    "RefineSummarizer",
    "HierarchicalTreeSummarizer",
    "KeyPhraseExtractor",
    "TimelineExtractor",
    "KeyMoment",
    "compute_rouge",
    "compute_compression_ratio",
    "compute_redundancy_score",
    "compute_lexical_diversity"
]
