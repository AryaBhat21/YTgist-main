import re
import numpy as np
from typing import List, Dict, Any, Optional
from .chunking import TokenChunker, SentenceBoundaryChunker, Chunk
from .vectorizer import FastTFIDF

class TextRankSummarizer:
    """
    Graph-based extractive summarization based on Mihalcea & Tarau (2004).
    Builds sentence similarity matrix via TF-IDF cosine similarity,
    runs PageRank iterations, and returns top ranked sentences chronologically.
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-4):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip().split()) >= 3]

    def summarize(self, text: str, num_sentences: int = 5) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Build TF-IDF sentence representation
        vectorizer = FastTFIDF(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        if tfidf_matrix.shape[1] == 0:
            return " ".join(sentences[:num_sentences])

        # Cosine similarity matrix S = (A * A.T)
        sim_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)
        np.fill_diagonal(sim_matrix, 0.0)

        # Row normalize
        row_sums = sim_matrix.sum(axis=1)
        row_sums[row_sums == 0.0] = 1.0
        norm_matrix = sim_matrix / row_sums[:, np.newaxis]

        # Power iteration for PageRank
        N = len(sentences)
        scores = np.ones(N, dtype=np.float32) / N
        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            scores = (1 - self.damping) / N + self.damping * np.dot(norm_matrix.T, prev_scores)
            if np.linalg.norm(scores - prev_scores) < self.tol:
                break

        # Pick top sentences and preserve original narrative order
        top_indices = sorted(np.argsort(scores)[-num_sentences:])
        selected = [sentences[i] for i in top_indices]
        return " ".join(selected)


class TFIDFSalienceSummarizer:
    """
    Salience-based extractive summarizer ranking sentences by the cumulative
    importance of non-stopword TF-IDF terms normalized by sentence length.
    """

    def __init__(self):
        self.vectorizer = FastTFIDF(stop_words="english")

    def summarize(self, text: str, num_sentences: int = 5) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip().split()) >= 3]
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        X = self.vectorizer.fit_transform(sentences)
        if X.shape[1] == 0:
            return " ".join(sentences[:num_sentences])

        # Sum of TF-IDF scores per sentence normalized by length
        raw_scores = X.sum(axis=1)
        lengths = np.array([max(1, len(s.split())) for s in sentences], dtype=np.float32)
        scores = raw_scores / np.sqrt(lengths)

        top_indices = sorted(np.argsort(scores)[-num_sentences:])
        return " ".join([sentences[i] for i in top_indices])


class MapReduceSummarizer:
    """
    Map-Reduce summarization paradigm:
    1. Map: Split document into chunks and generate intermediate chunk-level summaries.
    2. Reduce: Concatenate chunk summaries and generate a consolidated meta-summary.
    """

    def __init__(self, chunk_size: int = 512, overlap_ratio: float = 0.1, base_summarizer=None):
        self.chunker = TokenChunker(target_tokens=chunk_size, overlap_ratio=overlap_ratio)
        self.base_summarizer = base_summarizer or TextRankSummarizer()

    def summarize(self, text: str, final_sentences: int = 5) -> Dict[str, Any]:
        chunks = self.chunker.chunk_text(text)
        if len(chunks) <= 1:
            summary = self.base_summarizer.summarize(text, num_sentences=final_sentences)
            return {
                "summary": summary,
                "intermediate_summaries": [summary],
                "num_chunks": 1
            }

        # MAP phase
        intermediate_summaries: List[str] = []
        for chunk in chunks:
            chunk_summary = self.base_summarizer.summarize(chunk.text, num_sentences=3)
            if chunk_summary.strip():
                intermediate_summaries.append(chunk_summary)

        # REDUCE phase
        reduced_corpus = " ".join(intermediate_summaries)
        final_summary = self.base_summarizer.summarize(reduced_corpus, num_sentences=final_sentences)

        return {
            "summary": final_summary,
            "intermediate_summaries": intermediate_summaries,
            "num_chunks": len(chunks)
        }


class RefineSummarizer:
    """
    Iterative Refine summarization paradigm:
    Sequentially processes chunks, updating an ongoing accumulator summary
    as new chunk information arrives.
    """

    def __init__(self, chunk_size: int = 512, base_summarizer=None):
        self.chunker = SentenceBoundaryChunker(target_tokens=chunk_size, overlap_sentences=1)
        self.base_summarizer = base_summarizer or TextRankSummarizer()

    def summarize(self, text: str, final_sentences: int = 5) -> Dict[str, Any]:
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            return {"summary": "", "history": []}

        # Initialize with first chunk
        running_summary = self.base_summarizer.summarize(chunks[0].text, num_sentences=3)
        history = [running_summary]

        for chunk in chunks[1:]:
            # Combine current state with new context and refine
            combined = f"{running_summary}. Furthermore, {chunk.text}"
            running_summary = self.base_summarizer.summarize(combined, num_sentences=final_sentences)
            history.append(running_summary)

        return {
            "summary": running_summary,
            "refinement_steps": len(chunks),
            "history": history
        }


class HierarchicalTreeSummarizer:
    """
    Hierarchical multi-level summarizer designed for long transcripts (>60 minutes).
    Compacts micro-level chunks to intermediate synopses, then consolidates
    at macro level.
    """

    def __init__(self, micro_tokens: int = 512, base_summarizer=None):
        self.micro_chunker = SentenceBoundaryChunker(target_tokens=micro_tokens, overlap_sentences=1)
        self.base_summarizer = base_summarizer or TextRankSummarizer()

    def summarize(self, text: str, final_sentences: int = 7) -> Dict[str, Any]:
        chunks = self.micro_chunker.chunk_text(text)
        if len(chunks) <= 2:
            return {
                "summary": self.base_summarizer.summarize(text, num_sentences=final_sentences),
                "levels": 1
            }

        # Level 1: Micro chunk summaries
        l1_summaries = [
            self.base_summarizer.summarize(c.text, num_sentences=2)
            for c in chunks
        ]

        # Level 2: Cluster every 3 micro-summaries
        l2_groups: List[str] = []
        cluster_size = 3
        for i in range(0, len(l1_summaries), cluster_size):
            group_text = " ".join(l1_summaries[i:i + cluster_size])
            l2_groups.append(self.base_summarizer.summarize(group_text, num_sentences=3))

        # Root level: Final synthesis
        root_input = " ".join(l2_groups)
        final_summary = self.base_summarizer.summarize(root_input, num_sentences=final_sentences)

        return {
            "summary": final_summary,
            "levels": 3,
            "num_micro_chunks": len(chunks),
            "num_l2_clusters": len(l2_groups)
        }
