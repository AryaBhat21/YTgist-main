import re
from typing import List, Dict, Tuple, Set
from collections import Counter

def tokenize(text: str) -> List[str]:
    """Basic lowercased alphanumeric tokenizer."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())

def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def lcs_length(x: List[str], y: List[str]) -> int:
    """Computes length of Longest Common Subsequence between two token sequences."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def compute_ngram_rouge(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> Dict[str, float]:
    cand_ngrams = get_ngrams(candidate_tokens, n)
    ref_ngrams = get_ngrams(reference_tokens, n)

    if not cand_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    overlap = 0
    for ng, count in cand_counts.items():
        overlap += min(count, ref_counts.get(ng, 0))

    precision = overlap / len(cand_ngrams) if cand_ngrams else 0.0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> Dict[str, float]:
    lcs = lcs_length(candidate_tokens, reference_tokens)
    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def compute_rouge(candidate: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores between candidate summary and reference.
    Returns precision, recall, and f1 for each metric.
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    return {
        "rouge-1": compute_ngram_rouge(cand_tokens, ref_tokens, 1),
        "rouge-2": compute_ngram_rouge(cand_tokens, ref_tokens, 2),
        "rouge-l": compute_rouge_l(cand_tokens, ref_tokens)
    }

def compute_compression_ratio(summary: str, source: str) -> Dict[str, float]:
    """Calculates compression ratio in terms of words and characters."""
    sum_words = len(tokenize(summary))
    src_words = len(tokenize(source))
    sum_chars = len(summary)
    src_chars = len(source)

    word_ratio = sum_words / src_words if src_words > 0 else 0.0
    char_ratio = sum_chars / src_chars if src_chars > 0 else 0.0

    return {
        "source_words": src_words,
        "summary_words": sum_words,
        "compression_ratio": round(word_ratio, 4),
        "reduction_percentage": round((1.0 - word_ratio) * 100, 2),
        "char_ratio": round(char_ratio, 4)
    }

def compute_redundancy_score(text: str, n: int = 3) -> float:
    """
    Measures repetitive content based on repeated n-grams.
    0.0 = completely unique n-grams; 1.0 = highly redundant.
    """
    tokens = tokenize(text)
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    unique_count = len(set(ngrams))
    total_count = len(ngrams)
    # Ratio of repeated ngrams
    redundancy = 1.0 - (unique_count / total_count)
    return round(max(0.0, redundancy), 4)

def compute_lexical_diversity(text: str) -> float:
    """Type-Token Ratio (TTR): unique tokens divided by total tokens."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)
