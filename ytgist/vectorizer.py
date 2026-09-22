import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
import numpy as np

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

class FastTFIDF:
    """
    Lightweight, high-performance TF-IDF vectorizer with zero heavy dependencies.
    Computes standard sublinear TF-IDF and cosine normalized sentence vectors.
    """

    def __init__(
        self,
        stop_words: str = "english",
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = None,
        min_df: int = 1
    ):
        self.stop_words = ENGLISH_STOP_WORDS if stop_words == "english" else set()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.idf_: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        tokens = []
        n_min, n_max = self.ngram_range
        for n in range(n_min, n_max + 1):
            if n == 1:
                tokens.extend([w for w in words if w not in self.stop_words])
            else:
                for i in range(len(words) - n + 1):
                    gram = words[i:i + n]
                    if not all(w in self.stop_words for w in gram):
                        tokens.append(" ".join(gram))
        return tokens

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        doc_tokens = [self._tokenize(doc) for doc in documents]
        N = len(documents)

        # Document frequencies
        df_counts: Counter = Counter()
        for toks in doc_tokens:
            df_counts.update(set(toks))

        # Filter by min_df
        valid_terms = [t for t, count in df_counts.items() if count >= self.min_df]
        if self.max_features:
            valid_terms = sorted(valid_terms, key=lambda t: df_counts[t], reverse=True)[:self.max_features]

        valid_terms.sort()
        self.vocabulary_ = {term: idx for idx, term in enumerate(valid_terms)}
        self.feature_names_ = valid_terms

        V = len(valid_terms)
        if V == 0 or N == 0:
            return np.zeros((N, 0), dtype=np.float32)

        # Compute smooth IDF: log((1 + N) / (1 + df)) + 1
        idf = np.zeros(V, dtype=np.float32)
        for term, idx in self.vocabulary_.items():
            idf[idx] = math.log((1.0 + N) / (1.0 + df_counts[term])) + 1.0
        self.idf_ = idf

        # Build TF-IDF matrix
        matrix = np.zeros((N, V), dtype=np.float32)
        for d_idx, toks in enumerate(doc_tokens):
            t_counts = Counter(toks)
            total = len(toks) if toks else 1
            for term, count in t_counts.items():
                if term in self.vocabulary_:
                    w_idx = self.vocabulary_[term]
                    matrix[d_idx, w_idx] = (count / total) * self.idf_[w_idx]

        # L2 row normalization
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        matrix = matrix / row_norms

        return matrix

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self.feature_names_)
