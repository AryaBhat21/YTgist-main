import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from .data_loader import TranscriptItem
from .vectorizer import FastTFIDF

@dataclass
class KeyMoment:
    timestamp_sec: float
    timestamp_str: str
    text: str
    importance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_sec": self.timestamp_sec,
            "timestamp_str": self.timestamp_str,
            "text": self.text,
            "importance_score": round(self.importance_score, 4)
        }


def format_seconds(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS or MM:SS string."""
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def parse_timestamp_str(ts_str: str) -> float:
    """Parses timestamp string like '04:12' or '1:15:30' into total seconds."""
    parts = [int(p) for p in ts_str.split(":")]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    elif len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    elif len(parts) == 1:
        return float(parts[0])
    return 0.0


class KeyPhraseExtractor:
    """
    Extracts high-salience technical terms, entities, and multi-word key phrases
    from YouTube transcripts using n-gram TF-IDF and term co-occurrence salience.
    """

    def __init__(self, max_phrases: int = 15, ngram_range: Tuple[int, int] = (1, 3)):
        self.max_phrases = max_phrases
        self.ngram_range = ngram_range

    def extract(self, text: str) -> List[Tuple[str, float]]:
        if not text.strip():
            return []

        # Split text into pseudo-documents (sentences or paragraphs)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]
        if len(sentences) < 2:
            sentences = [text]

        vectorizer = FastTFIDF(
            stop_words="english",
            ngram_range=self.ngram_range,
            min_df=1,
            max_features=200
        )
        X = vectorizer.fit_transform(sentences)
        features = vectorizer.get_feature_names_out()
        if len(features) == 0:
            return []

        scores = X.sum(axis=0)
        ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)

        selected: List[Tuple[str, float]] = []
        for phrase, score in ranked:
            if len(selected) >= self.max_phrases:
                break
            selected.append((phrase, float(score)))

        return selected


class TimelineExtractor:
    """
    Extracts key moments and timestamps across the video duration.
    Correlates high-density transcript moments with optional video description chapters.
    """

    def __init__(self, top_k: int = 6):
        self.top_k = top_k

    def extract_from_transcript(self, items: List[TranscriptItem]) -> List[KeyMoment]:
        if not items:
            return []

        corpus = [item.text for item in items]
        vectorizer = FastTFIDF(stop_words="english")
        X = vectorizer.fit_transform(corpus)
        if X.shape[1] > 0:
            scores = X.sum(axis=1)
        else:
            scores = [len(item.text.split()) for item in items]

        # Select top-k distinct time points
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chosen: List[KeyMoment] = []
        used_times = []

        for idx in top_indices:
            item = items[idx]
            # Avoid picking moments closer than 30 seconds to an existing moment
            if any(abs(item.start - t) < 30 for t in used_times):
                continue

            chosen.append(KeyMoment(
                timestamp_sec=item.start,
                timestamp_str=format_seconds(item.start),
                text=item.text.strip(),
                importance_score=float(scores[idx])
            ))
            used_times.append(item.start)
            if len(chosen) >= self.top_k:
                break

        # Sort chronologically
        return sorted(chosen, key=lambda m: m.timestamp_sec)

    @staticmethod
    def extract_from_description(description: str) -> List[Tuple[str, str]]:
        """Parses description text for chapters/timestamps e.g. '02:15 Introduction'."""
        results: List[Tuple[str, str]] = []
        for line in description.split("\n"):
            match = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", line)
            if match:
                ts = match.group(1)
                label = line.replace(ts, "").strip(" -:|")
                results.append((ts, label))
        return results
