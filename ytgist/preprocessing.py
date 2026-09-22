import re
import html
from typing import List, Tuple
from .data_loader import TranscriptItem

# Regular expressions for ASR noise and artifacts
ASR_NOISE_PATTERNS = [
    r"\[(?:music|applause|laughter|cheering|silence|snort|inaudible)\]",
    r"\((?:music|applause|laughter|cheering|silence|inaudible)[^\)]*\)",
    r"^>>\s*",
    r"\[\s*__\s*\]"  # Bleeped profanity/blank tokens
]

FILLER_WORDS = [
    r"\b(?:uh|um|er|ah|like|you know|sort of|kind of|i mean)\b"
]

def clean_transcript_text(text: str, remove_fillers: bool = False) -> str:
    """
    Cleans raw transcript text:
    - Decodes HTML entities (&amp; -> &)
    - Removes ASR system markers like [Music], [Applause]
    - Normalizes multiple spaces and newlines
    - Optionally suppresses verbal fillers
    """
    if not text:
        return ""

    # Decode HTML
    cleaned = html.unescape(text)

    # Remove sound markers
    for pat in ASR_NOISE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    # Optionally remove filler words
    if remove_fillers:
        for filler in FILLER_WORDS:
            cleaned = re.sub(filler, " ", cleaned, flags=re.IGNORECASE)

    # Collapse consecutive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def restore_punctuation_heuristic(text: str) -> str:
    """
    Restores basic sentence boundaries and capitalization using syntax cues
    common in spoken YouTube transcripts when raw ASR lacks punctuation.
    """
    if not text:
        return ""

    # Common discourse markers that frequently indicate sentence beginnings
    sentence_starters = [
        r"(?<=[a-z0-9])\s+(so|now|and then|first of all|secondly|furthermore|however|in addition|on the other hand|in conclusion|finally)\b",
        r"(?<=[a-z0-9])\s+(what we have here|the next thing is|as you can see|let's talk about)\b"
    ]

    restored = text
    for starter in sentence_starters:
        restored = re.sub(starter, r". \1", restored, flags=re.IGNORECASE)

    # Capitalize first letter of each sentence
    def cap(match):
        return match.group(1) + match.group(2).upper()

    restored = re.sub(r"(^|[.!?]\s+)([a-z])", cap, restored)
    
    # Ensure ending punctuation
    if restored and restored[-1] not in ".!?":
        restored += "."

    return restored


def clean_transcript(
    items: List[TranscriptItem],
    min_word_length: int = 1,
    remove_fillers: bool = False,
    merge_short_intervals: bool = True,
    interval_threshold_sec: float = 1.5
) -> List[TranscriptItem]:
    """
    Cleans and standardizes a list of TranscriptItems:
    1. Cleans noise artifacts per segment.
    2. Filters out empty or degenerate segments.
    3. Merges rapid adjacent segments if start times are within interval_threshold_sec.
    """
    cleaned_items: List[TranscriptItem] = []

    for item in items:
        cleaned_text = clean_transcript_text(item.text, remove_fillers=remove_fillers)
        if len(cleaned_text.split()) >= min_word_length:
            cleaned_items.append(TranscriptItem(
                text=cleaned_text,
                start=item.start,
                duration=item.duration
            ))

    if not merge_short_intervals or not cleaned_items:
        return cleaned_items

    merged: List[TranscriptItem] = []
    current = cleaned_items[0]

    for nxt in cleaned_items[1:]:
        gap = nxt.start - (current.start + current.duration)
        # If segments are contiguous or overlapping (typical in YouTube auto-captions)
        if gap <= interval_threshold_sec and len(current.text.split()) < 25:
            current = TranscriptItem(
                text=f"{current.text} {nxt.text}".strip(),
                start=current.start,
                duration=(nxt.start + nxt.duration) - current.start
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def normalize_asr_text(text: str) -> str:
    """Full normalization pipeline for raw continuous transcript text."""
    cleaned = clean_transcript_text(text, remove_fillers=False)
    punctuated = restore_punctuation_heuristic(cleaned)
    return punctuated
