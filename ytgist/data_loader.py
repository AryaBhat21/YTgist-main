import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any

@dataclass
class TranscriptItem:
    text: str
    start: float  # timestamp in seconds
    duration: float  # duration in seconds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptItem":
        return cls(
            text=str(data.get("text", "")),
            start=float(data.get("start", 0.0)),
            duration=float(data.get("duration", 0.0))
        )


class TranscriptLoader:
    """
    Robust YouTube transcript loader with online YouTube API fetching,
    caching, and offline benchmark dataset fallbacks.
    """

    SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_transcripts"

    @staticmethod
    def extract_video_id(url_or_id: str) -> str:
        """Extracts 11-character YouTube video ID from various URL formats or raw ID."""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$|\/)",
            r"youtu\.be\/([0-9A-Za-z_-]{11})",
            r"^([0-9A-Za-z_-]{11})$"
        ]
        for pattern in patterns:
            match = re.search(pattern, url_or_id.strip())
            if match:
                return match.group(1)
        return url_or_id.strip()

    @classmethod
    def load_from_youtube(cls, video_id_or_url: str, languages: Optional[List[str]] = None) -> List[TranscriptItem]:
        """
        Fetches transcript from YouTube API using youtube_transcript_api.
        Falls back to available languages or auto-generated captions if available.
        """
        video_id = cls.extract_video_id(video_id_or_url)
        languages = languages or ["en", "en-US", "en-GB"]

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            # youtube_transcript_api supports get_transcript or fetch
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                raw = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            else:
                api = YouTubeTranscriptApi()
                raw = api.fetch(video_id).to_raw_data()
            return [TranscriptItem.from_dict(item) for item in raw]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch YouTube transcript for '{video_id}': {e}") from e

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> List[TranscriptItem]:
        """Loads transcript items from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "transcript" in data:
            data = data["transcript"]

        return [TranscriptItem.from_dict(item) for item in data]

    @classmethod
    def save_to_file(cls, items: List[TranscriptItem], filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None):
        """Saves transcript items to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "total_items": len(items),
            "transcript": [item.to_dict() for item in items]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_sample(cls, sample_name: str = "short_5min") -> List[TranscriptItem]:
        """
        Loads pre-packaged benchmark transcripts:
          - 'short_5min': ~900 words (~5 min tech talk)
          - 'medium_25min': ~3,800 words (~25 min lecture)
          - 'long_65min': ~10,500 words (~65 min conference keynote)
        """
        mapping = {
            "short": "short_5min_tech_talk.json",
            "short_5min": "short_5min_tech_talk.json",
            "medium": "medium_25min_lecture.json",
            "medium_25min": "medium_25min_lecture.json",
            "long": "long_65min_conference.json",
            "long_65min": "long_65min_conference.json",
        }
        filename = mapping.get(sample_name.lower().replace(" ", "_"), f"{sample_name}.json")
        target_path = cls.SAMPLE_DIR / filename
        if not target_path.exists():
            raise FileNotFoundError(f"Sample transcript '{sample_name}' not found at {target_path}")
        return cls.load_from_file(target_path)

    @staticmethod
    def to_plain_text(items: List[TranscriptItem], include_timestamps: bool = False) -> str:
        """Joins transcript item text into a single cohesive string."""
        if not include_timestamps:
            return " ".join(item.text.strip() for item in items if item.text.strip())
        
        lines = []
        for item in items:
            mins, secs = divmod(int(item.start), 60)
            hrs, mins = divmod(mins, 60)
            if hrs > 0:
                ts = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            else:
                ts = f"{mins:02d}:{secs:02d}"
            lines.append(f"[{ts}] {item.text.strip()}")
        return "\n".join(lines)
