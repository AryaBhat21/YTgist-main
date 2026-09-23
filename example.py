import argparse
from ytgist import (
    TranscriptLoader,
    clean_transcript,
    restore_punctuation_heuristic,
    MapReduceSummarizer,
    KeyPhraseExtractor,
    TimelineExtractor,
)


def summarize_video(video_source: str = "short_5min", sentences: int = 4):
    """
    Summarize a YouTube video transcript or offline sample.
    
    Args:
        video_source: YouTube video ID/URL or sample name ('short_5min', 'medium_25min', 'long_65min')
        sentences: Number of summary sentences to generate
    """
    print("=" * 60)
    print(f"YTgist: Summarizing [{video_source}]")
    print("=" * 60)

    # 1. Fetch transcript
    if video_source in ["short_5min", "medium_25min", "long_65min"]:
        print(f"Loading offline sample: {video_source}...")
        items = TranscriptLoader.load_sample(video_source)
    else:
        print(f"Fetching YouTube transcript from YouTube API...")
        items = TranscriptLoader.load_from_youtube(video_source)

    print(f"Successfully loaded {len(items)} caption segments.")

    # 2. Preprocess & restore discourse punctuation
    print("Cleaning noise and formatting text...")
    cleaned_items = clean_transcript(items, remove_fillers=True)
    plain_text = TranscriptLoader.to_plain_text(cleaned_items)
    punctuated_text = restore_punctuation_heuristic(plain_text)

    # 3. Summarize using Map-Reduce
    print("Generating summary...")
    summarizer = MapReduceSummarizer(chunk_size=512, overlap_ratio=0.10)
    result = summarizer.summarize(punctuated_text, final_sentences=sentences)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result["summary"])

    # 4. Extract Top Keyphrases
    print("\n" + "=" * 60)
    print("TOP KEY PHRASES")
    print("=" * 60)
    extractor = KeyPhraseExtractor(max_phrases=5)
    for phrase, score in extractor.extract(punctuated_text):
        print(f" - {phrase} (score: {score:.2f})")

    # 5. Extract Video Timeline Key Moments
    print("\n" + "=" * 60)
    print("TIMELINE & KEY MOMENTS")
    print("=" * 60)
    timeline = TimelineExtractor(top_k=4).extract_from_transcript(cleaned_items)
    for moment in timeline:
        preview = (moment.text[:75] + "...") if len(moment.text) > 75 else moment.text
        print(f" [{moment.timestamp_str}] {preview}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YTgist: YouTube Video Summarizer & Key Moments Extractor")
    parser.add_argument(
        "--video",
        type=str,
        default="short_5min",
        help="YouTube video ID, full URL, or sample name (short_5min, medium_25min, long_65min). Default: short_5min",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=4,
        help="Number of summary sentences to extract (default: 4)",
    )
    args = parser.parse_args()
    summarize_video(args.video, args.sentences)
