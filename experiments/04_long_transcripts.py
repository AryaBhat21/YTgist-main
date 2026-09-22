import json
import time
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ytgist.data_loader import TranscriptLoader
from ytgist.summarizers import (
    TextRankSummarizer,
    MapReduceSummarizer,
    HierarchicalTreeSummarizer
)
from ytgist.metrics import compute_rouge, compute_compression_ratio

def run_experiment_4() -> Dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 4: Handling Long YouTube Transcripts (Scalability & Hierarchy)")
    print("=" * 70)

    # 1. Short (5m)
    short_items = TranscriptLoader.load_sample("short_5min")
    short_text = TranscriptLoader.to_plain_text(short_items)

    # 2. Medium (25m)
    med_items = TranscriptLoader.load_sample("medium_25min")
    med_text = TranscriptLoader.to_plain_text(med_items)

    # 3. Long (65m)
    long_items = TranscriptLoader.load_sample("long_65min")
    long_text = TranscriptLoader.to_plain_text(long_items)

    # 4. Stress Mega (2x Long, ~21,000 words, >2 hours)
    mega_text = f"{long_text}\n\n{long_text}"

    test_corpora = [
        ("Short (5m talk)", short_text, "short_5min_tech_talk.json"),
        ("Medium (25m lecture)", med_text, "medium_25min_lecture.json"),
        ("Long (65m keynote)", long_text, "long_65min_conference.json"),
        ("Stress-Test (~2hr video)", mega_text, "long_65min_conference.json")
    ]

    models = [
        ("Single-Shot TextRank", lambda t: TextRankSummarizer().summarize(t, num_sentences=6)),
        ("Map-Reduce (512 tokens)", lambda t: MapReduceSummarizer(chunk_size=512).summarize(t, final_sentences=6)["summary"]),
        ("Hierarchical Tree", lambda t: HierarchicalTreeSummarizer(micro_tokens=512).summarize(t, final_sentences=6)["summary"])
    ]

    results = []

    for corp_name, text, ref_file in test_corpora:
        word_count = len(text.split())
        ref_path = Path(__file__).parent.parent / "data" / "sample_transcripts" / ref_file
        with open(ref_path, "r", encoding="utf-8") as f:
            reference = json.load(f)["metadata"]["reference_summary"]

        print(f"\n--- Corpus: {corp_name} ({word_count:,} words) ---")

        for m_name, fn in models:
            t0 = time.perf_counter()
            summary = fn(text)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            rouge = compute_rouge(summary, reference)
            comp = compute_compression_ratio(summary, text)

            rec = {
                "corpus": corp_name,
                "word_count": word_count,
                "strategy": m_name,
                "latency_ms": elapsed_ms,
                "summary_words": comp["summary_words"],
                "reduction_percentage": comp["reduction_percentage"],
                "rouge_1_f1": rouge["rouge-1"]["f1"],
                "rouge_l_f1": rouge["rouge-l"]["f1"]
            }
            results.append(rec)
            print(f"{m_name:<26} | Latency: {elapsed_ms:8.1f}ms | R-1 F1: {rec['rouge_1_f1']:.4f} | R-L F1: {rec['rouge_l_f1']:.4f}")

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp4_long_transcripts.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"experiment": "Experiment 4: Long Transcripts", "results": results}, f, indent=2)

    print(f"\n[Saved results to {out_file}]")
    return {"experiment": 4, "results": results}

if __name__ == "__main__":
    run_experiment_4()
