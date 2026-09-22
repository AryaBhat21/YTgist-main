import json
import time
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ytgist.data_loader import TranscriptLoader
from ytgist.summarizers import (
    TextRankSummarizer,
    TFIDFSalienceSummarizer,
    MapReduceSummarizer,
    RefineSummarizer,
    HierarchicalTreeSummarizer
)
from ytgist.metrics import (
    compute_rouge,
    compute_compression_ratio,
    compute_redundancy_score,
    compute_lexical_diversity
)

def run_experiment_2() -> Dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 2: Comparison of Multi-Paradigm Summarization Approaches")
    print("=" * 70)

    # Load medium benchmark transcript
    items = TranscriptLoader.load_sample("medium_25min")
    full_text = TranscriptLoader.to_plain_text(items)

    data_path = Path(__file__).parent.parent / "data" / "sample_transcripts" / "medium_25min_lecture.json"
    with open(data_path, "r", encoding="utf-8") as f:
        meta = json.load(f)["metadata"]
    reference = meta["reference_summary"]

    approaches = [
        ("Direct TextRank (Single-Shot)", lambda t: TextRankSummarizer().summarize(t, num_sentences=6)),
        ("TF-IDF Salience", lambda t: TFIDFSalienceSummarizer().summarize(t, num_sentences=6)),
        ("Map-Reduce (512 tokens)", lambda t: MapReduceSummarizer(chunk_size=512, overlap_ratio=0.1).summarize(t, final_sentences=6)["summary"]),
        ("Iterative Refine", lambda t: RefineSummarizer(chunk_size=512).summarize(t, final_sentences=6)["summary"]),
        ("Hierarchical Tree", lambda t: HierarchicalTreeSummarizer(micro_tokens=512).summarize(t, final_sentences=6)["summary"])
    ]

    results = []

    for name, fn in approaches:
        t0 = time.perf_counter()
        summary = fn(full_text)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        rouge = compute_rouge(summary, reference)
        comp = compute_compression_ratio(summary, full_text)
        redundancy = compute_redundancy_score(summary)
        lex_div = compute_lexical_diversity(summary)

        record = {
            "approach": name,
            "latency_ms": elapsed_ms,
            "rouge_1": rouge["rouge-1"],
            "rouge_2": rouge["rouge-2"],
            "rouge_l": rouge["rouge-l"],
            "summary_words": comp["summary_words"],
            "reduction_percentage": comp["reduction_percentage"],
            "redundancy_score": redundancy,
            "lexical_diversity": lex_div,
            "summary": summary
        }
        results.append(record)

        print(f"{name:<30} | Latency: {elapsed_ms:6.1f}ms | R-1 F1: {record['rouge_1']['f1']:.4f} | "
              f"R-2 F1: {record['rouge_2']['f1']:.4f} | R-L F1: {record['rouge_l']['f1']:.4f} | Redundancy: {redundancy:.4f}")

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp2_summarization_approaches.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"experiment": "Experiment 2: Summarization Approaches", "results": results}, f, indent=2)

    print(f"\n[Saved results to {out_file}]")
    return {"experiment": 2, "results": results}

if __name__ == "__main__":
    run_experiment_2()
