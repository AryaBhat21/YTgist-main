import json
import time
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ytgist.data_loader import TranscriptLoader
from ytgist.summarizers import TextRankSummarizer
from ytgist.metrics import (
    compute_rouge,
    compute_compression_ratio,
    compute_redundancy_score,
    compute_lexical_diversity
)

def run_experiment_3() -> Dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 3: Summary Length vs Quality Trade-off & Pareto Frontier")
    print("=" * 70)

    items = TranscriptLoader.load_sample("medium_25min")
    full_text = TranscriptLoader.to_plain_text(items)

    data_path = Path(__file__).parent.parent / "data" / "sample_transcripts" / "medium_25min_lecture.json"
    with open(data_path, "r", encoding="utf-8") as f:
        meta = json.load(f)["metadata"]
    reference = meta["reference_summary"]

    sentence_budgets = [2, 4, 6, 8, 10, 15]
    summarizer = TextRankSummarizer()
    results = []

    for budget in sentence_budgets:
        t0 = time.perf_counter()
        summary = summarizer.summarize(full_text, num_sentences=budget)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        rouge = compute_rouge(summary, reference)
        comp = compute_compression_ratio(summary, full_text)
        redundancy = compute_redundancy_score(summary)
        lex_div = compute_lexical_diversity(summary)

        r1_p = rouge["rouge-1"]["precision"]
        r1_r = rouge["rouge-1"]["recall"]
        r1_f1 = rouge["rouge-1"]["f1"]
        rl_f1 = rouge["rouge-l"]["f1"]

        record = {
            "sentence_budget": budget,
            "word_count": comp["summary_words"],
            "compression_ratio": comp["compression_ratio"],
            "reduction_percentage": comp["reduction_percentage"],
            "rouge_1_precision": r1_p,
            "rouge_1_recall": r1_r,
            "rouge_1_f1": r1_f1,
            "rouge_l_f1": rl_f1,
            "redundancy_score": redundancy,
            "lexical_diversity": lex_div,
            "latency_ms": elapsed_ms,
            "summary": summary
        }
        results.append(record)

        print(f"Sentences: {budget:2d} | Words: {comp['summary_words']:3d} | Comp: {comp['compression_ratio']*100:4.1f}% | "
              f"R-1 Prec: {r1_p:.4f} | R-1 Rec: {r1_r:.4f} | R-1 F1: {r1_f1:.4f} | R-L: {rl_f1:.4f} | Redundancy: {redundancy:.4f}")

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp3_length_vs_quality.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"experiment": "Experiment 3: Length vs Quality", "results": results}, f, indent=2)

    print(f"\n[Saved results to {out_file}]")
    return {"experiment": 3, "results": results}

if __name__ == "__main__":
    run_experiment_3()
