import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ytgist.data_loader import TranscriptLoader
from ytgist.chunking import TokenChunker, FixedCharChunker, SentenceBoundaryChunker
from ytgist.summarizers import MapReduceSummarizer, TextRankSummarizer
from ytgist.metrics import compute_rouge, compute_redundancy_score, compute_compression_ratio

def check_boundary_cutoffs(chunks: list) -> float:
    """Calculates percentage of chunks whose ends do not terminate with a sentence boundary (. ! ?)."""
    if not chunks:
        return 0.0
    cutoffs = sum(1 for c in chunks if c.text.strip() and c.text.strip()[-1] not in ".!?")
    return round((cutoffs / len(chunks)) * 100, 2)

def run_experiment_1() -> Dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 1: Impact of Chunk Sizes & Overlap Ratios on Summary Quality")
    print("=" * 70)

    # Load medium benchmark transcript
    items = TranscriptLoader.load_sample("medium_25min")
    full_text = TranscriptLoader.to_plain_text(items)
    
    # Load reference summary
    data_path = Path(__file__).parent.parent / "data" / "sample_transcripts" / "medium_25min_lecture.json"
    with open(data_path, "r", encoding="utf-8") as f:
        meta = json.load(f)["metadata"]
    reference = meta["reference_summary"]

    chunk_sizes = [256, 512, 1024, 2048]
    overlaps = [0.0, 0.10, 0.20]

    results = []

    for size in chunk_sizes:
        for ov in overlaps:
            t0 = time.perf_counter()
            chunker = TokenChunker(target_tokens=size, overlap_ratio=ov)
            chunks = chunker.chunk_text(full_text)
            
            # Map-Reduce summarization using this chunk size
            summarizer = MapReduceSummarizer(chunk_size=size, overlap_ratio=ov)
            summary_res = summarizer.summarize(full_text, final_sentences=6)
            summary = summary_res["summary"]
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            rouge = compute_rouge(summary, reference)
            redundancy = compute_redundancy_score(summary)
            cutoffs = check_boundary_cutoffs(chunks)

            record = {
                "chunk_size_tokens": size,
                "overlap_ratio": ov,
                "num_chunks": len(chunks),
                "avg_words_per_chunk": round(sum(len(c.text.split()) for c in chunks) / len(chunks), 1) if chunks else 0,
                "boundary_cutoffs_pct": cutoffs,
                "latency_ms": elapsed_ms,
                "rouge_1_f1": rouge["rouge-1"]["f1"],
                "rouge_2_f1": rouge["rouge-2"]["f1"],
                "rouge_l_f1": rouge["rouge-l"]["f1"],
                "redundancy_score": redundancy,
                "summary": summary
            }
            results.append(record)

            print(f"Size: {size:4d} | Overlap: {int(ov*100):2d}% | Chunks: {len(chunks):2d} | "
                  f"R-1: {record['rouge_1_f1']:.4f} | R-L: {record['rouge_l_f1']:.4f} | "
                  f"Cutoffs: {cutoffs:5.1f}% | Time: {elapsed_ms:6.1f}ms")

    # Save results
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp1_chunk_sizes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"experiment": "Experiment 1: Chunk Sizes", "results": results}, f, indent=2)

    print(f"\n[Saved results to {out_file}]")
    return {"experiment": 1, "results": results}

if __name__ == "__main__":
    run_experiment_1()
