"""
Master Runner for YTgist Experimental Benchmark Suite.
Executes all 5 experiments and outputs consolidated research results.
"""

import time
import json
from pathlib import Path

import importlib.util
from pathlib import Path

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

exp_dir = Path(__file__).parent / "experiments"
run_exp1 = load_module("exp1", exp_dir / "01_chunk_sizes.py").run_experiment_1
run_exp2 = load_module("exp2", exp_dir / "02_summarization_approaches.py").run_experiment_2
run_exp3 = load_module("exp3", exp_dir / "03_length_vs_quality.py").run_experiment_3
run_exp4 = load_module("exp4", exp_dir / "04_long_transcripts.py").run_experiment_4
run_exp5 = load_module("exp5", exp_dir / "05_error_analysis.py").run_experiment_5


def main():
    print("#" * 80)
    print("YTgist Research Benchmark Suite: Full Reproduction")
    print("Architecture: Video -> Transcript -> Preprocessing -> Chunking -> Summarization -> Extraction")
    print("#" * 80)

    start_total = time.perf_counter()
    summary_report = {}

    # Experiment 1: Chunk Sizes
    print("\n>>> Launching Experiment 1: Chunk Sizes & Overlap Ratios...")
    exp1_out = run_exp1()
    summary_report["exp1_chunk_sizes"] = exp1_out

    # Experiment 2: Summarization Paradigms
    print("\n>>> Launching Experiment 2: Summarization Approaches...")
    exp2_out = run_exp2()
    summary_report["exp2_summarization_approaches"] = exp2_out

    # Experiment 3: Length vs Quality
    print("\n>>> Launching Experiment 3: Length vs Quality & Pareto Frontier...")
    exp3_out = run_exp3()
    summary_report["exp3_length_vs_quality"] = exp3_out

    # Experiment 4: Long Transcripts
    print("\n>>> Launching Experiment 4: Long Transcript Scalability...")
    exp4_out = run_exp4()
    summary_report["exp4_long_transcripts"] = exp4_out

    # Experiment 5: Error Analysis
    print("\n>>> Launching Experiment 5: Error Analysis & Taxonomy...")
    exp5_out = run_exp5()
    summary_report["exp5_error_analysis"] = exp5_out

    total_time = round(time.perf_counter() - start_total, 2)
    print("\n" + "#" * 80)
    print(f"ALL 5 EXPERIMENTS COMPLETED SUCCESSFULLY in {total_time}s!")
    print("Consolidated reports saved to ./results/")
    print("#" * 80)

    out_file = Path(__file__).parent / "results" / "consolidated_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

if __name__ == "__main__":
    main()
