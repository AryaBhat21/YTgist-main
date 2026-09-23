# YTgist

> **Fast, lightweight YouTube video summarization and key moments extraction.**

YTgist transforms raw, noisy YouTube video transcripts into clean summaries, top key phrases, and timestamped key moments without needing heavy external models.

---

## ⚡ Quickstart

### 1. Installation

```bash
git clone https://github.com/AryaBhat21/YTgist-main.git
cd YTgist-main
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Run the Example Script (Easiest)

Run on the bundled sample transcript:
```bash
python example.py
```

Run on any public YouTube video:
```bash
python example.py --video "https://www.youtube.com/watch?v=VIDEO_ID"
```
*(or simply pass the 11-character video ID, e.g. `--video 6mgkoqcm6Sg`)*

Customize the summary sentence count:
```bash
python example.py --video short_5min --sentences 5
```

---

### Option 2: Run the Benchmark Experiments

Run all 5 experiments at once (finishes in ~1-2 seconds):
```bash
python run_all_experiments.py
```

Or run any single experiment file directly:
```bash
python experiments/01_chunk_sizes.py
python experiments/02_summarization_approaches.py
python experiments/03_length_vs_quality.py
python experiments/04_long_transcripts.py
python experiments/05_error_analysis.py
```
*Results are automatically saved to the `results/` folder.*

---

### Option 3: Run in Jupyter Notebook

Launch Jupyter and open either interactive notebook:
```bash
jupyter notebook
```
- **[YTgist.ipynb](YTgist.ipynb)**: End-to-end interactive YouTube summarization workflow.
- **[experiments/YTgist_Experiments.ipynb](experiments/YTgist_Experiments.ipynb)**: Visual benchmark suite with interactive Plotly charts.

---

### Option 4: Use in Python Code

```python
from ytgist import (
    TranscriptLoader,
    clean_transcript,
    restore_punctuation_heuristic,
    MapReduceSummarizer,
    KeyPhraseExtractor,
    TimelineExtractor
)

# 1. Load transcript (by YouTube video ID or offline sample)
items = TranscriptLoader.load_sample("short_5min")
# items = TranscriptLoader.load_from_youtube("YOUR_VIDEO_ID")

# 2. Clean and format transcript
cleaned = clean_transcript(items, remove_fillers=True)
punctuated_text = restore_punctuation_heuristic(TranscriptLoader.to_plain_text(cleaned))

# 3. Summarize
summarizer = MapReduceSummarizer(chunk_size=512, overlap_ratio=0.10)
summary = summarizer.summarize(punctuated_text, final_sentences=4)["summary"]
print("SUMMARY:\n", summary)

# 4. Extract Key Phrases
phrases = KeyPhraseExtractor(max_phrases=5).extract(punctuated_text)
print("\nKEY PHRASES:", phrases)

# 5. Extract Timeline with Timestamps
timeline = TimelineExtractor(top_k=4).extract_from_transcript(cleaned)
for moment in timeline:
    print(f"[{moment.timestamp_str}] {moment.text}")
```

---

## 🛠️ Features

- 🧹 **ASR Denoising**: Strips filler words, auditory tags (`[Music]`, `[Applause]`), and restores sentence punctuation.
- ✂️ **Smart Chunking**: Token and sentence-boundary chunking that prevents mid-sentence truncation.
- 📝 **Multi-Paradigm Summarization**: Map-Reduce, TextRank, TF-IDF Salience, Refine, and Hierarchical Tree summarizers.
- ⏱️ **Timestamp Alignment**: Automatically extracts key moments with exact video timestamps (`MM:SS`).
- ⚡ **Blazing Fast**: Runs 100% locally on CPU in milliseconds—no GPU required.

---

## 📁 Project Structure

```text
ytgist/
├── ytgist/                  # Core Python library
│   ├── data_loader.py       # Fetch from YouTube API or local JSON
│   ├── preprocessing.py     # Clean transcripts & restore punctuation
│   ├── chunking.py          # Token & sentence boundary chunkers
│   ├── summarizers.py       # Map-Reduce, TextRank, and TF-IDF engines
│   ├── extraction.py        # Key phrase & timeline moment extraction
│   └── metrics.py           # ROUGE, compression ratio, redundancy metrics
├── experiments/             # Benchmark experiment scripts (01 to 05)
├── data/sample_transcripts/ # Offline benchmark transcripts (5m, 25m, 65m)
├── results/                 # Pre-computed benchmark results
├── example.py               # Simple quickstart CLI script
├── run_all_experiments.py   # Runs all 5 benchmark experiments
└── YTgist.ipynb             # Interactive notebook
```

---

## 📄 License

MIT License.
