import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "sample_transcripts"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Medium 25-minute Lecture (~3,800 words, ~1500 seconds)
medium_paragraphs = [
    ("00:00", "Welcome everyone to CS 244 Distributed Systems Engineering. Today we cover the Raft consensus algorithm."),
    ("01:15", "Consensus in distributed systems guarantees that multiple nodes agree on a shared sequence of state machine transitions despite network delays, packet drops, or crash failures."),
    ("02:40", "Historically, Paxos was the gold standard for distributed consensus. However, Paxos is notoriously difficult to understand and implement correctly in production environments."),
    ("04:10", "Ongaro and Ousterhout at Stanford developed Raft with an explicit design goal: understandability without sacrificing formal safety or performance."),
    ("05:35", "Raft achieves consensus by decomposing the consensus problem into three relatively independent subproblems: leader election, log replication, and safety enforcement."),
    ("07:00", "Let us examine the node states. At any given moment, a Raft node resides in one of three states: Leader, Follower, or Candidate."),
    ("08:25", "Followers remain completely passive; they do not issue requests independently but simply respond to RPCs from Candidates and Leaders."),
    ("09:50", "If a Follower hears no communication from a valid Leader within a randomized election timeout, it assumes the Leader has failed and transitions to the Candidate state."),
    ("11:15", "The Candidate increments the current term counter, votes for itself, and broadcasts RequestVote RPCs to all other cluster members."),
    ("12:40", "Randomized election timeouts, typically between 150 milliseconds and 300 milliseconds, are essential to prevent split votes where multiple candidates split the cluster evenly."),
    ("14:05", "Once a Candidate receives votes from a strict majority of nodes, it transitions to the Leader state and immediately begins sending periodic empty AppendEntries heartbeats."),
    ("15:30", "Now let us turn to log replication. When a client issues a state machine command, it submits it directly to the cluster Leader."),
    ("16:55", "The Leader appends the command to its local log and issues AppendEntries RPCs containing the log entry to all Follower nodes."),
    ("18:20", "When a majority of Followers successfully write the log entry to stable disk storage and acknowledge back, the Leader commits the entry."),
    ("19:45", "Once committed, the Leader applies the entry to its local state machine and returns the execution result to the client."),
    ("21:10", "Safety is paramount in Raft. The Election Safety property ensures that at most one leader can be elected per term."),
    ("22:35", "Furthermore, the Leader Completeness property states that if a log entry is committed in a given term, that entry will be present in the logs of leaders for all higher terms."),
    ("24:00", "In conclusion, Raft simplifies consensus through strong leadership, randomized election timeouts, and rigid log matching invariants, powering modern systems like etcd, Consul, and CockroachDB.")
]

def expand_transcript(sections, target_words):
    items = []
    current_sec = 0.0
    ref_summary = " ".join([p[1] for p in sections[:6]])
    
    # We will expand sections with realistic spoken sentences until target word count
    sec_idx = 0
    while sum(len(it["text"].split()) for it in items) < target_words:
        ts_str, base_text = sections[sec_idx % len(sections)]
        sec_idx += 1
        
        # Add realistic conversational phrases
        variants = [
            f"As we discussed earlier, {base_text.lower()}",
            f"Let's look more closely at this specific property: {base_text}",
            f"In practical engineering deployments, {base_text.lower()} This is critical for high availability.",
            f"If we monitor the cluster metrics here, {base_text.lower()}",
            f"[Applause] Notice how the failover behaves smoothly. {base_text}"
        ]
        chosen = variants[sec_idx % len(variants)]
        duration = max(3.0, len(chosen.split()) * 0.4)
        items.append({
            "start": round(current_sec, 2),
            "duration": round(duration, 2),
            "text": chosen
        })
        current_sec += duration + 0.3

    return items

medium_items = expand_transcript(medium_paragraphs, 3800)
medium_data = {
    "metadata": {
        "title": "CS244: Distributed Consensus and the Raft Protocol",
        "video_id": "raft_lecture_25m",
        "duration_sec": round(medium_items[-1]["start"] + medium_items[-1]["duration"], 1),
        "total_words": sum(len(x["text"].split()) for x in medium_items),
        "domain": "Distributed Systems",
        "reference_summary": "Raft is a distributed consensus algorithm designed for understandability as an alternative to Paxos. It decomposes consensus into leader election, log replication, and safety. Nodes cycle between Follower, Candidate, and Leader states using randomized election timeouts to avoid split votes. Leaders append client commands to logs, replicate them via AppendEntries RPCs across a majority of nodes, and apply them to local state machines. Strong safety invariants guarantee leader completeness and state machine consistency across production clusters."
    },
    "transcript": medium_items
}

with open(DATA_DIR / "medium_25min_lecture.json", "w", encoding="utf-8") as f:
    json.dump(medium_data, f, indent=2)

# 2. Long 65-minute Conference Keynote (~10,500 words)
long_paragraphs = [
    ("00:00", "Good morning and welcome to the Global AI Systems Summit 2026 keynote on Scalable Foundation Model Infrastructure."),
    ("03:15", "Over the past three years, foundation models have expanded from tens of billions of parameters to trillions of multimodal tokens."),
    ("06:30", "Training at this planetary scale requires co-designing hardware accelerators, optical interconnects, distributed compilers, and fault-tolerant storage."),
    ("09:45", "Let us first address 3D parallelism: combining Tensor Parallelism across NVLink, Pipeline Parallelism across InfiniBand switches, and ZeRO-style Data Parallelism across the cluster."),
    ("13:00", "Tensor parallelism splits linear matrix multiplications across GPUs within a single NVLink node to minimize communication latency."),
    ("16:15", "Pipeline parallelism partitions transformer layers sequentially across multiple nodes, utilizing 1F1B scheduling to minimize execution bubble overhead."),
    ("19:30", "However, as clusters scale beyond 16,000 accelerators, the Mean Time Between Failures drops dramatically to under four hours."),
    ("22:45", "Traditional synchronous checkpointing to distributed object stores stalls training for twenty minutes per crash, decimating Effective Floating Point Operations per Second."),
    ("26:00", "To conquer this barrier, we introduced asynchronous non-blocking memory snapshots coupled with in-network topology-aware consensus recovery."),
    ("29:15", "On the inference side, serving long-context transcripts and reasoning chains introduces massive KV-cache memory pressure."),
    ("32:30", "PagedAttention and chunked prefill techniques enable dynamic non-contiguous allocation of memory blocks, eliminating memory fragmentation."),
    ("35:45", "Furthermore, speculative decoding with compact draft models accelerates autoregressive token generation by up to three times without degrading generation fidelity."),
    ("39:00", "Quantization strategies such as FP8 and 4-bit weight activations allow deployment of massive mixture-of-experts models on edge and sovereign server environments."),
    ("42:15", "We must also consider the ecological footprint: data centers now consume gigawatts, necessitating carbon-aware scheduling and liquid immersion cooling."),
    ("45:30", "Algorithmic advancements in linear attention, state-space models like Mamba, and hybrid architectures show promising asymptotic scaling beyond quadratic attention."),
    ("48:45", "Our empirical evaluations across a 32,000 GPU cluster demonstrated 64% Model FLOPs Utilization over sustained multi-month training campaigns."),
    ("52:00", "In conclusion, the frontier of AI intelligence is governed not merely by dataset volume, but by our collective ability to orchestrate resilient, distributed computing fabrics at scale.")
]

long_items = expand_transcript(long_paragraphs, 10500)
long_data = {
    "metadata": {
        "title": "Scaling Foundation Model Systems: Hardware, Networking, and Resilience",
        "video_id": "scaling_keynote_65m",
        "duration_sec": round(long_items[-1]["start"] + long_items[-1]["duration"], 1),
        "total_words": sum(len(x["text"].split()) for x in long_items),
        "domain": "AI Systems & Infrastructure",
        "reference_summary": "The 2026 AI Systems keynote addresses infrastructure challenges for training trillion-parameter foundation models across 32,000 accelerators. Key solutions include 3D parallelism (Tensor, Pipeline, and ZeRO Data Parallelism), non-blocking asynchronous checkpointing to combat low MTBF, and PagedAttention with speculative decoding for inference. Algorithmic innovations like state-space models and FP8 quantization improve efficiency while achieving 64% sustained Model FLOPs Utilization."
    },
    "transcript": long_items
}

with open(DATA_DIR / "long_65min_conference.json", "w", encoding="utf-8") as f:
    json.dump(long_data, f, indent=2)

print("Generated sample transcripts successfully:")
print(f"- Short: {len(medium_items)} items")
print(f"- Medium: {len(medium_items)} items, {medium_data['metadata']['total_words']} words")
print(f"- Long: {len(long_items)} items, {long_data['metadata']['total_words']} words")
