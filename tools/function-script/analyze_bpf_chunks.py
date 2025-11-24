#!/usr/bin/env python3
"""
Analyze BPF hooks trace to understand chunk behavior patterns

This script analyzes how chunks move through different BPF hooks:
- Which chunks get activated?
- Which chunks get populated?
- Which chunks never get depopulated?
- What are the typical hook sequences for each chunk?
"""

import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class ChunkLifecycle:
    """Track a chunk's journey through BPF hooks"""
    addr: str
    hooks: List[Tuple[int, str]] = field(default_factory=list)  # (timestamp, hook_type)
    first_seen: int = 0
    last_seen: int = 0

    def add_hook(self, timestamp: int, hook_type: str):
        if not self.hooks:
            self.first_seen = timestamp
        self.hooks.append((timestamp, hook_type))
        self.last_seen = timestamp

    def get_hook_sequence(self, max_len: int = None) -> List[str]:
        """Get sequence of hook types"""
        hooks = self.hooks if max_len is None else self.hooks[:max_len]
        return [h[1] for h in hooks]

    def get_hook_counts(self) -> Counter:
        """Count occurrences of each hook type"""
        return Counter(h[1] for h in self.hooks)

    def duration_ms(self) -> int:
        """Total time chunk was tracked (ms)"""
        return self.last_seen - self.first_seen if self.hooks else 0

def parse_bpf_trace(filename: str) -> Dict[str, ChunkLifecycle]:
    """Parse BPF hooks trace log"""
    chunks = {}

    # Pattern: TIME(ms)   HOOK_TYPE   CHUNK_ADDR   LIST_ADDR
    hook_pattern = re.compile(r'^\s*(\d+)\s+(ACTIVATE|POPULATE|DEPOPULATE|EVICTION_PREPARE)\s+(0x[0-9a-f]+|---)')

    with open(filename, 'r') as f:
        for line in f:
            match = hook_pattern.match(line)
            if not match:
                continue

            timestamp = int(match.group(1))
            hook_type = match.group(2)
            chunk_addr = match.group(3)

            # Skip eviction_prepare (no specific chunk)
            if chunk_addr == '---' or hook_type == 'EVICTION_PREPARE':
                continue

            if chunk_addr not in chunks:
                chunks[chunk_addr] = ChunkLifecycle(chunk_addr)

            chunks[chunk_addr].add_hook(timestamp, hook_type)

    return chunks

def analyze_hook_patterns(chunks: Dict[str, ChunkLifecycle]):
    """Analyze common hook sequences"""
    print("="*80)
    print("HOOK SEQUENCE PATTERNS")
    print("="*80)

    # Analyze first N hooks for each chunk
    for n in [3, 5]:
        print(f"\nTop 10 {n}-Hook Sequences:")
        print(f"{'Sequence':<60s} {'Count':>10s}")
        print("-" * 71)

        patterns = Counter()
        for chunk in chunks.values():
            if len(chunk.hooks) >= n:
                seq = ' → '.join(chunk.get_hook_sequence(n))
                patterns[seq] += 1

        for seq, count in patterns.most_common(10):
            print(f"{seq:<60s} {count:>10,d}")

def analyze_hook_statistics(chunks: Dict[str, ChunkLifecycle]):
    """Analyze hook call statistics per chunk"""
    print("\n" + "="*80)
    print("PER-CHUNK HOOK STATISTICS")
    print("="*80)

    # Count hooks per chunk
    activate_counts = []
    populate_counts = []
    depopulate_counts = []

    for chunk in chunks.values():
        counts = chunk.get_hook_counts()
        activate_counts.append(counts.get('ACTIVATE', 0))
        populate_counts.append(counts.get('POPULATE', 0))
        depopulate_counts.append(counts.get('DEPOPULATE', 0))

    print(f"\nTotal unique chunks tracked: {len(chunks):,}")

    print(f"\nACTIVATE hook per chunk:")
    print(f"  Min:     {min(activate_counts) if activate_counts else 0}")
    print(f"  Max:     {max(activate_counts) if activate_counts else 0}")
    print(f"  Average: {sum(activate_counts)/len(activate_counts) if activate_counts else 0:.2f}")
    print(f"  Chunks with ACTIVATE: {sum(1 for c in activate_counts if c > 0):,} ({sum(1 for c in activate_counts if c > 0)*100/len(chunks):.1f}%)")

    print(f"\nPOPULATE hook per chunk:")
    print(f"  Min:     {min(populate_counts) if populate_counts else 0}")
    print(f"  Max:     {max(populate_counts) if populate_counts else 0}")
    print(f"  Average: {sum(populate_counts)/len(populate_counts) if populate_counts else 0:.2f}")
    print(f"  Chunks with POPULATE: {sum(1 for c in populate_counts if c > 0):,} ({sum(1 for c in populate_counts if c > 0)*100/len(chunks):.1f}%)")

    print(f"\nDEPOPULATE hook per chunk:")
    print(f"  Min:     {min(depopulate_counts) if depopulate_counts else 0}")
    print(f"  Max:     {max(depopulate_counts) if depopulate_counts else 0}")
    print(f"  Average: {sum(depopulate_counts)/len(depopulate_counts) if depopulate_counts else 0:.2f}")
    print(f"  Chunks with DEPOPULATE: {sum(1 for c in depopulate_counts if c > 0):,} ({sum(1 for c in depopulate_counts if c > 0)*100/len(chunks):.1f}%)")

def analyze_chunk_categories(chunks: Dict[str, ChunkLifecycle]):
    """Categorize chunks by behavior"""
    print("\n" + "="*80)
    print("CHUNK BEHAVIOR CATEGORIES")
    print("="*80)

    # Categorize
    only_activate = []
    only_populate = []
    activate_and_populate = []
    populate_only = []
    multi_activate = []
    multi_populate = []
    no_depopulate = []

    for chunk in chunks.values():
        counts = chunk.get_hook_counts()
        has_activate = counts.get('ACTIVATE', 0) > 0
        has_populate = counts.get('POPULATE', 0) > 0
        has_depopulate = counts.get('DEPOPULATE', 0) > 0

        if has_activate and not has_populate:
            only_activate.append(chunk)
        elif has_populate and not has_activate:
            only_populate.append(chunk)
        elif has_activate and has_populate:
            activate_and_populate.append(chunk)

        if counts.get('ACTIVATE', 0) > 1:
            multi_activate.append(chunk)

        if counts.get('POPULATE', 0) > 1:
            multi_populate.append(chunk)

        if not has_depopulate:
            no_depopulate.append(chunk)

    total = len(chunks)

    print(f"\nChunk categories:")
    print(f"  Only ACTIVATE (no populate):       {len(only_activate):>6,} ({len(only_activate)*100/total:>5.1f}%)")
    print(f"  Only POPULATE (no activate):       {len(only_populate):>6,} ({len(only_populate)*100/total:>5.1f}%)")
    print(f"  Both ACTIVATE and POPULATE:        {len(activate_and_populate):>6,} ({len(activate_and_populate)*100/total:>5.1f}%)")
    print(f"  Multiple ACTIVATE (>1):            {len(multi_activate):>6,} ({len(multi_activate)*100/total:>5.1f}%)")
    print(f"  Multiple POPULATE (>1):            {len(multi_populate):>6,} ({len(multi_populate)*100/total:>5.1f}%)")
    print(f"  No DEPOPULATE:                     {len(no_depopulate):>6,} ({len(no_depopulate)*100/total:>5.1f}%)")

def analyze_first_hook(chunks: Dict[str, ChunkLifecycle]):
    """Analyze what is the first hook for each chunk"""
    print("\n" + "="*80)
    print("FIRST HOOK ANALYSIS")
    print("="*80)

    first_hooks = Counter()
    for chunk in chunks.values():
        if chunk.hooks:
            first_hooks[chunk.hooks[0][1]] += 1

    print(f"\nFirst hook seen for chunks:")
    print(f"{'Hook':<20s} {'Count':>10s} {'Percentage':>12s}")
    print("-" * 43)
    for hook, count in first_hooks.most_common():
        percentage = (count * 100 / len(chunks))
        print(f"{hook:<20s} {count:>10,d} {percentage:>11.2f}%")

def analyze_populate_without_activate(chunks: Dict[str, ChunkLifecycle]):
    """Analyze chunks that get populated without being activated"""
    print("\n" + "="*80)
    print("POPULATE WITHOUT ACTIVATE ANALYSIS")
    print("="*80)

    pop_no_act = []
    for chunk in chunks.values():
        counts = chunk.get_hook_counts()
        if counts.get('POPULATE', 0) > 0 and counts.get('ACTIVATE', 0) == 0:
            pop_no_act.append(chunk)

    print(f"\nChunks with POPULATE but no ACTIVATE: {len(pop_no_act):,} ({len(pop_no_act)*100/len(chunks):.1f}%)")

    if pop_no_act:
        print(f"\nSample 10 chunks:")
        print(f"{'Chunk Address':<20s} {'Populates':>10s} {'First 5 Hooks':<50s}")
        print("-" * 81)
        for chunk in pop_no_act[:10]:
            seq = ' → '.join(chunk.get_hook_sequence(5))
            counts = chunk.get_hook_counts()
            print(f"{chunk.addr:<20s} {counts['POPULATE']:>10d} {seq:<50s}")

def analyze_thrashing_pattern(chunks: Dict[str, ChunkLifecycle]):
    """Analyze thrashing: rapid activate → populate cycles"""
    print("\n" + "="*80)
    print("THRASHING PATTERN ANALYSIS")
    print("="*80)

    thrashing = []
    for chunk in chunks.values():
        # Look for pattern: ACTIVATE → POPULATE → ACTIVATE → POPULATE
        seq = chunk.get_hook_sequence()
        if len(seq) < 4:
            continue

        # Count activate→populate transitions
        transitions = 0
        for i in range(len(seq) - 1):
            if seq[i] == 'ACTIVATE' and seq[i+1] == 'POPULATE':
                transitions += 1
            elif seq[i] == 'POPULATE' and i+1 < len(seq) and seq[i+1] == 'ACTIVATE':
                transitions += 1

        if transitions >= 2:
            thrashing.append((chunk, transitions))

    thrashing.sort(key=lambda x: x[1], reverse=True)

    print(f"\nChunks with thrashing pattern: {len(thrashing):,} ({len(thrashing)*100/len(chunks):.1f}%)")
    print(f"  (Chunks with ≥2 activate↔populate transitions)")

    if thrashing:
        print(f"\nTop 10 thrashing chunks:")
        print(f"{'Chunk Address':<20s} {'Transitions':>12s} {'Total Hooks':>12s} {'First 8 Hooks':<60s}")
        print("-" * 105)
        for chunk, trans_count in thrashing[:10]:
            seq = ' → '.join(chunk.get_hook_sequence(8))
            print(f"{chunk.addr:<20s} {trans_count:>12d} {len(chunk.hooks):>12d} {seq:<60s}")

def analyze_chunk_lifetime(chunks: Dict[str, ChunkLifecycle]):
    """Analyze chunk lifetime distribution"""
    print("\n" + "="*80)
    print("CHUNK LIFETIME ANALYSIS")
    print("="*80)

    durations = [chunk.duration_ms() for chunk in chunks.values()]
    durations.sort()

    print(f"\nLifetime statistics (first hook to last hook):")
    print(f"  Min:     {min(durations) if durations else 0:,}ms")
    print(f"  Max:     {max(durations) if durations else 0:,}ms")
    print(f"  Average: {sum(durations)/len(durations) if durations else 0:,.1f}ms")
    print(f"  Median:  {durations[len(durations)//2] if durations else 0:,}ms")

    # Distribution
    buckets = {
        '0ms (instant)': 0,
        '1-10ms': 0,
        '10-100ms': 0,
        '100ms-1s': 0,
        '1s-5s': 0,
        '>5s': 0,
    }

    for d in durations:
        if d == 0:
            buckets['0ms (instant)'] += 1
        elif d < 10:
            buckets['1-10ms'] += 1
        elif d < 100:
            buckets['10-100ms'] += 1
        elif d < 1000:
            buckets['100ms-1s'] += 1
        elif d < 5000:
            buckets['1s-5s'] += 1
        else:
            buckets['>5s'] += 1

    print(f"\nLifetime distribution:")
    for bucket, count in buckets.items():
        pct = count * 100 / len(durations) if durations else 0
        print(f"  {bucket:<15s}: {count:>6,d} ({pct:>5.1f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_bpf_chunks.py <trace_log_file>")
        sys.exit(1)

    filename = sys.argv[1]
    print(f"Parsing {filename}...")

    chunks = parse_bpf_trace(filename)

    print(f"\n{'='*80}")
    print(f"Parsed {len(chunks):,} unique chunks")
    total_hooks = sum(len(c.hooks) for c in chunks.values())
    print(f"Total hook events: {total_hooks:,}")
    print(f"{'='*80}")

    # Run all analyses
    analyze_first_hook(chunks)
    analyze_hook_statistics(chunks)
    analyze_chunk_categories(chunks)
    analyze_hook_patterns(chunks)
    analyze_populate_without_activate(chunks)
    analyze_thrashing_pattern(chunks)
    analyze_chunk_lifetime(chunks)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
