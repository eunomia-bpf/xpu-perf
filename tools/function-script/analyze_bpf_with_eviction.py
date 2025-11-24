#!/usr/bin/env python3
"""
Analyze BPF hooks trace including EVICTION_PREPARE events

This script analyzes:
- Per-chunk hook sequences (ACTIVATE, POPULATE, DEPOPULATE)
- Global EVICTION_PREPARE events and their timing
- Relationship between eviction and chunk state changes
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

@dataclass
class EvictionEvent:
    """Track an eviction_prepare event"""
    timestamp: int
    used_list: str
    unused_list: str

def parse_bpf_trace(filename: str) -> Tuple[Dict[str, ChunkLifecycle], List[EvictionEvent]]:
    """Parse BPF hooks trace log"""
    chunks = {}
    eviction_events = []

    # Pattern: TIME(ms)   HOOK_TYPE   CHUNK_ADDR   LIST_ADDR
    hook_pattern = re.compile(r'^\s*(\d+)\s+(ACTIVATE|POPULATE|DEPOPULATE)\s+(0x[0-9a-f]+)')
    eviction_pattern = re.compile(r'^\s*(\d+)\s+EVICTION_PREPARE\s+---\s+used=(0x[0-9a-f]+)\s+unused=(0x[0-9a-f]+)')

    with open(filename, 'r') as f:
        for line in f:
            # Try to match chunk hooks
            match = hook_pattern.match(line)
            if match:
                timestamp = int(match.group(1))
                hook_type = match.group(2)
                chunk_addr = match.group(3)

                if chunk_addr not in chunks:
                    chunks[chunk_addr] = ChunkLifecycle(chunk_addr)

                chunks[chunk_addr].add_hook(timestamp, hook_type)
                continue

            # Try to match eviction_prepare
            match = eviction_pattern.match(line)
            if match:
                timestamp = int(match.group(1))
                used_list = match.group(2)
                unused_list = match.group(3)
                eviction_events.append(EvictionEvent(timestamp, used_list, unused_list))
                continue

    return chunks, eviction_events

def analyze_eviction_timing(chunks: Dict[str, ChunkLifecycle], evictions: List[EvictionEvent]):
    """Analyze timing relationship between eviction and chunk events"""
    print("="*80)
    print("EVICTION TIMING ANALYSIS")
    print("="*80)

    print(f"\nTotal EVICTION_PREPARE events: {len(evictions):,}")

    if not evictions:
        print("No eviction events found!")
        return

    # Find events around evictions (within 1ms window)
    events_before_eviction = Counter()
    events_after_eviction = Counter()

    for eviction in evictions[:1000]:  # Sample first 1000 evictions
        evict_time = eviction.timestamp

        # Check what happened within 1ms before eviction
        for chunk in chunks.values():
            for ts, hook_type in chunk.hooks:
                if evict_time - 1 <= ts < evict_time:
                    events_before_eviction[hook_type] += 1
                elif evict_time < ts <= evict_time + 1:
                    events_after_eviction[hook_type] += 1

    print("\nEvents within 1ms BEFORE eviction (sampled from first 1000 evictions):")
    print(f"{'Hook':<20s} {'Count':>10s}")
    print("-" * 31)
    for hook, count in events_before_eviction.most_common():
        print(f"{hook:<20s} {count:>10,d}")

    print("\nEvents within 1ms AFTER eviction:")
    print(f"{'Hook':<20s} {'Count':>10s}")
    print("-" * 31)
    for hook, count in events_after_eviction.most_common():
        print(f"{hook:<20s} {count:>10,d}")

    # Analyze eviction frequency over time
    if len(evictions) > 1:
        intervals = []
        for i in range(1, min(len(evictions), 1000)):
            interval = evictions[i].timestamp - evictions[i-1].timestamp
            intervals.append(interval)

        if intervals:
            print(f"\nEviction interval statistics (first 1000 evictions):")
            print(f"  Min interval:     {min(intervals)}ms")
            print(f"  Max interval:     {max(intervals)}ms")
            print(f"  Average interval: {sum(intervals)/len(intervals):.2f}ms")
            print(f"  Median interval:  {sorted(intervals)[len(intervals)//2]}ms")

def analyze_hook_patterns(chunks: Dict[str, ChunkLifecycle]):
    """Analyze common hook sequences"""
    print("\n" + "="*80)
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_bpf_with_eviction.py <trace_log_file>")
        sys.exit(1)

    filename = sys.argv[1]
    print(f"Parsing {filename}...")

    chunks, evictions = parse_bpf_trace(filename)

    print(f"\n{'='*80}")
    print(f"Parsed {len(chunks):,} unique chunks")
    print(f"Parsed {len(evictions):,} eviction events")
    total_hooks = sum(len(c.hooks) for c in chunks.values())
    print(f"Total chunk hook events: {total_hooks:,}")
    print(f"{'='*80}")

    # Run all analyses
    analyze_eviction_timing(chunks, evictions)
    analyze_first_hook(chunks)
    analyze_hook_statistics(chunks)
    analyze_hook_patterns(chunks)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
