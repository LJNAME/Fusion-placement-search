"""Exact makespan batching for fixed placement pools with identical partitions.

The scheduler recurrence is unchanged. Only the candidate dimension is
vectorized. This path intentionally omits memory/detail/critical-path output,
which the E2 acquisition and ranking metrics do not consume. Different fusion
partitions or nonzero interference use the original complete scheduler.
"""

from __future__ import annotations

import time

import numpy as np

from .e2_audit import PreparedCandidateRanker
from .search_sensitivity import _RankedPlan


class BatchedPlacementRanker(PreparedCandidateRanker):
    def __init__(self, graph, plans, n_devs, exchange, registry=None):
        super().__init__(graph, plans, n_devs, exchange, registry)
        started = time.perf_counter()
        self.identifiers = list(self.prepared)
        self.snapshots = list(self.prepared.values())
        first = self.snapshots[0]
        self.reference = first
        partition = tuple(group.node_ids for group in first.structure.groups)
        deps = tuple((dep.src_group, dep.dst_group, dep.bytes) for dep in first.structure.dependencies)
        self.batch_supported = all(
            tuple(group.node_ids for group in item.structure.groups) == partition and
            tuple((dep.src_group, dep.dst_group, dep.bytes) for dep in item.structure.dependencies) == deps
            for item in self.snapshots)
        if not self.batch_supported or not partition:
            self.batch_supported = False
            return
        self.devices = np.array([[group.device for group in item.structure.groups]
                                 for item in self.snapshots], dtype=np.intp).T
        self.columns = np.arange(len(self.snapshots))
        self.n_devs = n_devs
        self.group_variants = []
        for index in range(len(partition)):
            variants = {}
            for column, item in enumerate(self.snapshots):
                group = item.structure.groups[index]
                key = (group.device, group.pattern_id)
                if key not in variants:
                    variants[key] = (group, [])
                variants[key][1].append(column)
            self.group_variants.append([(group, np.array(columns, dtype=np.intp))
                                        for group, columns in variants.values()])
        self.preparation_wall_time_s += time.perf_counter() - started

    def rank(self, profile):
        if not self.batch_supported or profile.interference != 0.0 or profile.n_devs != self.n_devs:
            return super().rank(profile)
        started = time.perf_counter()
        count = len(self.snapshots)
        structure = self.reference.structure
        durations = np.empty((len(structure.groups), count), dtype=np.float64)
        confidence = np.ones(count, dtype=np.float64)
        for index, variants in enumerate(self.group_variants):
            for group, columns in variants:
                cost = profile.estimate_group(self.graph, group)
                durations[index, columns] = cost.duration
                confidence[columns] = np.minimum(confidence[columns], cost.confidence)
        ends = np.zeros_like(durations)
        device_free = np.zeros((self.n_devs, count), dtype=np.float64)
        transfer_ends = np.zeros((len(structure.dependencies), count), dtype=np.float64)
        link_free = {}
        transfer_cache = {}
        for group in structure.groups:
            index = group.index
            devices = self.devices[index]
            ready = device_free[devices, self.columns].copy()
            for dep_index in self.reference.incoming[index]:
                dep = structure.dependencies[dep_index]
                cross = self.devices[dep.src_group] != devices
                ready = np.maximum(ready, np.where(cross, transfer_ends[dep_index], ends[dep.src_group]))
            ends[index] = ready + durations[index]
            device_free[devices, self.columns] = ends[index]
            for dep_index in self.reference.outgoing[index]:
                dep = structure.dependencies[dep_index]
                targets = self.devices[dep.dst_group]
                # Each direction is disjoint within a dependency. Links may
                # share the same key (half-duplex), exactly as in the reference.
                for source in range(self.n_devs):
                    for target in range(self.n_devs):
                        if source == target:
                            continue
                        mask = (devices == source) & (targets == target)
                        if not np.any(mask):
                            continue
                        key = (source, target, dep.bytes)
                        if key not in transfer_cache:
                            transfer_cache[key] = profile.estimate_transfer(source, target, dep.bytes)
                        cost = transfer_cache[key]
                        link = profile.link_key(source, target)
                        if link not in link_free:
                            link_free[link] = np.zeros(count, dtype=np.float64)
                        completed = np.maximum(ends[index, mask], link_free[link][mask]) + cost.duration
                        transfer_ends[dep_index, mask] = completed
                        link_free[link][mask] = completed
                        confidence[mask] = np.minimum(confidence[mask], cost.confidence)
                        if not profile.overlap_send:
                            device_free[source, mask] = np.maximum(device_free[source, mask], completed)
        makespans = ends.max(axis=0)
        ranked = [_RankedPlan(identifier, item.plan, float(makespans[column]), (), float(confidence[column]))
                  for column, (identifier, item) in enumerate(zip(self.identifiers, self.snapshots))]
        ranked.sort(key=lambda item: (item.makespan, item.plan_id))
        self.calls += 1
        self.scoring_wall_time_s += time.perf_counter() - started
        return ranked
