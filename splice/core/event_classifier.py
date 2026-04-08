"""
Module 10: core/event_classifier.py

Classify splicing modules into classical alternative splicing event types
(SE, A3SS, A5SS, MXE, RI, TandemCassette, AFE, ALE, Complex).
Integrates co-occurrence evidence for better classification.
"""

from __future__ import annotations

from typing import List, Optional, Set

from splicekit.core.cooccurrence import CooccurrenceGraph
from splicekit.core.splicegraph import SplicingModule
from splicekit.utils.genomic import Junction


def classify_event(
    module: SplicingModule,
    cooccurrence_graph: Optional[CooccurrenceGraph] = None,
) -> str:
    """Classify a splicing module into an event type.

    Classifies based on junction topology and optional co-occurrence patterns.

    Rules (applied in order):
    - SE (Skipped Exon): 3 junctions with exon-skip topology
    - A3SS (Alt 3' site): 2 junctions sharing donor, different acceptors
    - A5SS (Alt 5' site): 2 junctions sharing acceptor, different donors
    - MXE (Mutually Exclusive): 4 junctions with 2 mutually exclusive paths
    - RI (Retained Intron): 2 junctions, one subset of other
    - TandemCassette: Multiple exons coordinated via co-occurrence
    - Complex: >4 junctions or other patterns

    Args:
        module: SplicingModule to classify.
        cooccurrence_graph: Optional co-occurrence graph for mutual exclusivity detection.

    Returns:
        Event type string: "SE", "A3SS", "A5SS", "MXE", "RI", "TandemCassette", "Complex"
    """
    n = len(module.junctions)

    # Complex: too many junctions
    if n > 4:
        return "Complex"

    if n == 2:
        j1, j2 = module.junctions[0], module.junctions[1]

        # Check for A3SS: same donor, different acceptors
        if j1.donor == j2.donor and j1.acceptor != j2.acceptor:
            return "A3SS"

        # Check for A5SS: same acceptor, different donors
        if j1.acceptor == j2.acceptor and j1.donor != j2.donor:
            return "A5SS"

        # Check for RI: one intron contains the other
        # j1 is contained in j2 if j2.start <= j1.start < j1.end <= j2.end
        if j2.start <= j1.start and j1.end <= j2.end and j1.start < j1.end:
            # Need to verify at least one junction spans the region
            return "RI"
        if j1.start <= j2.start and j2.end <= j1.end and j2.start < j2.end:
            return "RI"

        return "Complex"

    elif n == 3:
        # Check for SE: skipped exon topology
        event_type = _classify_se_or_complex(module)
        if event_type == "SE":
            return "SE"

        return "Complex"

    elif n == 4:
        # Check for MXE using co-occurrence if available
        if cooccurrence_graph is not None:
            if _is_mxe_by_cooccurrence(cooccurrence_graph):
                return "MXE"

        # Check for tandem cassette (coordinated junctions)
        if cooccurrence_graph is not None:
            if len(module.coordinated_junctions) >= 2:
                return "TandemCassette"

        return "Complex"

    return "Complex"


def _classify_se_or_complex(module: SplicingModule) -> str:
    """Classify 3-junction modules as SE or Complex.

    SE topology: two junctions with same donor (or acceptor), third junction
    shares an endpoint with one of them.
    """
    j1, j2, j3 = module.junctions[0], module.junctions[1], module.junctions[2]

    # Check if any pair shares donor
    if j1.donor == j2.donor and j3.donor != j1.donor:
        return "SE"
    if j1.donor == j3.donor and j2.donor != j1.donor:
        return "SE"
    if j2.donor == j3.donor and j1.donor != j2.donor:
        return "SE"

    # Check if any pair shares acceptor
    if j1.acceptor == j2.acceptor and j3.acceptor != j1.acceptor:
        return "SE"
    if j1.acceptor == j3.acceptor and j2.acceptor != j1.acceptor:
        return "SE"
    if j2.acceptor == j3.acceptor and j1.acceptor != j2.acceptor:
        return "SE"

    return "Complex"


def _is_mxe_by_cooccurrence(graph: CooccurrenceGraph) -> bool:
    """Check if module is MXE based on mutually exclusive paths in co-occurrence graph.

    MXE confirmed if two junction sets never appear in the same read (zero co-occurrence).
    """
    if not graph.mutually_exclusive_paths:
        return False

    # Module is MXE if there are confirmed mutually exclusive paths
    return len(graph.mutually_exclusive_paths) >= 1


def classify_all_events(
    modules: List[SplicingModule],
) -> List[str]:
    """Classify all modules into event types.

    Args:
        modules: List of SplicingModule objects.

    Returns:
        List of event type strings, one per module.
    """
    return [
        classify_event(module, module.cooccurrence_graph) for module in modules
    ]


def filter_modules_by_event_type(
    modules: List[SplicingModule],
    event_type: str,
) -> List[SplicingModule]:
    """Filter modules by classified event type.

    Args:
        modules: List of SplicingModule objects.
        event_type: Event type to filter by (e.g., "SE", "A3SS").

    Returns:
        Filtered list of modules matching the event type.
    """
    return [
        m
        for m in modules
        if classify_event(m, m.cooccurrence_graph) == event_type
    ]


def get_event_type_counts(modules: List[SplicingModule]) -> dict:
    """Get count of modules for each event type.

    Args:
        modules: List of SplicingModule objects.

    Returns:
        Dict mapping event type -> count.
    """
    counts = {}
    for module in modules:
        event_type = classify_event(module, module.cooccurrence_graph)
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts
