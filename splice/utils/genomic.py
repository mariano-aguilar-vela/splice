"""
Module 1: utils/genomic.py

Core genomic data types: intervals, junctions, and interval operations.
All coordinates are 0-based half-open [start, end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True, order=True)
class GenomicInterval:
    """A genomic region with chromosome, half-open coordinates, and strand.

    Coordinates are 0-based half-open: start is inclusive, end is exclusive.
    strand must be one of '+', '-', or '.'.
    """

    chrom: str
    start: int  # 0-based inclusive
    end: int    # 0-based exclusive
    strand: str  # "+", "-", "."

    @property
    def length(self) -> int:
        """Length of the interval in base pairs."""
        return self.end - self.start


@dataclass(frozen=True, slots=True, order=True)
class Junction:
    """A splice junction defined by its intron coordinates.

    start is the first intron position (0-based inclusive).
    end is one past the last intron position (0-based exclusive).
    The donor and acceptor are strand-specific:
      On '+': donor = start (5' splice site), acceptor = end - 1 (3' splice site).
      On '-': donor = end - 1 (5' splice site in transcript), acceptor = start.
      On '.': donor = start, acceptor = end - 1 (same convention as '+').
    """

    chrom: str
    start: int  # intron start (0-based inclusive)
    end: int    # intron end (0-based exclusive)
    strand: str

    @property
    def donor(self) -> int:
        """Genomic position of the donor (5') splice site."""
        if self.strand == "-":
            return self.end - 1
        return self.start

    @property
    def acceptor(self) -> int:
        """Genomic position of the acceptor (3') splice site."""
        if self.strand == "-":
            return self.start
        return self.end - 1

    def shares_splice_site(self, other: Junction) -> bool:
        """Return True if this junction shares at least one splice site with other.

        Two junctions share a splice site if they are on the same chromosome and
        strand, and their donor or acceptor positions coincide.
        """
        if self.chrom != other.chrom or self.strand != other.strand:
            return False
        return self.donor == other.donor or self.acceptor == other.acceptor

    def to_string(self) -> str:
        """Return 'chrom:start-end:strand'."""
        return f"{self.chrom}:{self.start}-{self.end}:{self.strand}"


@dataclass(frozen=True, slots=True)
class JunctionPair:
    """Two junctions observed in the same read (co-occurrence evidence)."""

    junction1: Junction
    junction2: Junction


# ---------------------------------------------------------------------------
# Interval operations
# ---------------------------------------------------------------------------


def overlaps(a: GenomicInterval, b: GenomicInterval) -> bool:
    """Return True if intervals a and b share at least one base.

    Two intervals overlap when they are on the same chromosome, their strands
    are compatible (both equal or at least one is '.'), and their coordinate
    ranges intersect: a.start < b.end and b.start < a.end.
    """
    if a.chrom != b.chrom:
        return False
    strands_compatible = (
        a.strand == "." or b.strand == "." or a.strand == b.strand
    )
    if not strands_compatible:
        return False
    return a.start < b.end and b.start < a.end


def merge_intervals(intervals: List[GenomicInterval]) -> List[GenomicInterval]:
    """Merge a list of intervals into a minimal non-overlapping set.

    Only intervals with identical chrom and strand are merged together.
    '.' strand intervals are merged only with other '.' strand intervals.
    The returned list is sorted by (chrom, strand, start).
    """
    if not intervals:
        return []

    # Sort by (chrom, strand, start, end) so same-chrom/strand intervals are adjacent
    sorted_ivs = sorted(intervals, key=lambda iv: (iv.chrom, iv.strand, iv.start, iv.end))

    merged: List[GenomicInterval] = []
    current = sorted_ivs[0]

    for iv in sorted_ivs[1:]:
        # Only merge if same chrom and same strand (strict equality)
        if iv.chrom == current.chrom and iv.strand == current.strand and iv.start <= current.end:
            # Extend the current interval
            new_end = max(current.end, iv.end)
            current = GenomicInterval(current.chrom, current.start, new_end, current.strand)
        else:
            merged.append(current)
            current = iv

    merged.append(current)
    return merged


def subtract_interval(
    a: GenomicInterval, b: GenomicInterval
) -> List[GenomicInterval]:
    """Subtract interval b from interval a.

    Returns the list of GenomicInterval fragments of a that do not overlap b.
    If b does not overlap a, returns [a].
    If b fully covers a, returns [].
    Partial overlaps produce one or two fragments.

    The strand of returned fragments matches a.strand.
    b is treated as a region to remove regardless of its strand.
    """
    # No overlap: a is unchanged
    if a.chrom != b.chrom or a.end <= b.start or b.end <= a.start:
        return [a]

    result: List[GenomicInterval] = []

    # Left fragment: [a.start, min(a.end, b.start))
    if a.start < b.start:
        result.append(GenomicInterval(a.chrom, a.start, b.start, a.strand))

    # Right fragment: [max(a.start, b.end), a.end)
    if b.end < a.end:
        result.append(GenomicInterval(a.chrom, b.end, a.end, a.strand))

    return result
