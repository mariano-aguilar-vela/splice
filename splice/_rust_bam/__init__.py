"""
Rust-accelerated BAM reader for SPLICE.

Provides a fast compiled alternative to the Python BAM reading code.
Falls back to pure Python if the Rust extension is not available.
"""

try:
    from splice_rust import extract_junction_stats_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    extract_junction_stats_rust = None
