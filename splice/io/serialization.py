"""
Module 23: io/serialization.py

Save and load intermediate results for incremental processing.
Uses Parquet for tabular data and pickle for complex Python objects.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


def save_checkpoint(data: Any, path: str) -> None:
    """Serialize pipeline state to pickle for resumption.

    Checkpoint can contain any Python object (results, intermediate data, config).
    Uses pickle for maximum compatibility with complex data structures.

    Args:
        data: Any Python object to save.
        path: Path to checkpoint file (typically .pkl or .pickle).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str) -> Any:
    """Load pipeline state from checkpoint.

    Loads a previously saved checkpoint containing pipeline state.

    Args:
        path: Path to checkpoint file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def save_junction_evidence(evidence: Dict[str, dict], path: str) -> None:
    """Save junction evidence as Parquet and JSON for fast reload.

    Junction evidence is stored with structured numeric fields in Parquet
    (for efficiency) and a metadata JSON file for non-numeric fields.

    Args:
        evidence: Dict mapping junction_id to evidence dict with fields:
            'junction' (Junction), 'gene_id', 'gene_name', 'is_annotated',
            'motif', 'motif_score', 'total_reads', 'mean_mapq', 'sample_counts'.
        path: Path for Parquet file (directory will be created if needed).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Prepare data for Parquet storage
    records = []
    metadata = {}

    # First pass: determine max sample count length
    max_samples = max(
        (len(sc) for ev in evidence.values() if isinstance(ev.get("sample_counts"), list) for sc in [ev.get("sample_counts", [])]),
        default=0,
    )

    for junction_id, ev in evidence.items():
        from splicekit.utils.genomic import Junction

        junction: Junction = ev.get("junction")
        sample_counts: List[int] = ev.get("sample_counts", [])

        # Extract numeric fields for Parquet
        record = {
            "junction_id": junction_id,
            "chrom": junction.chrom if junction else "NA",
            "start": junction.start if junction else 0,
            "end": junction.end if junction else 0,
            "strand": junction.strand if junction else "NA",
            "is_annotated": bool(ev.get("is_annotated", False)),
            "motif_score": float(ev.get("motif_score", np.nan)),
            "total_reads": float(ev.get("total_reads", 0)),
            "mean_mapq": float(ev.get("mean_mapq", np.nan)),
        }

        # Add sample counts as separate columns (pad to uniform length)
        for i in range(max_samples):
            sample_counts_list = ev.get("sample_counts", [])
            record[f"sample_count_{i}"] = (
                sample_counts_list[i] if i < len(sample_counts_list) else 0
            )

        records.append(record)

        # Store non-numeric metadata separately
        metadata[junction_id] = {
            "gene_id": str(ev.get("gene_id", "NA")),
            "gene_name": str(ev.get("gene_name", "NA")),
            "motif": str(ev.get("motif", "NA")),
            "sample_count_length": len(sample_counts),  # Store actual length
        }

    # Save numeric data to Parquet
    if PARQUET_AVAILABLE and records:
        df = pd.DataFrame(records)
        parquet_path = path if path.endswith(".parquet") else f"{path}.parquet"
        df.to_parquet(parquet_path, index=False)

        # Save metadata as pickle alongside Parquet
        metadata_path = f"{path}.metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Fall back to pickle if Parquet not available
        full_data = {"records": records, "metadata": metadata}
        pickle_path = path if path.endswith(".pkl") else f"{path}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_junction_evidence(path: str) -> Dict[str, dict]:
    """Load junction evidence from Parquet or pickle.

    Loads junction evidence previously saved with save_junction_evidence.
    Automatically detects format (Parquet or pickle) and loads accordingly.

    Args:
        path: Path to junction evidence file (Parquet or pickle).

    Returns:
        Dict mapping junction_id to evidence dict.

    Raises:
        FileNotFoundError: If evidence file does not exist.
    """
    from splicekit.utils.genomic import Junction

    # Try Parquet first
    parquet_path = path if path.endswith(".parquet") else f"{path}.parquet"
    if os.path.exists(parquet_path) and PARQUET_AVAILABLE:
        try:
            df = pd.read_parquet(parquet_path)

            # Load metadata
            metadata_path = f"{path}.metadata.pkl"
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)

            # Reconstruct evidence dict
            evidence = {}
            for _, row in df.iterrows():
                junction_id = row["junction_id"]

                # Extract sample counts from columns
                sample_counts = []
                i = 0
                while f"sample_count_{i}" in df.columns:
                    sample_counts.append(int(row[f"sample_count_{i}"]))
                    i += 1

                # Reconstruct Junction object
                junction = Junction(
                    chrom=row["chrom"],
                    start=int(row["start"]),
                    end=int(row["end"]),
                    strand=row["strand"],
                )

                # Get metadata
                meta = metadata.get(junction_id, {})

                # Trim sample_counts to actual length if stored in metadata
                sample_count_length = meta.get("sample_count_length", len(sample_counts))
                sample_counts = sample_counts[:sample_count_length]

                evidence[junction_id] = {
                    "junction": junction,
                    "gene_id": meta.get("gene_id", "NA"),
                    "gene_name": meta.get("gene_name", "NA"),
                    "is_annotated": bool(row["is_annotated"]),
                    "motif": meta.get("motif", "NA"),
                    "motif_score": float(row["motif_score"]),
                    "total_reads": float(row["total_reads"]),
                    "mean_mapq": float(row["mean_mapq"]),
                    "sample_counts": sample_counts,
                }

            return evidence

        except Exception as e:
            # Fall through to pickle attempt
            pass

    # Try pickle
    pickle_path = path if path.endswith(".pkl") else f"{path}.pkl"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "records" in data:
            # Old format with separated records and metadata
            records = data["records"]
            metadata = data.get("metadata", {})

            evidence = {}
            for record in records:
                junction_id = record["junction_id"]
                meta = metadata.get(junction_id, {})

                # Extract sample counts
                sample_counts = []
                i = 0
                while f"sample_count_{i}" in record:
                    sample_counts.append(int(record[f"sample_count_{i}"]))
                    i += 1

                # Trim to actual length if stored
                sample_count_length = meta.get("sample_count_length", len(sample_counts))
                sample_counts = sample_counts[:sample_count_length]

                junction = Junction(
                    chrom=record["chrom"],
                    start=int(record["start"]),
                    end=int(record["end"]),
                    strand=record["strand"],
                )

                evidence[junction_id] = {
                    "junction": junction,
                    "gene_id": meta.get("gene_id", "NA"),
                    "gene_name": meta.get("gene_name", "NA"),
                    "is_annotated": bool(record["is_annotated"]),
                    "motif": meta.get("motif", "NA"),
                    "motif_score": float(record["motif_score"]),
                    "total_reads": float(record["total_reads"]),
                    "mean_mapq": float(record["mean_mapq"]),
                    "sample_counts": sample_counts,
                }

            return evidence
        else:
            # Direct evidence dict (shouldn't happen with new format)
            return data

    raise FileNotFoundError(f"Junction evidence file not found: {path}")
