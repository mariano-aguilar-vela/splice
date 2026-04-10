use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::ToPyObject;
use rust_htslib::bam::{self, Read, Record};
use rust_htslib::bam::record::Cigar;
use std::collections::HashMap;

type JunctionKey = (String, i64, i64, String);
type PairKey = (JunctionKey, JunctionKey);

#[derive(Clone, Default)]
struct JunctionSampleStats {
    counts: i64,
    mapq_sum: f64,
    mapq_sq_sum: f64,
    nh_sum: f64,
    n: i64,
    max_anchor: i64,
}

struct BamStats {
    junction_stats: HashMap<JunctionKey, JunctionSampleStats>,
    cooccurrence_counts: HashMap<PairKey, i64>,
    total_reads: i64,
    mapped_reads: i64,
    junction_reads: i64,
    multi_mapped_reads: i64,
    mapq_sum: f64,
    mapq_count: i64,
}

fn get_aligned_blocks(record: &Record) -> Vec<(i64, i64)> {
    let mut blocks = Vec::new();
    let mut ref_pos = record.pos() as i64;

    for cigar_op in record.cigar().iter() {
        match cigar_op {
            Cigar::Match(len) | Cigar::Equal(len) | Cigar::Diff(len) => {
                let len = *len as i64;
                blocks.push((ref_pos, ref_pos + len));
                ref_pos += len;
            }
            Cigar::Del(len) | Cigar::RefSkip(len) => {
                ref_pos += *len as i64;
            }
            Cigar::Ins(_) | Cigar::SoftClip(_) | Cigar::HardClip(_)
            | Cigar::Pad(_) => {}
        }
    }
    blocks
}

fn extract_junctions(
    blocks: &[(i64, i64)],
    chrom: &str,
    is_reverse: bool,
    min_anchor: i64,
) -> Vec<(JunctionKey, i64, i64)> {
    let mut junctions = Vec::new();
    let strand = if is_reverse { "-" } else { "+" };

    for i in 0..blocks.len().saturating_sub(1) {
        let intron_start = blocks[i].1;
        let intron_end = blocks[i + 1].0;
        if intron_end <= intron_start {
            continue;
        }
        let left_anchor = blocks[i].1 - blocks[i].0;
        let right_anchor = blocks[i + 1].1 - blocks[i + 1].0;
        if left_anchor >= min_anchor && right_anchor >= min_anchor {
            junctions.push((
                (chrom.to_string(), intron_start, intron_end, strand.to_string()),
                left_anchor,
                right_anchor,
            ));
        }
    }
    junctions
}

fn get_nh_tag(record: &Record) -> i64 {
    match record.aux(b"NH") {
        Ok(value) => match value {
            rust_htslib::bam::record::Aux::I8(v) => v as i64,
            rust_htslib::bam::record::Aux::U8(v) => v as i64,
            rust_htslib::bam::record::Aux::I16(v) => v as i64,
            rust_htslib::bam::record::Aux::U16(v) => v as i64,
            rust_htslib::bam::record::Aux::I32(v) => v as i64,
            rust_htslib::bam::record::Aux::U32(v) => v as i64,
            _ => 1,
        },
        Err(_) => 1,
    }
}

fn passes_quality_filters(record: &Record) -> bool {
    let flags = record.flags();
    (flags & 0x4) == 0
        && (flags & 0x100) == 0
        && (flags & 0x400) == 0
        && (flags & 0x200) == 0
}

/// Process a single record and update stats
fn process_record(
    record: &Record,
    chrom: &str,
    min_anchor: i64,
    min_mapq: i64,
    stats: &mut BamStats,
) {
    stats.total_reads += 1;

    if record.is_unmapped() {
        return;
    }

    stats.mapped_reads += 1;
    let mapq = record.mapq() as i64;
    stats.mapq_count += 1;
    stats.mapq_sum += mapq as f64;

    if !passes_quality_filters(record) {
        return;
    }
    if mapq < min_mapq {
        return;
    }

    let blocks = get_aligned_blocks(record);
    let junctions = extract_junctions(&blocks, chrom, record.is_reverse(), min_anchor);

    if junctions.is_empty() {
        return;
    }

    stats.junction_reads += 1;

    let nh = get_nh_tag(record);
    if nh > 1 {
        stats.multi_mapped_reads += 1;
    }

    for (junc_key, left_anchor, right_anchor) in junctions.iter() {
        let s = stats.junction_stats.entry(junc_key.clone()).or_default();
        s.counts += 1;
        s.mapq_sum += mapq as f64;
        s.mapq_sq_sum += (mapq * mapq) as f64;
        s.nh_sum += nh as f64;
        s.n += 1;
        s.max_anchor = s.max_anchor.max(*left_anchor).max(*right_anchor);
    }

    if junctions.len() >= 2 {
        for i in 0..junctions.len() {
            for j in (i + 1)..junctions.len() {
                let mut k1 = junctions[i].0.clone();
                let mut k2 = junctions[j].0.clone();
                if k1 > k2 {
                    std::mem::swap(&mut k1, &mut k2);
                }
                *stats.cooccurrence_counts.entry((k1, k2)).or_insert(0) += 1;
            }
        }
    }
}

fn stats_to_pydict(py: Python<'_>, stats: &BamStats) -> PyResult<PyObject> {
    let result_dict = PyDict::new_bound(py);

    let py_junction_stats = PyDict::new_bound(py);
    for (key, s) in &stats.junction_stats {
        let py_key = (key.0.as_str(), key.1, key.2, key.3.as_str()).to_object(py);
        let py_stats = PyDict::new_bound(py);
        py_stats.set_item("counts", s.counts)?;
        py_stats.set_item("mapq_sum", s.mapq_sum)?;
        py_stats.set_item("mapq_sq_sum", s.mapq_sq_sum)?;
        py_stats.set_item("nh_sum", s.nh_sum)?;
        py_stats.set_item("n", s.n)?;
        py_stats.set_item("max_anchor", s.max_anchor)?;
        py_junction_stats.set_item(py_key, py_stats)?;
    }
    result_dict.set_item("junction_stats", py_junction_stats)?;

    let py_cooccurrence = PyDict::new_bound(py);
    for (pair_key, count) in &stats.cooccurrence_counts {
        let py_k1 = (
            pair_key.0.0.as_str(), pair_key.0.1, pair_key.0.2, pair_key.0.3.as_str(),
        ).to_object(py);
        let py_k2 = (
            pair_key.1.0.as_str(), pair_key.1.1, pair_key.1.2, pair_key.1.3.as_str(),
        ).to_object(py);
        let py_pair = (&py_k1, &py_k2).to_object(py);
        py_cooccurrence.set_item(py_pair, *count)?;
    }
    result_dict.set_item("cooccurrence_counts", py_cooccurrence)?;

    result_dict.set_item("total_reads", stats.total_reads)?;
    result_dict.set_item("mapped_reads", stats.mapped_reads)?;
    result_dict.set_item("junction_reads", stats.junction_reads)?;
    result_dict.set_item("multi_mapped_reads", stats.multi_mapped_reads)?;
    result_dict.set_item("mapq_sum", stats.mapq_sum)?;
    result_dict.set_item("mapq_count", stats.mapq_count)?;

    Ok(result_dict.into())
}

#[pyfunction]
#[pyo3(signature = (bam_path, region=None, min_anchor=6, min_mapq=0))]
fn extract_junction_stats_rust(
    py: Python<'_>,
    bam_path: &str,
    region: Option<&str>,
    min_anchor: i64,
    min_mapq: i64,
) -> PyResult<PyObject> {
    let mut stats = BamStats {
        junction_stats: HashMap::new(),
        cooccurrence_counts: HashMap::new(),
        total_reads: 0,
        mapped_reads: 0,
        junction_reads: 0,
        multi_mapped_reads: 0,
        mapq_sum: 0.0,
        mapq_count: 0,
    };

    let mut record = Record::new();

    if let Some(reg) = region {
        // Region-based fetch using indexed reader
        let mut bam = bam::IndexedReader::from_path(bam_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                format!("Cannot open BAM: {}", e)
            ))?;
        bam.fetch(reg)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Invalid region '{}': {}", reg, e)
            ))?;

        while let Some(result) = bam.read(&mut record) {
            if result.is_err() { continue; }
            let tid = record.tid();
            if tid < 0 { stats.total_reads += 1; continue; }
            let chrom = std::str::from_utf8(bam.header().tid2name(tid as u32))
                .unwrap_or("").to_string();
            process_record(&record, &chrom, min_anchor, min_mapq, &mut stats);
        }
    } else {
        // Full BAM scan using regular reader (no index needed)
        let mut bam = bam::Reader::from_path(bam_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                format!("Cannot open BAM: {}", e)
            ))?;

        while let Some(result) = bam.read(&mut record) {
            if result.is_err() { continue; }
            let tid = record.tid();
            if tid < 0 { stats.total_reads += 1; continue; }
            let chrom = std::str::from_utf8(bam.header().tid2name(tid as u32))
                .unwrap_or("").to_string();
            process_record(&record, &chrom, min_anchor, min_mapq, &mut stats);
        }
    }

    stats_to_pydict(py, &stats)
}

#[pymodule]
fn splice_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_junction_stats_rust, m)?)?;
    Ok(())
}
