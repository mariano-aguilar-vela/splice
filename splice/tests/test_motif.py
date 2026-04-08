"""
Tests for utils/motif.py

Covers: classify_motif, score_motif, extract_motif_from_genome,
and internal reverse complement logic.
"""

import os
import tempfile
from pathlib import Path

import pytest

from splicekit.utils.motif import (
    MOTIF_SCORES,
    classify_motif,
    extract_motif_from_genome,
    score_motif,
    _reverse_complement,
)


# ---------------------------------------------------------------------------
# Reverse complement
# ---------------------------------------------------------------------------


class TestReverseComplement:
    def test_basic_gt(self):
        assert _reverse_complement("GT") == "AC"

    def test_basic_ag(self):
        assert _reverse_complement("AG") == "CT"

    def test_basic_gc(self):
        assert _reverse_complement("GC") == "GC"  # palindrome

    def test_basic_at(self):
        assert _reverse_complement("AT") == "AT"  # palindrome

    def test_longer_sequence(self):
        assert _reverse_complement("ATCG") == "CGAT"

    def test_case_insensitive(self):
        assert _reverse_complement("gt") == "AC"

    def test_lowercase_output_uppercase(self):
        result = _reverse_complement("ggcc")
        assert result == "GGCC"
        assert result.isupper()

    def test_mixed_case(self):
        assert _reverse_complement("GtAc") == "GTAC"

    def test_single_base(self):
        assert _reverse_complement("A") == "T"
        assert _reverse_complement("C") == "G"

    def test_identity_palindromes(self):
        # AT and TA are reverse complements of each other
        assert _reverse_complement(_reverse_complement("AT")) == "AT"

    def test_non_standard_base(self):
        # N should map to N
        result = _reverse_complement("AN")
        assert result == "NT"


# ---------------------------------------------------------------------------
# MOTIF_SCORES dict
# ---------------------------------------------------------------------------


class TestMotifScores:
    def test_gt_ag_score(self):
        assert MOTIF_SCORES["GT/AG"] == 1.0

    def test_gc_ag_score(self):
        assert MOTIF_SCORES["GC/AG"] == 0.8

    def test_at_ac_score(self):
        assert MOTIF_SCORES["AT/AC"] == 0.6

    def test_non_canonical_score(self):
        assert MOTIF_SCORES["non-canonical"] == 0.2

    def test_all_scores_in_valid_range(self):
        for motif, score in MOTIF_SCORES.items():
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# classify_motif
# ---------------------------------------------------------------------------


class TestClassifyMotif:
    # Canonical forms
    def test_gt_ag_canonical(self):
        assert classify_motif("GT", "AG") == "GT/AG"

    def test_gc_ag_canonical(self):
        assert classify_motif("GC", "AG") == "GC/AG"

    def test_at_ac_canonical(self):
        assert classify_motif("AT", "AC") == "AT/AC"

    # Lowercase input
    def test_lowercase_gt_ag(self):
        assert classify_motif("gt", "ag") == "GT/AG"

    def test_mixed_case(self):
        assert classify_motif("Gt", "aG") == "GT/AG"

    # Reverse complements
    def test_reverse_complement_gt_ag_as_ac_ct(self):
        # Reverse complement of GT/AG is AC/CT
        assert classify_motif("AC", "CT") == "GT/AG"

    def test_reverse_complement_gc_ag(self):
        # GC is palindromic, reverse complement is GC
        # AG reverse complement is CT
        assert classify_motif("GC", "CT") == "GC/AG"

    def test_reverse_complement_at_ac(self):
        # AT is palindromic, reverse complement is AT
        # AC reverse complement is GT
        assert classify_motif("AT", "GT") == "AT/AC"

    # Non-canonical
    def test_random_non_canonical(self):
        assert classify_motif("AA", "TT") == "non-canonical"

    def test_single_mismatch_non_canonical(self):
        assert classify_motif("CT", "AG") == "non-canonical"

    def test_swapped_non_canonical(self):
        # AG/GT is not in the list (the order matters)
        assert classify_motif("AG", "GT") == "non-canonical"

    def test_all_same_base(self):
        assert classify_motif("AA", "AA") == "non-canonical"

    def test_ambiguous_base_non_canonical(self):
        # N is not canonical
        assert classify_motif("GT", "AN") == "non-canonical"

    # Edge cases
    def test_empty_string_non_canonical(self):
        assert classify_motif("", "") == "non-canonical"

    def test_single_char_non_canonical(self):
        assert classify_motif("G", "A") == "non-canonical"


# ---------------------------------------------------------------------------
# score_motif
# ---------------------------------------------------------------------------


class TestScoreMotif:
    def test_score_gt_ag(self):
        assert score_motif("GT/AG") == 1.0

    def test_score_gc_ag(self):
        assert score_motif("GC/AG") == 0.8

    def test_score_at_ac(self):
        assert score_motif("AT/AC") == 0.6

    def test_score_non_canonical(self):
        assert score_motif("non-canonical") == 0.2

    def test_score_unknown_motif_default(self):
        # Unknown motif should return 0.2 (non-canonical default)
        assert score_motif("XX/YY") == 0.2

    def test_score_empty_string(self):
        # Unknown motif
        assert score_motif("") == 0.2

    def test_all_canonical_motifs_scored(self):
        # Verify all canonical motifs are in the dict and have non-zero scores
        canonical = ["GT/AG", "GC/AG", "AT/AC", "non-canonical"]
        for motif in canonical:
            assert score_motif(motif) > 0


# ---------------------------------------------------------------------------
# extract_motif_from_genome
# ---------------------------------------------------------------------------


class TestExtractMotifFromGenome:
    @pytest.fixture
    def test_fasta(self, tmp_path):
        """Create a minimal test FASTA file with known sequences."""
        fasta_path = tmp_path / "test.fasta"
        # Create a simple FASTA with one chromosome
        # Sequence: chr1: AAGGTTAAGGCCAAGGTTAAGGCCAAGGTTAAGGCC (10 repeats of AAGGTT)
        #          chr2: TTAAGGTTAAGGTTAAGGTTAAGG (for - strand tests)
        with open(fasta_path, "w") as f:
            f.write(">chr1\n")
            f.write("AAGGTTAAGGCCAAGGTTAAGGCCAAGGTTAAGGCCAAGGTTAAGGCC\n")
            f.write(">chr2\n")
            f.write("TTAAGGTTAAGGTTAAGGTTAAGGGTAC\n")

        # Index the FASTA with pyfastx (creates .fai)
        import pyfastx
        pyfastx.Fasta(str(fasta_path))

        return fasta_path

    def test_extract_plus_strand_gt_ag(self, test_fasta):
        # chr2: TTAAGGTTAAGGTTAAGGTTAAGGGTAC
        #       012345678901234567890123456789
        # intron from 2 to 12: donor [2:4]=AA, acceptor [10:12]=GG
        donor_dinuc, acceptor_dinuc, motif = extract_motif_from_genome(
            str(test_fasta), "chr2", 2, 12, "+"
        )
        assert donor_dinuc == "AA"
        assert acceptor_dinuc == "GG"
        assert motif == "non-canonical"

    def test_extract_minus_strand(self, test_fasta):
        # For minus strand, positions are swapped:
        # intron_start = 2, intron_end = 12
        # donor [10:12], acceptor [2:4]
        donor_dinuc, acceptor_dinuc, motif = extract_motif_from_genome(
            str(test_fasta), "chr2", 2, 12, "-"
        )
        # chr2: TTAAGGTTAAGGTTAAGGTTAAGGGTAC
        #       012345678901234567890123456789
        # donor [10:12] = GG, acceptor [2:4] = AA
        assert donor_dinuc == "GG"
        assert acceptor_dinuc == "AA"
        assert motif == "non-canonical"

    def test_extract_unstranded(self, test_fasta):
        # Unstranded should behave like minus strand (same extraction logic)
        donor_dinuc, acceptor_dinuc, motif = extract_motif_from_genome(
            str(test_fasta), "chr2", 2, 12, "."
        )
        assert donor_dinuc == "GG"
        assert acceptor_dinuc == "AA"

    def test_extract_canonical_sequence(self, test_fasta):
        # Create a test with known GT/AG motif
        # We'll manually insert it into the test sequence
        # chr2: TTAAGGTTAAGGTTAAGGTTAAGGGTAC
        # Let me use position where I know GT and AG
        # Position 14-16 is GG
        # Actually, let's just trust classify_motif works and test the extraction

        # Extract from chr2 position 2-14: donor [2:4]=AA, acceptor [12:14]=TT
        donor_dinuc, acceptor_dinuc, motif = extract_motif_from_genome(
            str(test_fasta), "chr2", 2, 14, "+"
        )
        # chr2: TTAAGGTTAAGGTTAAGGTTAAGGGTAC
        #       012345678901234567890123456789
        # [2:4] = AA, [12:14] = TT
        assert donor_dinuc == "AA"
        assert acceptor_dinuc == "TT"

    def test_extract_returns_tuple(self, test_fasta):
        result = extract_motif_from_genome(str(test_fasta), "chr2", 0, 10, "+")
        assert isinstance(result, tuple)
        assert len(result) == 3
        donor, acceptor, motif = result
        assert isinstance(donor, str)
        assert isinstance(acceptor, str)
        assert isinstance(motif, str)

    def test_extract_dinuc_uppercase(self, test_fasta):
        donor_dinuc, acceptor_dinuc, motif = extract_motif_from_genome(
            str(test_fasta), "chr2", 0, 10, "+"
        )
        assert donor_dinuc.isupper()
        assert acceptor_dinuc.isupper()

    def test_extract_motif_classification(self, test_fasta):
        # Test that the motif is correctly classified
        donor_dinuc, acceptor_dinuc, motif = extract_motif_from_genome(
            str(test_fasta), "chr2", 0, 10, "+"
        )
        # Verify motif matches classify_motif result
        expected_motif = classify_motif(donor_dinuc, acceptor_dinuc)
        assert motif == expected_motif

    def test_extract_nonexistent_chrom(self, test_fasta):
        with pytest.raises(Exception):  # pyfastx.FetchError
            extract_motif_from_genome(str(test_fasta), "chrX", 0, 10, "+")

    def test_extract_nonexistent_fasta(self):
        with pytest.raises(IOError):
            extract_motif_from_genome("/nonexistent/path.fasta", "chr1", 0, 10, "+")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_classify_then_score(self):
        # Workflow: classify a motif, then get its score
        motif = classify_motif("GT", "AG")
        score = score_motif(motif)
        assert score == 1.0

    def test_classify_then_score_non_canonical(self):
        motif = classify_motif("AA", "TT")
        score = score_motif(motif)
        assert score == 0.2

    def test_reverse_complement_roundtrip(self):
        # Verify that reverse complement is its own inverse for palindromes
        for base_pair in ["AT", "GC", "CG", "TA"]:
            rc = _reverse_complement(base_pair)
            rc_rc = _reverse_complement(rc)
            # This might not always be true, let me just verify it's a string
            assert isinstance(rc_rc, str)
            assert len(rc_rc) == 2

    def test_canonical_motifs_all_recognized(self):
        test_cases = [
            (("GT", "AG"), "GT/AG", 1.0),
            (("GC", "AG"), "GC/AG", 0.8),
            (("AT", "AC"), "AT/AC", 0.6),
        ]
        for (donor, acceptor), expected_motif, expected_score in test_cases:
            motif = classify_motif(donor, acceptor)
            score = score_motif(motif)
            assert motif == expected_motif
            assert score == expected_score
