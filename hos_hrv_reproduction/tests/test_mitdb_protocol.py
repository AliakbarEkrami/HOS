"""Offline checks of the frozen cohort rules, independent of model accuracy."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hos_hrv_reproduction import mitdb_protocol as protocol
from hos_hrv_reproduction.mitdb_common import read_csv


def annotation(samples, symbols, notes=None):
    return SimpleNamespace(sample=np.asarray(samples), symbol=list(symbols),
                           aux_note=notes or [""] * len(samples))


@pytest.mark.parametrize("ventricular,total,expected", [
    (0, 100, ("LOW_PVC", 0)), (1, 100, ("LOW_PVC", 0)),
    (2, 100, ("", None)), (4, 100, ("", None)),
    (5, 100, ("HIGH_PVC", 1)), (100, 100, ("HIGH_PVC", 1)),
    (1, 101, ("LOW_PVC", 0)), (5, 101, ("", None)),
    (0, 0, ("", None)), (-1, 100, ("", None)),
])
def test_exact_burden_boundaries(ventricular, total, expected):
    assert protocol.burden_label(ventricular, total) == expected


def test_v1_cannot_be_silently_redefined():
    assert protocol.MitdbProtocol().protocol_id == "mitdb_pvc_v1"
    with pytest.raises(ValueError, match="frozen"):
        protocol.MitdbProtocol(low_pvc_max=0.02)
    modified = protocol.MitdbProtocol(protocol_id="mitdb_pvc_test", low_pvc_max=0.02)
    assert modified.low_pvc_max == 0.02


def test_denominator_contains_only_real_beats_and_records_quality_markers():
    beats = sorted(protocol.BEAT_SYMBOLS)
    symbols = ["+", *beats, "~", "!", "[", "]", "|"]
    notes = ["(N", *([""] * (len(symbols) - 1))]
    stats = protocol.annotation_statistics(annotation(range(len(symbols)), symbols, notes), 0, 100)
    assert stats["n_beats"] == 15
    assert stats["n_v"] == 1
    assert stats["pvc_fraction"] == pytest.approx(1 / 15)
    assert stats["purity_fraction"] == pytest.approx(4 / 15)
    assert set(json.loads(stats["beat_counts"])) == protocol.BEAT_SYMBOLS
    assert set(json.loads(stats["quality_flags"])) == {"+", "~", "!", "[", "]", "|"}


def test_rhythm_state_carries_across_start_and_intervals_are_half_open():
    ann = annotation([0, 90, 100, 195, 200], ["+", "N", "V", "+", "V"],
                     ["(N", "", "", "(AFIB", ""])
    stats = protocol.annotation_statistics(ann, 100, 200)
    assert stats["n_beats"] == stats["n_v"] == 1
    assert stats["allowed_rhythm_fraction"] == 0.95
    assert "rhythm_coverage" not in stats["reasons"]
    stats = protocol.annotation_statistics(ann, 101, 200)
    assert stats["n_beats"] == 0
    assert stats["allowed_rhythm_fraction"] == pytest.approx(94 / 99)
    assert "rhythm_coverage" in stats["reasons"]
    assert "no_reference_beats" in stats["reasons"]


def test_unknown_initial_rhythm_is_not_assumed_normal():
    ann = annotation([10, 20, 50], ["N", "+", "N"], ["", "(N", ""])
    assert protocol.annotation_statistics(ann, 0, 100)["allowed_rhythm_fraction"] == 0.8


def test_shared_person_and_flatline_duration_threshold():
    assert protocol.subject_id("201") == protocol.subject_id("202") == "201"
    assert protocol.subject_id("200") == "200"
    assert protocol.flatline_seconds([0, 0, 0, 1], 1) == 2
    assert protocol.flatline_seconds([0, 0, 0, 0, 1], 1) == 3
    assert protocol.flatline_seconds([1], 360) == 0


def _mock_record(tmp_path, monkeypatch, samples=None, fs=360, names=("V1", "MLII")):
    for extension in ("hea", "dat", "atr"):
        (tmp_path / f"100.{extension}").write_bytes(b"offline fixture")
    p = protocol.MitdbProtocol(protocol_id="mitdb_pvc_test", segment_seconds=3, record_seconds=6)
    ann = annotation([0, 180, 360, 540, 900, 1260, 1620, 1980],
                     ["+", "N", "N", "N", "N", "N", "N", "N"],
                     ["(N", "", "", "", "", "", "", ""])
    monkeypatch.setattr(protocol.wfdb, "rdheader", lambda path: SimpleNamespace(fs=fs, sig_name=names))
    monkeypatch.setattr(protocol.wfdb, "rdann", lambda *args: ann)
    calls = []
    waveform = np.sin(np.arange(2160) / 17) if samples is None else samples
    def read(path, channels):
        calls.append(channels)
        return np.asarray(waveform)[:, None], {"fs": fs}
    monkeypatch.setattr(protocol.wfdb, "rdsamp", read)
    return p, calls


def test_prepare_selects_mlii_by_name_and_excludes_paced_records(tmp_path, monkeypatch):
    p, calls = _mock_record(tmp_path, monkeypatch)
    result = protocol.prepare_mitdb(tmp_path, tmp_path / "run", p, records=("100", "102"))
    assert calls == [[1]]
    rows = read_csv(tmp_path / "run" / "cohort.csv")
    assert len(rows) == 4
    assert all(row["channel"] == "1" for row in rows[:2])
    assert all("paced_record" in row["exclusion_reasons"] for row in rows[2:])
    assert result["included_segments"] == 2
    assert result["status"] == "insufficient_data"
    assert (tmp_path / "run" / "source_files.json").is_file()


@pytest.mark.parametrize("failure", ["flatline", "nonfinite_signal", "incomplete_segment", "unexpected_sampling_rate", "missing_MLII"])
def test_prepare_records_waveform_and_header_failures(tmp_path, monkeypatch, failure):
    waveform = np.sin(np.arange(2160) / 17)
    kwargs = {}
    if failure == "flatline":
        waveform[:730] = 0
    elif failure == "nonfinite_signal":
        waveform[20] = np.nan
    elif failure == "incomplete_segment":
        waveform = waveform[:1000]
    elif failure == "unexpected_sampling_rate":
        kwargs["fs"] = 320
    elif failure == "missing_MLII":
        kwargs["names"] = ("V1", "V5")
    p, _ = _mock_record(tmp_path, monkeypatch, samples=waveform, **kwargs)
    protocol.prepare_mitdb(tmp_path, tmp_path / "run", p, records=("100",))
    rows = read_csv(tmp_path / "run" / "cohort.csv")
    assert rows[0]["included"] == "0"
    assert failure in rows[0]["exclusion_reasons"]


def test_official_checksum_mismatch_is_an_exclusion(tmp_path, monkeypatch):
    p, calls = _mock_record(tmp_path, monkeypatch)
    (tmp_path / "SHA256SUMS.txt").write_text("0" * 64 + "  100.dat\n")
    protocol.prepare_mitdb(tmp_path, tmp_path / "run", p, records=("100",))
    assert not calls
    assert "checksum_mismatch_dat" in read_csv(tmp_path / "run" / "cohort.csv")[0]["exclusion_reasons"]
