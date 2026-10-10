import numpy as np
import wfdb

from hos_hrv_reproduction.io import load_array


def test_wfdb_second_channel_is_selected_once(tmp_path):
    samples = np.column_stack([np.sin(np.arange(500) / 10),
                               2 * np.cos(np.arange(500) / 13)])
    wfdb.wrsamp("test_record", fs=320, units=["mV", "mV"],
                sig_name=["I", "II"], p_signal=samples, fmt=["16", "16"],
                write_dir=str(tmp_path))
    basename = tmp_path / "test_record"
    selected = load_array(basename, channel=1)
    assert selected.shape == (500,)
    assert np.allclose(selected, samples[:, 1], atol=0.001)
    assert np.array_equal(selected, load_array(basename.with_suffix(".hea"), channel=1))
