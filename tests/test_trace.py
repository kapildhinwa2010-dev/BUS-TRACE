import numpy as np
import pytest

from bus_trace.trace import top_writers, write_report


def test_basic_report():
    w0 = np.eye(4) * 10.0
    w1 = np.zeros((4, 4))
    w2 = np.ones((4, 4))
    r = write_report([w0, w1, w2])
    assert r["num_blocks"] == 3
    assert len(r["write_norms"]) == 3
    assert abs(sum(r["write_shares"]) - 1.0) < 1e-9
    assert r["silent_blocks"] == [1]
    assert r["dominant_block"] == 0
    assert r["growth"][0] is None
    assert r["stream_norms"][0] == pytest.approx(20.0)
    assert r["stream_norms"][1] == pytest.approx(20.0)
    assert r["stream_norms"][2] == pytest.approx(np.sqrt(20.0 ** 2 + 4.0 ** 2))


def test_stream_norm_growth():
    w0 = np.ones((2, 3))
    w1 = np.ones((2, 3)) * 2.0
    r = write_report([w0, w1])
    assert r["stream_norms"][0] == pytest.approx(np.sqrt(6))
    assert r["stream_norms"][1] == pytest.approx(3.0 * np.sqrt(6))
    assert r["growth"][1] == pytest.approx(3.0)


def test_batched_writes():
    w0 = np.zeros((2, 3, 4))
    w0[:, 0, 0] = 1.0
    r = write_report([w0])
    assert r["num_blocks"] == 1
    assert r["write_norms"][0] == pytest.approx(np.sqrt(2.0))
    assert r["dominant_block"] == 0


def test_top_writers():
    writes = [np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2)) * 3.0]
    r = write_report(writes)
    assert top_writers(r, k=2) == [2, 1]


def test_errors():
    with pytest.raises(ValueError):
        write_report([])
    with pytest.raises(ValueError):
        write_report([np.zeros((2, 2)), np.zeros((3, 3))])
    with pytest.raises(ValueError):
        top_writers({"not": "a report"})