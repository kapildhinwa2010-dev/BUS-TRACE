"""Measure how much each block writes into the residual stream.

The core takes a list of per-block write tensors (the amount each
block adds to the stream) and reports write norms, write shares,
stream norm growth, silent blocks, and the dominant writer.

Pure NumPy. A separate optional adapter captures writes from a
PyTorch model with forward hooks.
"""

import numpy as np


def write_report(writes):
    """Summarize per-block writes into a residual-style stream.

    ``writes`` is a list of arrays with shape (T, D) or (B, T, D),
    one entry per block, in the same order as the blocks.
    """
    if not isinstance(writes, (list, tuple)) or len(writes) == 0:
        raise ValueError("writes must be a non-empty list")
    w = [np.asarray(x, dtype=float) for x in writes]
    for x in w:
        if x.ndim not in (2, 3):
            raise ValueError("each write must have shape (T, D) or (B, T, D)")
    shapes = {x.shape[1:] for x in w}
    if len(shapes) != 1:
        raise ValueError("all writes must share one (T, D) shape")

    norms = [float(np.linalg.norm(x)) for x in w]
    total = float(np.sum(norms))
    shares = [n / total if total > 0 else 0.0 for n in norms]

    stream_norms = []
    acc = np.zeros_like(w[0])
    for x in w:
        acc = acc + x
        stream_norms.append(float(np.linalg.norm(acc)))

    growth = []
    for i in range(len(w)):
        if i == 0 or stream_norms[i - 1] < 1e-12:
            growth.append(None)
        else:
            growth.append(stream_norms[i] / stream_norms[i - 1])

    return {
        "num_blocks": len(w),
        "write_norms": norms,
        "write_shares": shares,
        "stream_norms": stream_norms,
        "growth": growth,
        "silent_blocks": [i for i, s in enumerate(shares) if s < 0.01],
        "dominant_block": int(np.argmax(norms)),
    }


def top_writers(report, k=3):
    """Return the indices of the k largest write shares."""
    if "write_shares" not in report:
        raise ValueError("report must come from write_report")
    order = sorted(range(len(report["write_shares"])),
                   key=lambda i: report["write_shares"][i], reverse=True)
    return order[:k]