# BUS-TRACE

See who writes what into the residual stream, block by block.

In a residual-stream model, every block adds a write to a shared vector. BUS-TRACE takes those writes and reports the write norm and share of every block, the stream norm growth curve, silent blocks (under 1% share), and the dominant writer.

## What it does

- Per-block write norms and write shares.
- Cumulative stream norm growth after each block.
- Flags silent blocks and the dominant writer.
- Pure NumPy core; optional PyTorch hook adapter.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from bus_trace.trace import write_report, top_writers

writes = [np.load(f) for f in ["block0.npy", "block1.npy", "block2.npy"]]
report = write_report(writes)
print(report["write_shares"])
print(report["silent_blocks"], report["dominant_block"])
print(top_writers(report, k=3))
```

## How it works

Each block write is an array (T, D) or (B, T, D). BUS-TRACE computes the L2 norm of every write, its share of the total write mass, and the running stream norm if the writes are summed in order.

## Honest scope

- Share numbers describe write magnitude, not causal importance: a small write can matter and a large write can be redundant.
- The PyTorch adapter is optional and not covered by the test suite.

## License

MIT