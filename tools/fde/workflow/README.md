# FDE workflow policy and profiling

This package keeps decision logic and performance aggregation out of the
restart/I/O-heavy `fde_workflow.py` driver.

- `scf_policy.py` selects an adaptive inner-SCF stage from the preceding FT
  residual, recent residual ratios, subsystem iteration pressure, recovery
  use, and the strict-confirmation deadline. It is a pure policy: all evidence
  enters as JSON-like mappings and the complete decision is returned for the
  checkpoint.
- `profiler.py` converts subsystem metrics into nonoverlapping workflow
  preparation, session startup, electronic-step, ABACUS-overhead, and artifact
  validation phases. It never parses human-readable ABACUS timing tables.

Both modules are intentionally stateless. A restarted workflow reconstructs
the next decision entirely from `checkpoint.json`, and concurrent geometries or
states cannot share hidden optimizer/profiler history.

Run their tests together with the main workflow suite:

```bash
python3 -m unittest discover -s tools/fde -p 'test_*.py'
```
