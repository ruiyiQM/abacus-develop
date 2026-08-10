# ABACUS FDE workflow tools

`fde_workflow.py` runs restartable frozen-density embedding freeze--thaw (FT)
calculations, constructs state-specific determinants, invokes the native ABACUS
diabatic postprocessor, and writes a potential-energy-surface table.  The
workflow uses the previous FT density artifact as the next active subsystem
initial density; an `ATOMIC` line printed during ABACUS setup does not mean that
this FDE density override was skipped.

## Commands

Validate a specification without launching ABACUS:

```bash
python3 tools/fde/fde_workflow.py validate workflow.json
```

Run or resume it:

```bash
OMP_NUM_THREADS=1 python3 tools/fde/fde_workflow.py run workflow.json
```

All external commands are supplied as JSON arrays, so MPI and Slurm launchers
are explicit and shell quoting is not reinterpreted.  The workflow contains no
polling sleeps.  Time spent by a separate queue-monitoring script is not part of
the calculation.

## Performance records

Every active-fragment directory contains `fde_performance.json` with:

- launcher wall time and return code;
- requested `scf_nmax`, `scf_thr`, and mixing controls;
- electronic-step count, first/final `drho`, and accumulated step time read from
  `OUT.*/abacus.json` when available.

Each state contains restart-safe `performance.json` and line-oriented
`performance.jsonl`.  The PES root contains `fde_performance.json`, which sums
all state calls.  These reports are rewritten from checkpoint history rather
than appended blindly, so resuming a calculation does not duplicate records.

ABACUS also reports `PotFde` timers for charge validation, active-density
preparation, functional evaluation, MPI energy reduction, effective-potential
assembly, and the complete `cal_v_eff` call.  Compare both electronic-step
counts and wall time when evaluating an SCF policy: reducing FT cycles while
making every inner solve much tighter is not necessarily a speedup.

## Tests

```bash
python3 -m unittest tools/fde/test_fde_workflow.py -v
python3 -m unittest tools/fde/test_fde_workflow_e2e.py -v
```

The example under `examples/fdedft/01_f_ch3_cl_uks` is the maintained starting
point for a complete two-state UKS calculation.
