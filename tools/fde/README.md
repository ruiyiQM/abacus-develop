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

## Adaptive inner SCF

`adaptive_scf` selects an inner-SCF stage from the preceding FT density RMS.
The first cycle uses the first (loosest) stage.  Thresholds must decrease, and
only the final stage is marked `strict`.  `force_strict_cycle` guarantees enough
remaining cycles for the requested strict confirmations even if the outer
residual stalls.

```json
"adaptive_scf": {
  "enabled": true,
  "force_strict_cycle": 45,
  "stages": [
    {
      "name": "loose",
      "minimum_density_rms": 0.001,
      "maximum_iterations": 25,
      "density_tolerance": 0.0001,
      "strict": false
    },
    {
      "name": "medium",
      "minimum_density_rms": 0.00001,
      "maximum_iterations": 60,
      "density_tolerance": 0.00001,
      "strict": false
    },
    {
      "name": "strict",
      "minimum_density_rms": 0.0,
      "maximum_iterations": 200,
      "density_tolerance": 0.000003,
      "strict": true
    }
  ]
}
```

Set `allow_partial_scf: true` because non-strict stages are deliberately allowed
to pass a valid partial density to the next subsystem.  Final convergence still
requires `strict_confirmation_cycles` complete strict cycles, the FT density
criterion, and the energy criterion.  The legacy fixed
`inexact_freeze_thaw_cycles` schedule remains supported but cannot be combined
with `adaptive_scf`.

## Inner mixing and recovery

The following ABACUS INPUT controls can be set globally, per fragment, per
adaptive stage, or per fragment within a stage.  More specific settings win:

```text
mixing_type  mixing_beta  mixing_beta_mag  mixing_ndim  mixing_restart
mixing_dmr   mixing_gg0   mixing_gg0_mag   mixing_gg0_min
```

An optional recovery list is used only when a strict inner SCF returns a valid
partial density.  Each retry starts from that partial density in a new process,
so stale Pulay/Broyden history is discarded:

```json
"mixing_recovery": {
  "enabled": true,
  "fallbacks": [
    {"mixing_type": "pulay", "mixing_beta": 0.05,
     "mixing_beta_mag": 0.05, "mixing_ndim": 8},
    {"mixing_type": "plain", "mixing_beta": 0.03,
     "mixing_beta_mag": 0.03}
  ]
}
```

The workflow does not retry a crashed ABACUS process and does not retry a
deliberately partial loose/medium stage.  It also never carries an inner mixing
history between different embedding potentials.  Retry directories use names
such as `F-retry-01`, and all attempts are retained in performance metadata.

## Tests

```bash
python3 -m unittest tools/fde/test_fde_workflow.py -v
python3 -m unittest tools/fde/test_fde_workflow_e2e.py -v
```

The example under `examples/fdedft/01_f_ch3_cl_uks` is the maintained starting
point for a complete two-state UKS calculation.
