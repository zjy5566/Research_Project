# Experiment scripts

This directory is the single entry point for experiment execution and result
analysis. Paths are resolved from each script's own location, so commands can
be launched from any working directory.

## Current B/N experiments

- `run_b_experiments.py` / `.sh`: redesigned B0--B4 matrix.
- `run_n_experiments.py` / `.sh`: redesigned N1--N5 matrix.
- `supervision_experiment_runner.py`: shared configuration and execution helper.

Use the shell launchers for sequential or Slurm execution, for example:

```bash
bash Research_Project/scripts/run_b_experiments.sh
bash Research_Project/scripts/run_n_experiments.sh n1 n5
DRY_RUN=1 bash Research_Project/scripts/run_b_experiments.sh b0 b4
```

## Legacy and ablation experiments

- `run_b1_lr_posweight_sweep.py` / `.sh`
- `run_b2_b3_experiments.py` / `.sh`
- `run_n1_experiment.py` / `.sh`
- `run_n2_n3_experiments.py` / `.sh`
- `run_generalization_ablation.py` / `.sh`

## Result analysis

The `analyze_*.py` and `build_*.py` scripts read from the workspace-level
`result/` directory and write their reports beneath that directory.

Core training modules remain one level above in `Research_Project/`; dataset
preparation scripts remain grouped separately in `Research_Project/preprocessing/`.
