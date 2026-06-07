# ESOL Scaffold-Aware Solubility Project

This repository contains a reproducible ESOL analysis workflow for scaffold-aware aqueous solubility modelling.

## Run

```powershell
python .\src\esol_project.py --config .\config.yaml
```

The workflow reads `data/esol.csv` and writes:

- `results/tables/*.csv`: quality-control, benchmark, CV, ablation, uncertainty, AD, error-analysis, and screening-demo tables.
- `results/figures/*.png`: distribution, correlation, chemical-space, calibration, AD, error, bias, and molecule panels.

## Notes

The local dataset contains only ESOL SMILES and logS values. Because no external screening library is provided, the virtual-screening section is implemented as a local held-out candidate demonstration.
