# FatigueBench-RT

Open benchmark and reference code for real-time probabilistic fatigue prognosis under random loading.

Companion code for: M. Farid, *Physics-informed Bayesian neural networks for real-time fatigue life prediction and uncertainty quantification under non-stationary random loading, with an open benchmark (FatigueBench-RT)*, submitted to the International Journal of Fatigue.

## What is in the benchmark

Each trajectory is one component monitored block by block (1 block = 1 h of operation, represented by a 4 s stress window at 512 Hz) until fatigue failure. Per block we store the observable quantities (rainflow damage with nominal material, Dirlik damage, stress statistics, spectral moments) and the hidden truth (true block damage). The failure block `K_f` is the label.

| Scenario | Loading |
|---|---|
| S1 | stationary Gaussian |
| S2 | non-stationary mission mix (2 to 4 regimes per component), Gaussian |
| S3 | S2 + non-Gaussian regimes (kurtosis 3 to 7) |
| S4 | S3 + slow load drift + load-sequence effects (Manson-Halford damage curve approach) |

Splits per scenario: 600 train, 150 validation, 250 test components. Stress tests: life-scale extrapolation (train K_f < 100, test K_f > 150), real SAE steels (Man-Ten, RQC-100), kurtosis 7 to 10.

## Files

| File | Purpose |
|---|---|
| `code/fatigue_physics.py` | Basquin, Goodman, rainflow, Miner, Dirlik, Hermite model, spectral synthesis |
| `gen.py` | trajectory generator (`python gen.py S3 train 600 31`) |
| `run_gen.sh` | regenerates the full benchmark with the seeds used in the paper |
| `features.py` | causal real-time features and targets |
| `models.py` | Miner-RT, Dirlik-RT, ANN+GPR (Farid 2022), deep ensemble MLP and LSTM, PI-BNN, conformal calibration |
| `metrics.py` | CRPS, MALE, NRMSE, alpha-lambda, PICP, MPIW, ECE, prognostic horizon |
| `run_bench.py` | trains and evaluates methods on one scenario/split/seed |
| `run_all.sh` | every experiment in the paper |
| `fig_*.py`, `analysis.py` | figures and tables |

## Quick start

```bash
pip install numpy scipy scikit-learn torch rainflow matplotlib
bash run_gen.sh                      # about 1 h on 2 CPU cores
python run_bench.py S3 --seed 0      # all methods on scenario S3
```

To evaluate a new method, add a class with `fit(train, val, seed)` and `predict(test, rng)` returning `(point, samples)` to `models.py` and register it in `run_bench.py`.

## Licence

Code: MIT. Data: CC BY 4.0.
