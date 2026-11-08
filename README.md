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
| `code/gen.py` | trajectory generator (`python gen.py S3 train 600 31`) |
| `code/run_gen.sh` | regenerates the full benchmark with the seeds used in the paper |
| `code/features.py` | causal real-time features and targets |
| `code/models.py` | Miner-RT, Dirlik-RT, ANN+GPR (Farid 2022), deep ensemble MLP and LSTM, PI-BNN, conformal calibration |
| `code/metrics.py` | CRPS, MALE, NRMSE, alpha-lambda, PICP, MPIW, ECE, prognostic horizon |
| `code/run_bench.py` | trains and evaluates methods on one scenario/split/seed |
| `code/run_all.sh` | every experiment in the paper |
| `code/fig_*.py`, `code/analysis.py` | figures and tables |

## Quick start

```bash
pip install numpy scipy scikit-learn torch rainflow matplotlib
cd code
bash run_gen.sh                      # regenerates data/ (about 1 h on 2 CPU cores); the released data is already in data/
python run_bench.py S3 --seed 0      # all methods on scenario S3
```

To evaluate a new method, add a class with `fit(train, val, seed)` and `predict(test, rng)` returning `(point, samples)` to `models.py` and register it in `run_bench.py`.

## Data and results

`data/` holds every trajectory (pickled lists of dicts, one per component). `results/` holds the JSON metrics behind every table in the paper.

## Licence

Code: MIT. Data: CC BY 4.0.

## Real-data test set: SAE keyhole tests

`realdata/` turns the public SAE FD&E keyhole program (Tucker and Bussa, SAE 750038, 1975; data hosted at
https://fde.uwaterloo.ca/Fde/Loads/Keyhole/keyhole.html, GPL) into FatigueBench-RT trajectories:

```
cd realdata
python parse_keyhole.py      # 16 CA and 54 VA tests -> keyhole_tests.json
python calib_keyhole.py      # load-life curve fitted on the CA tests only; Miner check on VA
python local_strain.py       # FE notch stress (Kt = 3.12, S = 11.265 P[kN]) + Neuber + SWT, no keyhole data
python build_real.py         # 47 failed VA tests -> data/REAL_keyhole.pkl
cd ../code
python run_real.py --train S4 --seed 0   # train on simulation, test on the real tests
python real_analysis.py && python fig_real.py
```
The FE analysis of the keyhole specimen (SolidWorks Simulation 2020) was provided by Prof. M. Groper.

## Adapting to a new component family (few-shot)

```
cd code
python gen.py S4 train 600 71 wide && python gen.py S4 val 150 72 wide   # domain-randomized pretraining set
bash run_ft.sh S4 && bash run_ft.sh S4_wide                              # leave-one-group-out, k = 3, 5, 10, 20, all; 5 seeds
python -c "import ft_analysis as f; f.make_outputs()"
```
Each SAE history/steel group is predicted by a model pretrained on simulation and fine-tuned on tests of the
other groups only; conformal factors come from an inner leave-one-group-out loop on the training groups.
