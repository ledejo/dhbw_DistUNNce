# Untrained CNN MNIST Artifact

Eigenstaendiges Projekt fuer den Vergleich von Zero-Cost-Metriken auf MNIST mit zwei Architekturfamilien:

- `plain_cnn`: klassische kleine CNNs ohne Residual-Skip-Block
- `residual_cnn`: kleine Residual-CNNs mit Skip-Connections

## Was wird gemessen

- Zero-Cost-Metriken: `snip`, `grasp`, `synflow`, `fisher`, `jacob_cov`, `grad_norm`, `naswot`
- Training/Eval: `val_acc`, `val_f1_macro`, `val_mse`
- Effizienz: `train_time_sec`, `infer_ms_per_sample`
- Ensemble: `vote_score`
- Stages im Ergebnis: `stage_untrained_metrics=untrained`, `stage_trained_eval=trained`

Zusatz: Korrelationen werden `overall` und optional pro `family` gespeichert.

## Start

Im Projektordner `C:\Users\User\OneDrive\Desktop\Distunnce` ausfuehren:

```powershell
.\.venv\Scripts\python.exe .\main.py --require-cuda --log-batches 20 --include-family-split
```

Kurzer Smoke-Test:

```powershell
.\.venv\Scripts\python.exe .\main.py --max-models 2 --epochs 1 --metric-batches 1 --num-workers 0 --require-cuda --log-batches 10 --out-dir .\outputs_smoke
```

## Empfehlung Anzahl NNs

Die Standardkonfiguration nutzt `24` Modelle.
