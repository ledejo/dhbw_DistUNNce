# Untrained CNN MNIST Artifact

Projekt fuer den Vergleich von Zero-Cost-Metriken auf MNIST mit zwei Architekturfamilien:

- `plain_cnn`: klassische kleine CNNs ohne Residual-Skip-Block
- `residual_cnn`: kleine Residual-CNNs mit Skip-Connections

## Was wird gemessen

- Zero-Cost-Metriken: `snip`, `grasp`, `synflow`, `fisher`, `jacob_cov`, `grad_norm`, `naswot`
- Training/Eval: `val_acc`, `val_f1_macro`, `val_mse`
- Effizienz: `train_time_sec`, `infer_ms_per_sample`
- Ensemble: `vote_score`
- Stages im Ergebnis: `stage_untrained_metrics=untrained`, `stage_trained_eval=trained`

Zusatz: Korrelationen werden `overall` und optional pro `family` gespeichert.

## Voraussetzungen

- Python 3.10+ empfohlen
- CUDA-faehige GPU empfohlen (optional erzwungen mit `--require-cuda`)
- Abhaengigkeiten aus `requirements.txt`

## Setup

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Kompletter Lauf (Training + Analyse)

Fuehrt den kompletten Benchmark aus: Zero-Cost-Metriken berechnen, alle Modelle trainieren, CSVs und Plots erzeugen.

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe .\main.py --require-cuda --log-batches 20 --include-family-split --out-dir .\outputs
```

macOS/Linux:

```bash
.venv/bin/python main.py --require-cuda --log-batches 20 --include-family-split --out-dir ./outputs
```

Hinweis: Standardmaessig werden `24` Modelle ausgewertet.

## Nur Outputs erneuern (kein Training)

Verwendet eine vorhandene `results_mnist.csv` und erzeugt daraus neue Mapping-CSVs und Visualisierungen.
Es werden dabei keine Netze neu trainiert.

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe .\main.py --replot-only --out-dir .\outputs --include-family-split
```

macOS/Linux:

```bash
.venv/bin/python main.py --replot-only --out-dir ./outputs --include-family-split
```

Optional:

- Andere Quell-CSV nutzen: `--results-csv <pfad/zur/results_mnist.csv>`


## Wichtige Output-Dateien

- `outputs/results_mnist.csv`: Gesamtergebnis pro Modell
- `outputs/metric_mapping_overall.csv`: Spearman/Kendall ueber alle Modelle
- `outputs/metric_mapping_by_family.csv`: Spearman/Kendall pro Modellfamilie (bei `--include-family-split`)
- `outputs/*.png`: Scatter-Plots, Korrelations-Heatmaps und Rank-Agreement-Bump-Charts
