# Results

This directory contains the measured sweeps and release figures used by the NeuRoute paper.

## Layout

```text
results/
|-- figures/
|   |-- recall_qps.png
|   `-- buildtime.png
|-- build_time.csv
|-- neuroute_build_breakdown.csv
|-- bigann-1b/
|-- deep1b-1b/
|-- bigann-100m/
|-- deep1b-100m/
`-- gldv2/
```

Each dataset directory contains one CSV per method. NeuRoute and FAISS-family recalls are stored as fractions from 0 to 1. DiskANN recall columns are stored as percentages from 0 to 100, matching the raw logs.

## Build-Time Files

- `build_time.csv`: method-level build-time comparison in hours.
- `neuroute_build_breakdown.csv`: NeuRoute stage-level timing in seconds and hours.
- `figures/buildtime.png`: public release plot regenerated from `build_time.csv` with the NeuRoute name.

Regenerate the build-time figure from the repository root:

```bash
python scripts/plot_build_time.py
```

## Recall-QPS Sweeps

The per-method CSVs report the measured operating points used to draw the Recall@10-QPS frontier. Baselines include FAISS HNSW, IVF-Flat, IVF-PQ, OPQ+IVF-PQ with bounded refinement, and DiskANN where feasible under the paper's single-node resource budget.
