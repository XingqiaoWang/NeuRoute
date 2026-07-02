# NeuRoute: Logit-Guided Neural Routing for Billion-Scale Vector Search with Sub-Hour Index Construction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/NeuRoute_VLDB2026.pdf)

Official implementation of **NeuRoute**, a learned hashing index that turns short binary codes into an effective routing primitive for billion-scale approximate nearest neighbor (ANN) search.

> **NeuRoute: Logit-Guided Neural Routing for Billion-Scale Vector Search with Sub-Hour Index Construction**
> Xingqiao Wang, Zi Wang, Xiaowei Xu
> University of Arkansas at Little Rock

📄 [Paper](paper/NeuRoute_VLDB2026.pdf) · [Appendix](paper/Appendix.pdf)

![Recall–QPS trade-off](results/figures/recall_qps.png)

## Highlights

- **Sub-hour billion-scale indexing.** End-to-end training + index construction completes in under one hour on a single machine for BigANN-1B and Deep1B (encoder training GPU-accelerated; construction and retrieval CPU-only).
- **Logit-guided routing.** Encoder logits serve as an uncertainty signal: deviation-to-threshold scores prioritize uncertain-bit perturbations for query-adaptive multi-bucket probing.
- **Strong accuracy–throughput trade-off.** On BigANN-1B, NeuRoute reaches 90.3% Recall@10 at 2,414 QPS — 1.7× faster than OPQ+IVF-PQ (refine) at comparable accuracy.

## How It Works

1. **Encode.** A lightweight MLP encoder is trained with a selective similarity-preserving objective to produce short, well-balanced binary addresses (e.g., 22 bits).
2. **Build.** Vectors are grouped into buckets by binary code; bucket-local clustering in the encoder's low-dimensional space forms centroids. The index is stored in a compact CSR layout.
3. **Query.** At query time the encoder logits drive (a) DP-based enumeration of candidate buckets under a radius/score budget, (b) centroid-stage gating with distance-based scoring, and (c) heap-quality-driven early stopping before exact refinement.

The retrieval pipeline is implemented in C++ (TorchScript C++ API for the encoder, vectorized AVX2 distance kernels, OpenMP multi-threading).

## Repository Structure

```
NeuRoute/
├── paper/                          # Paper PDF + appendix
├── src/                            # Python: training, model selection, index construction
│   ├── SPHash_base.py              # Core framework (AutoHash class): training, model
│   │                               #   selection, CSR index build, early-stop calibration
│   ├── indexing_model_base.py      # Base encoder architecture
│   ├── AutoHash_bigann_eu.py       # Entry point: BigANN-1B
│   ├── AutoHash_bigann_eu_100m.py  # Entry point: BigANN-100M
│   ├── AutoHash_deep1b_eu.py       # Entry point: Deep1B
│   ├── AutoHash_deep1b_eu_100m.py  # Entry point: Deep1B-100M
│   ├── AutoHash_image_eu.py        # Entry point: GLDv2 (1536-D DINOv2 embeddings)
│   ├── Bigann/  Deep1b/  GLDv2/    # Dataset-specific encoders + data preparation
│   ├── retrieval/                  # Pipeline config generation for the C++ benchmark
│   └── utils/                      # CSR builders, config generation, misc tools
├── cpp/
│   ├── bucket_cluster/             # Bucket-local clustering (index construction, needs FAISS)
│   └── query_pipeline/             # Query-time benchmark (CMake project, needs libtorch)
│       ├── bench_enum_centroid_stage_only_v6_8_aligned.cpp   # Main benchmark
│       ├── csr_index_runtime_v3_with_subcsr.h                # CSR index runtime
│       ├── query_bins_bins_only_v3a_standalone.hpp           # DP bucket enumeration
│       ├── resrank_kernels.{h,cpp}                           # Vectorized distance kernels
│       ├── torch_scorer.{h,cpp}                              # TorchScript encoder wrapper
│       └── cfg.example.json                                  # Example run configuration
├── results/                        # Benchmark results (ours + baselines)
└── requirements.txt
```

> **Naming note.** `AutoHash` / `SPHash` in source code are historical internal names of the same system described in the paper as **NeuRoute**.

## Requirements

**Python** (training + index construction)

- Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA recommended for encoder training)
- `pip install -r requirements.txt`

**C++** (retrieval pipeline)

- GCC ≥ 11 with C++17, CMake ≥ 3.18, OpenMP
- CPU with AVX2/FMA
- libtorch (reuses the PyTorch installation from your conda env)
- [FAISS](https://github.com/facebookresearch/faiss) (only for `bucket_cluster`)

## Datasets

| Dataset | Type | Dim | Size | Source |
|---|---|---|---|---|
| BigANN-1B / 100M | SIFT | 128 (u8) | 1B / 100M | [big-ann-benchmarks](https://big-ann-benchmarks.com/) |
| Deep1B / 100M | CNN features | 96 (f32) | 1B / 100M | [big-ann-benchmarks](https://big-ann-benchmarks.com/) |
| GLDv2 (aug.) | DINOv2 image embeddings | 1536 | 9.39M | [Google Landmarks v2](https://github.com/cvdfoundation/google-landmark) |

Download base/query/ground-truth files following the big-ann-benchmarks instructions, then edit the `/path/to/...` placeholders in the entry scripts (`src/AutoHash_*.py`) to point to your local copies. Helpers in `src/Bigann/data_preprocessing.py` and `src/Deep1b/data_format_transfer.py` convert `.u8bin`/`.fbin` files to the expected formats.

## End-to-End Pipeline

### 1. Train encoder & select model (Python, GPU recommended)

```bash
cd src
# Edit the path placeholders in the entry script first
python AutoHash_bigann_eu.py
```

This trains candidate encoders, selects the best configuration (`compare_models_from_config`), and writes the selected model + thresholds into `Autohash_config.json`.

### 2. Build the CSR index (Python + C++)

`build_index` in the same entry script encodes the base vectors and materializes the bucket-level CSR index. Bucket-local clustering runs via the compiled `bucket_cluster` binary:

```bash
# Build the clustering binary (FAISS + OpenMP required)
g++ -O3 -march=native -fopenmp -std=c++17 \
    cpp/bucket_cluster/bucket_cluster_pipeline_v6_8_1_gt.cpp \
    -lfaiss -o bucket_cluster_pipeline
```

`prepare_cpp_inputs_and_run` / `build_subcodes_from_next_dir` (see the commented template at the bottom of each entry script) stage the encoded vectors, run clustering, and emit the sub-CSR structures consumed by the query pipeline. Fast local storage (e.g., `/dev/shm`) is recommended for staging.

### 3. Compile the query benchmark

```bash
cd cpp/query_pipeline
conda activate <env-with-pytorch>   # CMake reuses this env's libtorch
bash cmake.sh                       # configure + build into ./build
```

### 4. Run

```bash
cp cfg.example.json cfg.json        # then edit paths/parameters
./build/bench_enum_centroid_stage_only_v6_8_aligned cfg.json
```

Key configuration fields:

| Field | Meaning |
|---|---|
| `csr_base_dir`, `subcsr_*` | CSR index files produced in step 2 |
| `torch_model_path` | TorchScript encoder from step 1 |
| `query_path`, `gt_i64_bin`, `gt_K` | Queries and ground truth |
| `metric` | `l2_u8_128` (BigANN), `l2_f32_96` (Deep1B), `ip1536` (GLDv2) |
| `D`, `D_in` | Code length / input dimensionality |
| `Lsel`, `Rmax`, `max_score` | Bucket-enumeration budget (probing aggressiveness) |
| `big_topm`, `phase2_vec_budget` | Centroid gating and refinement budget |
| `omp_threads` | Search threads (24 in the paper) |

The benchmark reports QPS, mean/p99.9 latency, and Recall@K over the configured sweep.

## Results

NeuRoute operating points at ≥ 90% Recall@10 (24 CPU threads; extracted from `results/`):

| Dataset | Best QPS @ Recall@10 ≥ 0.90 | Max Recall@10 | End-to-end build |
|---|---|---|---|
| BigANN-1B | 2,414 | 0.930 | < 1 hour |
| Deep1B | 842 | 0.931 | < 1 hour |
| BigANN-100M | 6,971 | 0.959 | — |
| Deep1B-100M | 2,001 | 0.940 | — |
| GLDv2 (1536-D) | 771 | 0.927 | — |

`results/` contains the full measured sweeps for NeuRoute and all baselines, one folder per dataset, one CSV per method:

```
results/
├── figures/recall_qps.png        # Figure above (regenerable from the CSVs)
├── bigann-1b/    neuroute.csv, diskann.csv, ivfpq.csv, opq_ivfpq_refine.csv
├── deep1b-1b/    neuroute.csv, diskann.csv, ivfpq.csv, opq_ivfpq_refine.csv
├── bigann-100m/  neuroute.csv, hnsw.csv, diskann.csv, ivfpq.csv, ivfflat.csv
├── deep1b-100m/  neuroute.csv, hnsw.csv, diskann.csv, ivfpq.csv, ivfflat.csv
└── gldv2/        neuroute.csv, hnsw.csv, diskann.csv, ivfpq.csv, ivfflat.csv
```

Baselines: FAISS HNSW / IVF-Flat / IVF-PQ / OPQ+IVF-PQ (refine) and DiskANN, all measured under the same 24-thread protocol described in the paper. Note that recall columns in DiskANN CSVs are percentages (0–100); all others are fractions (0–1).

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026neuroute,
  title   = {NeuRoute: Logit-Guided Neural Routing for Billion-Scale Vector Search with Sub-Hour Index Construction},
  author  = {Wang, Xingqiao and Wang, Zi and Xu, Xiaowei},
  journal = {Proceedings of the VLDB Endowment},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
