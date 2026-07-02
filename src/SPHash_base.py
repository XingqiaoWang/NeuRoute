from __future__ import annotations
import numpy as np
import json, torch, os, time, shutil,subprocess
import torch.nn as nn
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["VECLIB_MAXIMUM_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "24"
from concurrent.futures import ThreadPoolExecutor
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from utils.evaluate_median_bucket_balance import evaluate_median_bucket_balance, evaluate_bucket_balance_with_threshold
from utils.csr_build_v2 import CSRBuildConfig, build_csr_artifacts
from utils.config_generation import transform_json,update_margin_position_vector, to_float_list
from utils.load_any_base import (
    load_u8_matrix_to_ram_auto,
    load_f32_matrix_to_ram_auto,
    load_any_npy_to_ram,
)
from utils.pair_dataset_viz import visualize_train_eval_and_earlystop
from utils.pair_dataset_builder import (
    SimilarPairSamplingConfig,
    build_train_npz_from_database_sampling,
    GroundTruthLoadConfig,
    build_eval_npz_from_groundtruth_source,
)

torch.set_num_threads(24)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass

def _run_streaming(cmd, env=None):
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed, rc={rc}")
    
def _json_default(o):
    """
    Make non-JSON objects serializable:
      - python type / numpy dtype / numpy scalar / numpy ndarray
      - pathlib.Path
      - bytes
      - fallback: str(o)
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    # 1) python "type" objects (e.g., np.uint8, float, int, etc.)
    if isinstance(o, type):
        return getattr(o, "__name__", str(o))

    # 2) numpy dtype objects
    if _np is not None:
        try:
            if isinstance(o, _np.dtype):
                return str(o)
        except Exception:
            pass

        # 3) numpy scalar -> python scalar
        try:
            if isinstance(o, _np.generic):
                return o.item()
        except Exception:
            pass

        # 4) numpy ndarray -> lightweight summary (avoid huge JSON)
        try:
            if isinstance(o, _np.ndarray):
                # store only metadata; DO NOT dump the full array
                return {
                    "__ndarray__": True,
                    "shape": list(o.shape),
                    "dtype": str(o.dtype),
                }
        except Exception:
            pass

    # 5) pathlib.Path
    try:
        from pathlib import Path
        if isinstance(o, Path):
            return str(o)
    except Exception:
        pass

    # 6) bytes -> repr (or base64 if you want)
    if isinstance(o, (bytes, bytearray)):
        return {"__bytes__": True, "len": len(o), "repr": repr(o[:64]) + ("..." if len(o) > 64 else "")}

    # 7) fallback
    return str(o)


def _load_json_if_exists(path: str) -> dict:
    import os, json
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {"_root": obj}
    except Exception as e:
        return {"_load_error": str(e)}


def _write_json_atomic(path: str, obj: dict) -> None:
    import os, json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _update_stage_json(path: str, stage: str, payload: dict) -> None:
    import time
    root = _load_json_if_exists(path)
    stages = root.get("stages", None)
    if not isinstance(stages, dict):
        stages = {}

    # IMPORTANT: payload may contain non-JSON types; json.dump(default=...) handles it.
    stages[stage] = payload
    root["stages"] = stages
    root["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    _write_json_atomic(path, root)

def update_earlystop_config_in_autohash_json(
        *,
        json_path: str,
        earlystop_params: dict,
        round_ndigits: int | None = 6,
        make_backup: bool = True,
    ) -> dict:
        """
        Write early-stop configuration into an existing AutoHash config.json.

        We store the parameters under:
            cfg["build_index"]["early_stop"] = earlystop_params

        This mirrors how you write back other parameters (e.g., margin_position),
        but keeps early-stop under a dedicated subtree.

        Args:
            json_path:
                Path to the AutoHash config.json you want to modify.
            earlystop_params:
                Dict of early-stop parameters to store in JSON.
            round_ndigits:
                If not None, recursively round float values to this precision to
                keep JSON readable/stable.
            make_backup:
                If True, create a timestamped backup file next to json_path.

        Returns:
            Updated config dict (entire JSON object).
        """

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"[earlystop][writeback] config not found: {json_path}")

        # Backup the original config.json if requested.
        if make_backup:
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = json_path + f".bak_{ts}"
            shutil.copy2(json_path, bak)

        with open(json_path, "r") as f:
            cfg = json.load(f)

        if "build_index" not in cfg or not isinstance(cfg["build_index"], dict):
            cfg["build_index"] = {}

        def _round_obj(o: Any):
            """
            Recursively round floats inside nested structures.
            """
            if round_ndigits is None:
                return o
            if isinstance(o, float):
                return round(o, int(round_ndigits))
            if isinstance(o, (list, tuple)):
                return [_round_obj(x) for x in o]
            if isinstance(o, dict):
                return {k: _round_obj(v) for k, v in o.items()}
            return o

        cfg["build_index"]["early_stop"] = _round_obj(earlystop_params)

        with open(json_path, "w") as f:
            json.dump(cfg, f, indent=2)

        return cfg


def _find_first_key(npz, keys) -> Optional[str]:
    """
    Return the first key that exists in npz among a candidate list, else None.
    """
    for k in keys:
        if k in npz:
            return k
    return None


def fit_earlystop_from_train_npz_only(
        *,
        train_npz: str,
        earlystop_mode: str = "shift_b",
        earlystop_train_y_transform=None,
        shiftb_targets=(0.0025,),
        verbose: bool = True,
    ) -> dict:
    """
    Fit early-stop parameters using TRAIN NPZ only (no eval/GT).

    Currently supports:
    - earlystop_mode = "shift_b"

    For "shift_b":
    We assume an upper-bound inequality of the form:
        x <= y_t + b
    where:
        y_t = transform(y)  (optional)
    We choose b to approximately satisfy a target violation rate:
        P(x > y_t + b) ~= target
    which can be achieved by:
        b = quantile(x - y_t, 1 - target)

    This yields a simple, robust estimator without requiring any evaluation set.

    Args:
        train_npz:
            Path to training NPZ generated by build_train_npz_from_database_sampling(...).
        earlystop_mode:
            Early-stop model type. Only "shift_b" is implemented here.
        earlystop_train_y_transform:
            Optional transform for y. Accepts:
            - None
            - callable(y)->y_t
            - string keywords: "identity", "sqrt", "square"/"pow2", "log1p"
        shiftb_targets:
            A tuple/list of target violation rates, e.g. (0.0025,).
            The first one is treated as "primary" selection.
        verbose:
            Print fitted values.

    Returns:
        dict:
        {
            "mode": "train_only_fit",
            "earlystop_mode": ...,
            "train_y_transform": ...,
            "shiftb_sweep": [ {target,b,train_violation_rate,train_violation_mean,x_key,y_key}, ... ],
            "selected": { ... } or None,
        }
    """

    z = np.load(train_npz, allow_pickle=False)

    # Try to locate x/y arrays robustly (adapt as needed if your NPZ uses different keys).
    kx = _find_first_key(z, ["x", "x_a", "x_masked", "masked_margin", "margin", "x_margin"])
    ky = _find_first_key(z, ["y", "y2", "y_l2sq", "l2sq", "dist", "d2", "y_distance"])

    if kx is None or ky is None:
        raise KeyError(
            f"[earlystop][train_only_fit] Cannot find x/y arrays in {train_npz}. "
            f"Found keys={list(z.keys())}. "
            f"Please ensure train_npz stores something like ('x','y') "
            f"or add your actual key names into the candidate lists."
        )

    x = np.asarray(z[kx]).reshape(-1).astype(np.float64, copy=False)
    y = np.asarray(z[ky]).reshape(-1).astype(np.float64, copy=False)

    # Apply an optional y transform.
    def apply_y_transform(arr: np.ndarray) -> np.ndarray:
        tr = earlystop_train_y_transform
        if tr is None:
            return arr
        if callable(tr):
            return tr(arr)
        if isinstance(tr, str):
            t = tr.lower()
            if t in ("identity", "none"):
                return arr
            if t in ("sqrt",):
                return np.sqrt(np.maximum(arr, 0.0))
            if t in ("square", "pow2"):
                return arr * arr
            if t in ("log1p",):
                return np.log1p(np.maximum(arr, 0.0))
        # Unknown transform spec => do nothing.
        return arr

    y_t = apply_y_transform(y)

    if earlystop_mode != "shift_b":
        raise ValueError(
            f"[earlystop][train_only_fit] Unsupported earlystop_mode={earlystop_mode} "
            f"(only 'shift_b' is implemented here)."
        )

    # Fit shift_b: choose b from quantile of diff = x - y_t.
    diff = x - y_t
    sweep = []

    for tgt in shiftb_targets:
        tgt = float(tgt)
        tgt = min(max(tgt, 0.0), 1.0)

        # b = quantile(diff, 1 - tgt) ensures P(diff > b) ~ tgt.
        q = 1.0 - tgt
        b = float(np.quantile(diff, q))

        # Compute violations on train set for reporting.
        viol_rate = float(np.mean(diff > b))
        viol_mean = float(np.mean(np.maximum(diff - b, 0.0)))

        sweep.append({
            "target": tgt,
            "b": b,
            "train_violation_rate": viol_rate,
            "train_violation_mean": viol_mean,
            "x_key": kx,
            "y_key": ky,
        })

    selected = sweep[0] if len(sweep) > 0 else None

    if verbose:
        print(f"[earlystop][train_only_fit] train_npz={train_npz} keys: x={kx} y={ky}")
        for row in sweep:
            print(
                f"target={row['target']:.6f}  b={row['b']:.6g}  "
                f"train_violation_rate={row['train_violation_rate']:.6f}  "
                f"train_violation_mean={row['train_violation_mean']:.6g}"
            )

    return {
        "mode": "train_only_fit",
        "earlystop_mode": earlystop_mode,
        "train_y_transform": (str(earlystop_train_y_transform) if earlystop_train_y_transform is not None else None),
        "shiftb_sweep": sweep,
        "selected": selected,
    }

            
def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_data(file_list, device = 'cpu'):
    """
    Load and preprocess real embeddings data.
    """
    def load_and_concatenate_numpy_files(file_list, axis=0):
        """
        Load multiple NumPy files from a list of file paths and concatenate them along the specified axis.

        Args:
            file_list (list of str): List of paths to .npy files.
            axis (int, optional): Axis along which to concatenate the arrays. Default is 0.

        Returns:
            np.array: The concatenated NumPy array, or None if no files were loaded.
        """
        arrays = []
        for file_path in file_list:
            try:
                arr = np.load(file_path)
                arrays.append(arr)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if arrays:
            # Concatenate along the given axis (ensuring the arrays are compatible).
            return np.concatenate(arrays, axis=axis)
        else:
            return None
    
    combined_array = load_and_concatenate_numpy_files(file_list, axis=0)
    if combined_array is not None:
        print("Combined array shape:", combined_array.shape)
    else:
        print("No arrays were loaded.")
    data = combined_array
    return data

def _ts_path_like_weight(weight_path: str) -> str:
    """
    Make TorchScript path consistent with weight path, only suffix differs.
    Example:
      /a/b/model_best.pt  -> /a/b/model_best.ts.pt
      /a/b/model_best.pth -> /a/b/model_best.ts.pt
      /a/b/model_best     -> /a/b/model_best.ts.pt
    """
    p = Path(weight_path)
    if p.suffix:
        return str(p.with_suffix(".ts.pt"))
    else:
        return str(p.with_name(p.name + ".ts.pt"))

def _infer_in_dim_from_model(model: nn.Module) -> int:
    """
    Best-effort infer input dim for a simple MLP-like encoder.
    Priority:
      1) model.in_dim if exists
      2) first nn.Linear.in_features
    """
    if hasattr(model, "in_dim"):
        v = int(getattr(model, "in_dim"))
        if v > 0:
            return v

    for m in model.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)

    raise RuntimeError(
        "Cannot infer in_dim from model. Please pass in_dim explicitly "
        "or add model.in_dim / ensure first layer is nn.Linear."
    )


def _save_torchscript_traced(
    model: nn.Module,
    *,
    out_path: str,
    in_dim: int,
    batch: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    strict: bool = False,
    freeze: bool = True,
    verbose: bool = True,
):
    """
    Trace + (optional) freeze + save TorchScript model.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    model = model.to(device=device)
    model.eval()

    example = torch.randn(batch, in_dim, device=device, dtype=dtype)

    with torch.no_grad():
        ts = torch.jit.trace(model, example, strict=strict)
        if freeze:
            # freeze needs eval() already; works best on CPU
            ts = torch.jit.freeze(ts)
        ts.save(out_path)

    if verbose:
        print(f"[torchscript] saved -> {out_path}")

# =============================================================================
# Helpers: JSON stage merge (append results instead of overwriting the file)
# =============================================================================


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge src into dst.
    - dict values: deep merge
    - other values: overwrite
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst



def _update_stage_json(path: str, stage: str, payload: dict) -> None:
            root = _load_json_if_exists(path)
            stages = root.get("stages", None)
            if not isinstance(stages, dict):
                stages = {}
            # keep historical stage if you want: root["stages_history"] etc (not doing here)
            stages[stage] = payload
            root["stages"] = stages
            # maintain a simple "last_updated"
            root["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            _write_json_atomic(path, root)
            
# =============================================================================
# Helper: list conversion
# =============================================================================

def to_float_list(x: Any, round_ndigits: Optional[int] = None) -> list:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if round_ndigits is None:
        return [float(v) for v in arr]
    return [round(float(v), int(round_ndigits)) for v in arr]
        
def _compute_bucket_balance_from_enc(
        enc_npy_path: str,
        threshold: np.ndarray,
        *,
        chunk_rows: int = 5_000_000,
        verbose: bool = False,
    ):
        """
        Compute bucket balance stats from encoded vectors and threshold.
        For B bits, total buckets = 2^B.

        Returns:
        stats dict: max, p99, p999, nonempty, empty_ratio
        """
        enc = np.load(enc_npy_path, mmap_mode="r")  # (N, B)
        N, B = enc.shape
        thr = np.asarray(threshold, dtype=np.float32).reshape(-1)
        if thr.size != B:
            raise ValueError(f"threshold length mismatch: thr={thr.size}, enc_dim={B}")

        nbuckets = 1 << int(B)
        counts = np.zeros(nbuckets, dtype=np.int64)

        if verbose:
            print(f"[balance] N={N:,} B={B} nbuckets={nbuckets:,} chunk_rows={chunk_rows:,}")

        row0 = 0
        while row0 < N:
            row1 = min(row0 + chunk_rows, N)
            x = np.asarray(enc[row0:row1, :], dtype=np.float32, order="C")

            codes = np.zeros(x.shape[0], dtype=np.uint32)
            for i in range(B):
                codes |= ((x[:, i] > thr[i]).astype(np.uint32) << np.uint32(i))

            counts += np.bincount(codes, minlength=nbuckets)

            row0 = row1
            if verbose:
                print(f"[balance] processed {row0:,}/{N:,}")

        nonempty_sizes = counts[counts > 0]
        nonempty = int(nonempty_sizes.size)
        empty_ratio = float(1.0 - (nonempty / nbuckets))

        if nonempty == 0:
            return {"max": 0, "p99": 0, "p999": 0, "nonempty": 0, "empty_ratio": 1.0}

        maxv = int(nonempty_sizes.max())
        p99 = int(np.quantile(nonempty_sizes, 0.99, method="linear"))
        p999 = int(np.quantile(nonempty_sizes, 0.999, method="linear"))

        return {"max": maxv, "p99": p99, "p999": p999, "nonempty": nonempty, "empty_ratio": empty_ratio}


def _resolve_weights_path(
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    model_selector: Optional[str] = None,
    results_json_path: Optional[str] = None,
) -> str:
    """
    Decide which .pt to load.

    Priority:
      1) model_path (direct .pt) if provided
      2) model_selector (stable_best / topk:K) from results_json_path or config["results"]
      3) fallback to config["build_index"]["model_path"] (old behavior)

    model_selector:
      - "stable_best"
      - "topk:K" (1-based rank)
    """
    # 1) direct path wins
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"[AutoHash] model_path not found: {model_path}")
        return model_path

    # Helper: load results container (either from file or from in-memory config)
    def _get_results_container() -> Dict[str, Any]:
        if results_json_path:
            if not os.path.isfile(results_json_path):
                raise FileNotFoundError(f"[AutoHash] results_json_path not found: {results_json_path}")
            with open(results_json_path, "r") as f:
                return json.load(f)
        return config

    # 2) selector
    if model_selector:
        container = _get_results_container()
        results = container.get("results", [])
        if not results:
            raise KeyError("[AutoHash] Cannot use model_selector: 'results' not found/empty in results json/config.")
        r0 = results[0]

        if model_selector == "stable_best":
            p = r0.get("stable_best_path", None)
            if not p:
                raise KeyError("[AutoHash] results[0]['stable_best_path'] missing.")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[AutoHash] stable_best_path not found: {p}")
            return p

        m = re.match(r"^topk:(\d+)$", model_selector)
        if m:
            k = int(m.group(1))
            best_models = r0.get("best_models", [])
            if not best_models:
                raise KeyError("[AutoHash] results[0]['best_models'] missing/empty.")
            if k < 1 or k > len(best_models):
                raise ValueError(f"[AutoHash] topk rank out of range: {k}, available 1..{len(best_models)}")
            p = best_models[k - 1].get("path", None)
            if not p:
                raise KeyError(f"[AutoHash] results[0]['best_models'][{k-1}]['path'] missing.")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[AutoHash] topk model path not found: {p}")
            return p

        raise ValueError(f"[AutoHash] Unknown model_selector='{model_selector}'. Use 'stable_best' or 'topk:K'.")

    # 3) fallback: old behavior
    p = config.get("build_index", {}).get("model_path", None)
    if not p:
        raise KeyError("[AutoHash] No model_path provided, no model_selector, and config['build_index']['model_path'] missing.")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"[AutoHash] config build_index.model_path not found: {p}")
    return p


class AutoHash:
    def __init__(self, 
                 config, 
                 metric,
                 device ='cpu',      
                 model_path=None,
                 model_selector=None,       
                 results_json_path=None):
        
        self.config = config
        # weights_path= config["build_index"]["model_path"]
        self.batch_size = config["build_index"]["batch_size"]
        self.margin_position = np.array(config["build_index"]["margin_position"])
        self.latent_dim = int(config["build_index"]["hidden_dim"])
        self.vector_dim = int(config["build_index"]["vector_dim"])
        self.metric = metric
        self.device = device

        # ---- ✅ NEW: choose weights path ----
        weights_path = _resolve_weights_path(
            config=config,
            model_path=model_path,
            model_selector=model_selector,
            results_json_path=results_json_path,
        )

        print(f"[AutoHash] Loading weights: {weights_path}")

        self.model = self.load_model_from_path(
            weights_path, self.vector_dim, self.latent_dim, metric, device
        )
                    
    def get_Autoencoder_class(self):
        """
        Should be overridden by subclasses to return the correct model class.
        """
        raise NotImplementedError("Subclasses must implement get_Autoencoder_class()")

    def reload_model(self, weights_path: str, device=None, verbose: bool = True):
        """
        Reload self.model from a weight file path.

        Args:
            weights_path: .pt weights file path
            device: override device, default self.device
            verbose: print loading info
        """
        if device is None:
            device = getattr(self, "device", "cpu")

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"[AutoHash] weights not found: {weights_path}")

        if verbose:
            print(f"[AutoHash] Reloading model weights: {weights_path}")
            print(f"[AutoHash] device={device} input_dim={self.vector_dim} encoding_dim={self.latent_dim} metric={self.metric}")

        self.model = self.load_model_from_path(
            weights_path,
            self.vector_dim,
            self.latent_dim,
            self.metric,
            device,
        )

        # bookkeeping (optional but useful)
        self.selected_weight_path = weights_path

        if verbose:
            print(f"[AutoHash] Model loaded OK: {os.path.basename(weights_path)}")
            
    def load_model_from_path(self, model_path, input_dim, encoding_dim, metric ,device):
        """
        Create the model architecture and load the saved weights.
        """
        model_class = self.get_Autoencoder_class()
        model = model_class(input_dim, encoding_dim, metric).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    

    def generate_encoder_vectors(
        self,
        *,
        x_npy_path: str,
        out_npy_path: str | None = None,   # None -> /dev/shm
        batch_size: int = 65536,
        chunk_rows: int = 4_000_000,
        num_threads: int | None = 24,
        flush_every_chunks: int = 0,       # 0 = flush only at end
        prefetch: bool = True,             # overlap I/O with compute
        verbose: bool = True,
        read_ratio: float = 1.0,           # ✅ NEW: read first ratio of rows (0~1], default 1.0
    ) -> "np.ndarray":
        """
        High-throughput CPU streaming encoder (NO input numpy memmap):
        - Parse .npy header
        - Stream input sequentially by chunks
        - Prefetch mode overlaps disk I/O with model compute using a 1-thread background reader
        - Reader uses os.preadv() into a preallocated NumPy array (avoids bytes->bytearray copies)
        - Output defaults to /dev/shm (RAM-backed tmpfs) and is returned as a memmap (mmap_mode='r')

        NEW:
        - read_ratio in (0,1]: only encode first read_ratio * N rows.

        Returns:
        enc: np.memmap loaded from out_npy_path (mmap_mode='r')
        """

        t0 = time.perf_counter()

        if num_threads is not None:
            torch.set_num_threads(int(num_threads))
            # DO NOT call torch.set_num_interop_threads() here (can abort)

        bs = int(batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {bs}")
        cr = int(chunk_rows)
        if cr <= 0:
            raise ValueError(f"chunk_rows must be positive, got {cr}")

        rr = float(read_ratio)
        if not (0.0 < rr <= 1.0):
            raise ValueError(f"read_ratio must be in (0,1], got {read_ratio}")

        # --- Parse .npy header to get shape/dtype and the raw data offset ---
        with open(x_npy_path, "rb") as f:
            magic = read_magic(f)
            if magic == (1, 0):
                (shape, fortran_order, dtype) = read_array_header_1_0(f)
            elif magic == (2, 0):
                (shape, fortran_order, dtype) = read_array_header_2_0(f)
            else:
                raise ValueError(f"Unsupported .npy version: {magic}")

            if fortran_order:
                raise ValueError("Input .npy is Fortran-order. Please convert to C-order before streaming.")
            if len(shape) != 2:
                raise ValueError(f"Expected 2D array in .npy, got shape={shape}")

            N_full, dim = int(shape[0]), int(shape[1])

            dtype = np.dtype(dtype)
            if dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
                raise ValueError(f"Expected input dtype float32/float16, got {dtype}")

            data_offset = f.tell()

        # ✅ NEW: only process first N rows
        N = int(max(1, int(N_full * rr)))

        elem_size = dtype.itemsize
        row_bytes = dim * elem_size

        # --- Model on CPU ---
        self.model = self.model.to("cpu").eval()

        with torch.inference_mode():
            dummy = torch.zeros(1, dim, dtype=torch.float32)
            enc_dim = int(self.model(dummy).shape[1])

        # --- Output path ---
        if out_npy_path is None:
            out_npy_path = os.path.join("/dev/shm", f"encoded_{os.getpid()}_{N}_{enc_dim}_f32.npy")

        out = np.lib.format.open_memmap(out_npy_path, mode="w+", dtype=np.float32, shape=(N, enc_dim))
        out.flush()  # header once

        if verbose:
            pct = 100.0 * (N / max(1, N_full))
            print(f"[encode] N_full={N_full:,}, using N={N:,} ({pct:.2f}%)")
            print(f"[encode] dim={dim}, enc_dim={enc_dim}")
            print(f"[encode] batch_size={bs:,}, chunk_rows={cr:,}, prefetch={prefetch}")
            print(f"[encode] torch_num_threads={torch.get_num_threads()}")
            print(f"[encode] input={x_npy_path}")
            print(f"[encode] output={out_npy_path}")

        # ---------- Low-level reader: preadv into a NumPy buffer (no extra copies) ----------
        MAX_IOV_BYTES = 1_073_741_824  # 1 GiB

        def _read_chunk_preadv(fd: int, row0: int, rows: int) -> np.ndarray:
            n_elems = rows * dim
            flat = np.empty(n_elems, dtype=dtype)

            b = flat.view(np.uint8)
            mv = memoryview(b)

            nbytes = n_elems * elem_size
            off0 = data_offset + row0 * row_bytes

            pos = 0
            while pos < nbytes:
                to_read = min(MAX_IOV_BYTES, nbytes - pos)
                got = os.preadv(fd, [mv[pos:pos + to_read]], off0 + pos)
                if got != to_read:
                    raise RuntimeError(
                        f"Short preadv at rows [{row0}, {row0+rows}): expected {to_read} bytes, got {got}"
                    )
                pos += got

            xchunk = flat.reshape(rows, dim)

            if xchunk.dtype != np.float32:
                xchunk = xchunk.astype(np.float32, copy=False)
            if not xchunk.flags["C_CONTIGUOUS"]:
                xchunk = np.ascontiguousarray(xchunk)

            return xchunk

        # ---------- Main loop ----------
        row0 = 0
        chunk_id = 0

        fd = os.open(x_npy_path, os.O_RDONLY)
        try:
            with torch.inference_mode():
                if not prefetch:
                    with open(x_npy_path, "rb") as f:
                        f.seek(data_offset, os.SEEK_SET)
                        while row0 < N:
                            row1 = min(row0 + cr, N)
                            rows = row1 - row0

                            flat = np.fromfile(f, dtype=dtype, count=rows * dim)
                            if flat.size != rows * dim:
                                raise RuntimeError(
                                    f"Short read at rows [{row0}, {row1}): expected {rows*dim} elems, got {flat.size}"
                                )

                            xchunk = flat.reshape(rows, dim)
                            if xchunk.dtype != np.float32:
                                xchunk = xchunk.astype(np.float32, copy=False)
                            if not xchunk.flags["C_CONTIGUOUS"]:
                                xchunk = np.ascontiguousarray(xchunk)

                            for i in range(0, rows, bs):
                                j = min(i + bs, rows)
                                xb = xchunk[i:j]
                                xt = torch.from_numpy(xb)
                                y = self.model(xt, True).numpy()
                                out[row0 + i: row0 + j, :] = y

                            row0 = row1
                            chunk_id += 1

                            if flush_every_chunks and (chunk_id % flush_every_chunks == 0):
                                out.flush()

                            if verbose:
                                dt = time.perf_counter() - t0
                                print(f"[encode] {row0:,}/{N:,} rows | elapsed {dt:.1f}s | {(row0/dt):.2f} vec/s")
                else:
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        rows0 = min(cr, N - row0)
                        fut = ex.submit(_read_chunk_preadv, fd, row0, rows0)

                        while row0 < N:
                            xchunk = fut.result()
                            rows = xchunk.shape[0]
                            row1 = row0 + rows

                            if row1 < N:
                                rows_next = min(cr, N - row1)
                                fut = ex.submit(_read_chunk_preadv, fd, row1, rows_next)

                            for i in range(0, rows, bs):
                                j = min(i + bs, rows)
                                xb = xchunk[i:j]
                                xt = torch.from_numpy(xb)
                                y = self.model(xt).numpy()
                                out[row0 + i: row0 + j, :] = y

                            row0 = row1
                            chunk_id += 1

                            if flush_every_chunks and (chunk_id % flush_every_chunks == 0):
                                out.flush()

                            if verbose:
                                dt = time.perf_counter() - t0
                                print(f"[encode] {row0:,}/{N:,} rows | elapsed {dt:.1f}s | {(row0/dt):.2f} vec/s")
        finally:
            os.close(fd)

        out.flush()
        enc = np.load(out_npy_path, mmap_mode="r")

        if verbose:
            dt = time.perf_counter() - t0
            print(f"[timing] total: {dt:.3f}s | {(N/dt):.2f} vec/s")

        return enc




    


    # =============================================================================
    # Main: compare_models_from_config (only export BEST torchscript, stage timing, stage json)
    # =============================================================================

    def compare_models_from_config(
        self,
        *,
        subset_x_npy_path: str,
        work_dir: str = "/dev/shm/model_select",
        topk: int = 5,
        read_ratio: float = 1.0,
        verbose: bool = True,
        save_json_path: str | None = None,

        # ✅ auto-apply best to AutoHash config
        target_autohash_config_path: str | None = None,
        write_back_model_path: bool = True,     # write best weight_path -> build_index.model_path
        round_ndigits: int | None = 6,
        make_backup: bool = True,

        # ✅ tmp artifacts behavior
        build_next_inputs: bool = False,        # model selection usually keep False
        cleanup_tmp: bool = True,               # delete enc_i.npy and csr_inputs_i

        # ✅ export best torchscript (same path, different suffix)
        save_best_torchscript: bool = True,
        ts_in_dim: int | None = None,           # if None, infer from model
        ts_device: str = "cpu",
        ts_strict: bool = False,
        ts_freeze: bool = True,

        # ✅ numba threads (pass-through to your evaluator)
        numba_threads: int = 24,

        # ✅ stage name in JSON (so later you can append other stages)
        stage_name: str = "model_selection",
    ):
        """
        Compare topK models from self.config.
        Score = max bucket size (smaller is better).
        Optionally export TorchScript for BEST model only.
        Write results into save_json_path under root["stages"][stage_name].

        Requirements:
        self.config contains training output json:
            {
            "config": {...},
            "results": [{
                "best_models": [{"path": ...}, ...],
                ...
            }],
            "timestamp": ...
            }
        """
        import time

        os.makedirs(work_dir, exist_ok=True)

        # ---------------- stage timing ----------------
        t0 = time.time()
        started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))

        # ---------------- resolve training container ----------------
        container = self.config
        results = container.get("results", None)
        if not results and isinstance(container.get("config", None), dict) and "results" in container["config"]:
            container = container["config"]
            results = container.get("results", None)

        if not results or not isinstance(results, list):
            raise RuntimeError(
                "[compare_models_from_config] Cannot find 'results' in self.config.\n"
                "You probably passed only self.config['config'] into AutoHash.\n"
                "Fix: pass the whole training output json (with keys: config/results/timestamp) into AutoHash."
            )

        best_models = results[0].get("best_models", [])
        if not best_models:
            raise RuntimeError("self.config['results'][0]['best_models'] missing/empty")

        candidates = [m["path"] for m in best_models[:topk] if isinstance(m, dict) and os.path.isfile(m.get("path", ""))]
        if not candidates:
            raise RuntimeError("No valid .pt paths found in best_models[:topk]")

        if save_json_path is None:
            save_json_path = target_autohash_config_path

        all_res = []
        best = None
        best_i = -1

        # ---------------- compare loop ----------------
        for i, wpath in enumerate(candidates):
            if verbose:
                print(f"\n[compare] ({i+1}/{len(candidates)}) {wpath}")

            # 1) load model weights
            self.reload_model(wpath, verbose=verbose)

            enc_out = os.path.join(work_dir, f"enc_{i}.npy")
            next_out_dir = os.path.join(work_dir, f"csr_inputs_{i}")
            if build_next_inputs:
                os.makedirs(next_out_dir, exist_ok=True)

            # 2) encode subset
            _ = self.generate_encoder_vectors(
                x_npy_path=subset_x_npy_path,
                out_npy_path=enc_out,
                read_ratio=read_ratio,
                verbose=verbose,
            )

            # 3) get best_threshold (your function)
            res = evaluate_median_bucket_balance(
                enc_out,
                num_groups=1,
                window_rows=10_000_000,
                chunk_rows=8_000_000,
                numba_threads=int(numba_threads),
                keep_codes=False,
                verbose=verbose,
                build_next_inputs=build_next_inputs,
                next_out_dir=next_out_dir if build_next_inputs else None,
            )

            best_thr = res.get("best_threshold", None)
            if best_thr is None:
                raise RuntimeError("evaluate_median_bucket_balance did not return 'best_threshold'")

            # 4) compute stats ourselves (max/p99/p999/...)
            stats = _compute_bucket_balance_from_enc(
                enc_out,
                np.asarray(best_thr, dtype=np.float32),
                chunk_rows=5_000_000,
                verbose=False,
            )

            if verbose:
                print("=== Bucket balance comparison (max / p99 / p999 / nonempty / empty_ratio) ===")
                print(
                    f"g= {int(res.get('best_group', 0)):>2d}  "
                    f"max={int(stats['max']):,}  "
                    f"p99={int(stats['p99']):,}  "
                    f"p999={int(stats['p999']):,}  "
                    f"nonempty={int(stats['nonempty']):,}  "
                    f"empty_ratio={float(stats['empty_ratio']):.6f}"
                )

            item = {
                "weight_path": wpath,
                # record what torchscript path WOULD be; we will save only for best
                "torchscript_path": _ts_path_like_weight(wpath),
                "score_max": int(stats["max"]),
                "best_group": res.get("best_group", None),
                "best_threshold": to_float_list(best_thr, round_ndigits=None),
                "stats": stats,
            }
            all_res.append(item)

            # update best
            if best is None or item["score_max"] < best["score_max"]:
                best = item
                best_i = i

            # cleanup non-best artifacts eagerly
            if cleanup_tmp:
                is_current_best = (i == best_i)
                if not is_current_best:
                    try:
                        if os.path.isfile(enc_out):
                            os.remove(enc_out)
                        if build_next_inputs and os.path.isdir(next_out_dir):
                            shutil.rmtree(next_out_dir, ignore_errors=True)
                    except Exception as e:
                        if verbose:
                            print(f"[warn] cleanup failed for {wpath}: {e}")

        assert best is not None and best_i >= 0

        # ---- save best to self.* ----
        self.selected_weight_path = best["weight_path"]
        self.selected_threshold = best["best_threshold"]
        self.selected_stats = best["stats"]

        # ---------------- export torchscript for BEST only ----------------
        best_ts_saved_path = None
        if save_best_torchscript:
            # ensure model is the best one (loop may end at non-best)
            self.reload_model(self.selected_weight_path, verbose=verbose)

            best_ts_path = _ts_path_like_weight(self.selected_weight_path)
            in_dim = int(ts_in_dim) if ts_in_dim is not None else _infer_in_dim_from_model(self.model)

            _save_torchscript_traced(
                self.model,
                out_path=best_ts_path,
                in_dim=in_dim,
                batch=1,
                device=ts_device,
                strict=ts_strict,
                freeze=ts_freeze,
                verbose=verbose,
            )
            best_ts_saved_path = best_ts_path

            # attach into best dict
            best["torchscript_path"] = best_ts_saved_path

        if verbose:
            print("\n[BEST MODEL]")
            print("weight =", self.selected_weight_path)
            bt = np.array(self.selected_threshold, dtype=np.float32)
            print("thr[:8] =", bt[:8])
            print("score(max) =", int(best["score_max"]))
            if best_ts_saved_path is not None:
                print("torchscript =", best_ts_saved_path)

        # ---------------- stage timing end ----------------
        t1 = time.time()
        finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
        run_time_sec = float(t1 - t0)

        # ---------------- stage payload (ONLY this stage) ----------------
        stage_payload = {
            "inputs": {
                "subset_x_npy_path": subset_x_npy_path,
                "work_dir": work_dir,
                "topk": int(topk),
                "read_ratio": float(read_ratio),
                "selection_metric": "max_bucket_size",
            },
            "outputs": {
                "best_model_path": best["weight_path"],
                "best_torchscript_path": best_ts_saved_path,
                "best": best,
                "all": all_res,
            },
            "timing": {
                "started_at": started_at,
                "finished_at": finished_at,
                "run_time_sec": run_time_sec,
            },
        }

        # write/merge into pipeline json under stages[stage_name]
        _update_stage_json(save_json_path, stage_name, stage_payload)

        # ---- ✅ AUTO APPLY: write best threshold back to AutoHash config.json ----
        if target_autohash_config_path is not None:
            update_margin_position_vector(
                json_path=target_autohash_config_path,
                vector=best["best_threshold"],
                round_ndigits=round_ndigits,
                make_backup=make_backup,
                model_path=(best["weight_path"] if write_back_model_path else None),
                model_selector="best_by_max",
                score_max=int(best["score_max"]),
                stats=best["stats"],
            )

        # ---- final cleanup: remove remaining best artifacts too ----
        if cleanup_tmp:
            enc_best = os.path.join(work_dir, f"enc_{best_i}.npy")
            dir_best = os.path.join(work_dir, f"csr_inputs_{best_i}")
            try:
                if os.path.isfile(enc_best):
                    os.remove(enc_best)
                if build_next_inputs and os.path.isdir(dir_best):
                    shutil.rmtree(dir_best, ignore_errors=True)
            except Exception as e:
                if verbose:
                    print(f"[warn] final cleanup failed: {e}")

        # return stage payload (not the whole json root)
        return stage_payload

    
    # ============================================================================
    # Early-stop evaluation/training utility (FULL, ready-to-replace)
    # ============================================================================
    # This file provides:
    #   1) update_earlystop_config_in_autohash_json(): write early-stop params into
    #      your existing AutoHash config.json under cfg["build_index"]["early_stop"].
    #   2) fit_earlystop_from_train_npz_only(): fit early-stop parameters from TRAIN
    #      NPZ only (no eval/GT). Currently supports earlystop_mode="shift_b".
    #   3) early_stop_function_evaluation(): your pipeline function with:
    #        - full timing (start/end + phase timings)
    #        - ALWAYS fits/writes early-stop function params:
    #            * if do_eval=True: fit from visualize output (train+eval)
    #            * if do_eval=False: fit from train_npz only (train-only)
    #        - optional write-back into config.json (target_autohash_config_path)
    #
    # Notes:
    # - This code assumes the following functions/classes already exist in your codebase:
    #     * load_u8_matrix_to_ram_auto, load_f32_matrix_to_ram_auto, load_any_npy_to_ram
    #     * SimilarPairSamplingConfig
    #     * build_train_npz_from_database_sampling
    #     * GroundTruthLoadConfig
    #     * build_eval_npz_from_groundtruth_source
    #     * visualize_train_eval_and_earlystop
    #     * self.generate_encoder_vectors
    # - The train-only fitting reads x/y arrays from train_npz by guessing keys. If it
    #   cannot find them, it throws a clear error listing available keys.
    # ============================================================================


    def early_stop_function_evaluation(
        self,
        *,
        database_file: str,
        metric: str = "l2",
        work_dir: str = "/dev/shm/earlystop_eval",

        # Base encoded vectors (database-side enc) - required for training
        base_enc_path: str = "/dev/shm/encoded_1b_22_f32.npy",

        # Query-related (ONLY needed if do_eval=True)
        query_file: str | None = None,
        q_enc_out: str | None = None,

        # Outputs
        train_npz: str | None = None,
        eval_npz: str | None = None,
        out_dir: str | None = None,

        # --- Encode params (ONLY used if do_eval=True) ---
        encode_batch_size: int = 262144,
        encode_chunk_rows: int = 8_000_000,
        encode_num_threads: int = 24,
        encode_prefetch: bool = True,
        encode_flush_every_chunks: int = 0,
        encode_verbose: bool = True,

        # --- Sampling config (training) ---
        sampling_batch_size: int = 4096,
        sampling_num_batches: int = 32,
        sampling_keep_frac: float = 0.005,
        sampling_y_keep_max=None,
        sampling_max_pairs_per_batch: int = 1_000_000,
        sampling_seed=None,
        sampling_verbose: bool = True,

        # --- GT config (ONLY used if do_eval=True) ---
        gt_mode: str = "benchmark",  # "benchmark" | "npy" | "npz"
        gt_basedir: str = "/path/to/big-ann-benchmarks/big-ann-benchmarks/data",
        gt_dataset_name: str = "bigann-1B",
        gt_k: int = 10,
        gt_I_path: str | None = None,
        gt_D_path: str | None = None,
        gt_npz_path: str | None = None,
        gt_npz_key_I: str = "gt_I",
        gt_npz_key_D: str = "gt_D",

        # --- Visualize earlystop (ONLY used if do_eval=True) ---
        earlystop_enable: bool = True,
        earlystop_mode: str = "shift_b",
        earlystop_train_y_transform=None,
        earlystop_eval_y_transform: str | None = "identity",
        shiftb_targets=(0.0025,),

        # --- Base load options ---
        base_loader: str = "u8_auto",       # "u8_auto" | "f32_auto" | "any_npy"
        base_expected_d: int | None = None,
        base_d_hint: int | None = None,

        # --- Behavior switches ---
        do_eval: bool = True,          # if False => training-only (no GT, no query)
        do_visualize: bool = True,     # if False => skip visualize even if do_eval=True

        # --- Write earlystop params back to config.json ---
        target_autohash_config_path: str | None = None,  # if not None => write build_index.early_stop
        round_ndigits: int | None = 6,
        make_backup: bool = True,

        # --- Cleanup ---
        cleanup_tmp: bool = False,

        # ---------- NEW: stage logging ----------
        stage_json_path: str | None = None,    # if set, append/merge results by stage into this JSON
        stage_name: str = "early_stop",        # base key under which to write this stage
    ):
        """
        Build early-stop training data from database sampling, and optionally evaluate/visualize
        on a query+GT set.

        IMPORTANT CHANGE (requested):
        Even if do_eval=False (no evaluation set), we still fit and write the early-stop
        function parameters from training NPZ only (no validation).

        Timing:
        Adds a unified "timing" dict to return payload, including started_at/finished_at/run_time_sec.

        Stage logging (NEW):
        If stage_json_path is provided, we merge-write into:
            root["stages"][<stage>] = payload
        We write:
            - <stage_name>.resolve_thr
            - <stage_name>.train
            - <stage_name>.train_only_fit   (if do_eval=False)
            - <stage_name>.eval.encode_query / <stage_name>.eval.build_eval_npz / <stage_name>.eval.visualize (if do_eval=True)
            - <stage_name> (overall summary)
        """


        # ---------------- overall timing ----------------
        t_all0 = time.time()
        started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all0))

        def _t() -> float:
            return time.time()

        timing = {"started_at": started_at}

        os.makedirs(work_dir, exist_ok=True)

        # Default output paths inside work_dir.
        if train_npz is None:
            train_npz = os.path.join(work_dir, "train_pairs_masked_margin.npz")
        if q_enc_out is None:
            q_enc_out = os.path.join(work_dir, "q_encoded.npy")
        if eval_npz is None:
            eval_npz = os.path.join(work_dir, "eval_pairs_masked_margin_gt_qside.npz")
        if out_dir is None:
            out_dir = work_dir + "/"

        # -------- (0) resolve threshold from self.config --------
        t0 = _t()
        if "build_index" not in self.config:
            raise KeyError("[earlystop] self.config missing 'build_index'")

        thr = self.config["build_index"].get("margin_position", None)
        if thr is None:
            raise KeyError("[earlystop] self.config['build_index']['margin_position'] missing (thr not found)")
        thr = np.asarray(thr, dtype=np.float32)
        timing["resolve_thr_sec"] = float(_t() - t0)

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.resolve_thr",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                    "run_time_sec": float(timing["resolve_thr_sec"]),
                    "summary": {"thr_len": int(thr.size), "thr_head": [float(x) for x in thr[:8]]},
                },
            )

        if encode_verbose:
            print(f"[earlystop] thr len={len(thr)} thr[:8]={thr[:8]}")

        # -------- (1) load base encoded vectors (memmap) --------
        t0 = _t()
        if not os.path.isfile(base_enc_path):
            raise FileNotFoundError(f"[earlystop] base_enc_path not found: {base_enc_path}")
        base_enc = np.load(base_enc_path, mmap_mode="r")  # shape [N, B]
        timing["load_base_enc_sec"] = float(_t() - t0)

        if encode_verbose:
            print(f"[earlystop] base_enc: {base_enc.shape} {base_enc.dtype}  path={base_enc_path}")

        # -------- (2) load base vectors into RAM (per your current loaders) --------
        t0 = _t()
        if base_expected_d is None:
            vd = self.config.get("build_index", {}).get("vector_dim", None)
            if vd is not None:
                base_expected_d = int(vd)
        if base_d_hint is None:
            base_d_hint = base_expected_d

        if base_loader == "u8_auto":
            base_vectors, N, d, fmt = load_u8_matrix_to_ram_auto(
                database_file, expected_d=base_expected_d, d_hint=base_d_hint
            )
        elif base_loader == "f32_auto":
            base_vectors, N, d, fmt = load_f32_matrix_to_ram_auto(
                database_file, expected_d=base_expected_d, d_hint=base_d_hint, prefer_memmap=False, make_contiguous=True
            )
        elif base_loader == "any_npy":
            base_vectors, N, d, fmt = load_any_npy_to_ram(database_file)
        else:
            raise ValueError(f"[earlystop] Unknown base_loader='{base_loader}'")

        timing["load_base_vectors_sec"] = float(_t() - t0)

        if encode_verbose:
            print(f"[earlystop] base_loader={base_loader}")
            print(f"[earlystop] base_vectors: N={N:,} d={d} fmt={fmt} dtype={getattr(base_vectors,'dtype',None)}")

        # -------- (3) build training NPZ by sampling from database --------
        t0 = _t()
        sampling_cfg = SimilarPairSamplingConfig(
            batch_size=sampling_batch_size,
            num_batches=sampling_num_batches,
            metric=metric,
            keep_frac=sampling_keep_frac,
            y_keep_max=sampling_y_keep_max,
            max_pairs_per_batch=sampling_max_pairs_per_batch,
            seed=sampling_seed,
            verbose=sampling_verbose,
        )

        train_out = build_train_npz_from_database_sampling(
            base_vectors=base_vectors,
            enc=base_enc,
            thr=thr,
            sampling_cfg=sampling_cfg,
            out_npz_path=train_npz,
        )
        timing["build_train_npz_sec"] = float(_t() - t0)

        print("[earlystop] train_out =", train_out)

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.train",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                    "run_time_sec": float(timing["build_train_npz_sec"]),
                    "inputs": {
                        "database_file": database_file,
                        "base_enc_path": base_enc_path,
                        "metric": metric,
                    },
                    "outputs": {"train_npz": train_npz},
                    "params": {
                        "sampling_batch_size": int(sampling_batch_size),
                        "sampling_num_batches": int(sampling_num_batches),
                        "sampling_keep_frac": float(sampling_keep_frac),
                        "sampling_y_keep_max": sampling_y_keep_max,
                        "sampling_max_pairs_per_batch": int(sampling_max_pairs_per_batch),
                        "sampling_seed": sampling_seed,
                    },
                    "summary": {
                        "train_out": train_out,  # usually small metadata; ok
                        "base_loader": base_loader,
                        "base_expected_d": base_expected_d,
                        "base_d_hint": base_d_hint,
                    },
                },
            )

        # ============================================================
        # TRAINING-ONLY MODE (NO GT, NO QUERY)
        # Still FIT and WRITE early-stop function params from training NPZ.
        # ============================================================
        if not do_eval:
            t_fit0 = _t()
            fit_out = None
            try:
                fit_out = fit_earlystop_from_train_npz_only(
                    train_npz=train_npz,
                    earlystop_mode=earlystop_mode,
                    earlystop_train_y_transform=earlystop_train_y_transform,
                    shiftb_targets=shiftb_targets,
                    verbose=encode_verbose,
                )
            except Exception as e:
                if encode_verbose:
                    print(f"[earlystop][train_only_fit][warn] failed: {e}")

            timing["train_only_fit_sec"] = float(_t() - t_fit0)

            selected = fit_out.get("selected") if isinstance(fit_out, dict) else None

            earlystop_params = {
                "enable": bool(earlystop_enable),
                "mode": str(earlystop_mode),
                "train_y_transform": (str(earlystop_train_y_transform) if earlystop_train_y_transform is not None else None),
                "eval_y_transform": None,
                "shiftb_targets": [float(x) for x in shiftb_targets] if shiftb_targets is not None else None,
                "selected": (
                    {
                        "target": float(selected["target"]),
                        "b": float(selected["b"]),
                        "train_violation_rate": float(selected.get("train_violation_rate", 0.0)),
                        "train_violation_mean": float(selected.get("train_violation_mean", 0.0)),
                        "x_key": selected.get("x_key", None),
                        "y_key": selected.get("y_key", None),
                    }
                    if selected is not None else None
                ),
                "note": "written by early_stop_function_evaluation (train_only, no eval)",
            }

            if stage_json_path:
                _update_stage_json(
                    stage_json_path,
                    stage=f"{stage_name}.train_only_fit",
                    payload={
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_fit0)),
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                        "run_time_sec": float(timing["train_only_fit_sec"]),
                        "inputs": {"train_npz": train_npz},
                        "params": {
                            "earlystop_enable": bool(earlystop_enable),
                            "earlystop_mode": str(earlystop_mode),
                            "earlystop_train_y_transform": (
                                str(earlystop_train_y_transform) if earlystop_train_y_transform is not None else None
                            ),
                            "shiftb_targets": [float(x) for x in shiftb_targets] if shiftb_targets is not None else None,
                        },
                        "summary": {
                            "fit_out": fit_out,
                            "earlystop_params": earlystop_params,
                        },
                    },
                )

            if target_autohash_config_path is not None:
                t_wb0 = _t()
                update_earlystop_config_in_autohash_json(
                    json_path=target_autohash_config_path,
                    earlystop_params=earlystop_params,
                    round_ndigits=round_ndigits,
                    make_backup=make_backup,
                )
                timing["writeback_config_sec"] = float(_t() - t_wb0)

                if stage_json_path:
                    _update_stage_json(
                        stage_json_path,
                        stage=f"{stage_name}.writeback",
                        payload={
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_wb0)),
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                            "run_time_sec": float(timing["writeback_config_sec"]),
                            "outputs": {"target_autohash_config_path": target_autohash_config_path},
                            "summary": {"round_ndigits": round_ndigits, "make_backup": bool(make_backup)},
                        },
                    )

            if cleanup_tmp:
                for p in [train_npz]:
                    try:
                        if p and os.path.isfile(p):
                            os.remove(p)
                    except Exception:
                        pass

            t_all1 = time.time()
            timing["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all1))
            timing["run_time_sec"] = float(t_all1 - t_all0)

            ret = {
                "mode": "train_only",
                "database_file": database_file,
                "base_enc_path": base_enc_path,
                "train_npz": train_npz,
                "thr_len": int(len(thr)),
                "train_out": train_out,
                "fit_out": fit_out,
                "earlystop_params_written": earlystop_params if target_autohash_config_path else None,
                "best_torch_params_written_to": target_autohash_config_path,
                "base_loader": base_loader,
                "base_expected_d": base_expected_d,
                "base_d_hint": base_d_hint,
                "timing": timing,
            }

            if stage_json_path:
                _update_stage_json(
                    stage_json_path,
                    stage=stage_name,
                    payload={
                        "started_at": started_at,
                        "finished_at": timing["finished_at"],
                        "run_time_sec": timing["run_time_sec"],
                        "mode": "train_only",
                        "inputs": {
                            "database_file": database_file,
                            "base_enc_path": base_enc_path,
                        },
                        "outputs": {
                            "train_npz": train_npz,
                            "written_config": target_autohash_config_path,
                        },
                        "summary": {
                            "fit_selected": (fit_out.get("selected") if isinstance(fit_out, dict) else None),
                        },
                        "timing_breakdown": timing,
                    },
                )

            return ret

        # ============================================================
        # EVAL MODE (requires query + GT)
        # ============================================================
        if query_file is None:
            raise ValueError("[earlystop] do_eval=True requires query_file")

        # -------- (4) encode query --------
        t0 = _t()
        _ = self.generate_encoder_vectors(
            x_npy_path=query_file,
            out_npy_path=q_enc_out,
            batch_size=encode_batch_size,
            chunk_rows=encode_chunk_rows,
            num_threads=encode_num_threads,
            prefetch=encode_prefetch,
            flush_every_chunks=encode_flush_every_chunks,
            verbose=encode_verbose,
        )
        timing["encode_query_sec"] = float(_t() - t0)

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.eval.encode_query",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                    "run_time_sec": float(timing["encode_query_sec"]),
                    "inputs": {"query_file": query_file},
                    "outputs": {"q_enc_out": q_enc_out},
                    "params": {
                        "encode_batch_size": int(encode_batch_size),
                        "encode_chunk_rows": int(encode_chunk_rows),
                        "encode_num_threads": int(encode_num_threads),
                        "encode_prefetch": bool(encode_prefetch),
                        "encode_flush_every_chunks": int(encode_flush_every_chunks),
                    },
                },
            )

        q_enc = np.load(q_enc_out, mmap_mode="r")
        if encode_verbose:
            print(f"[earlystop] q_enc: {q_enc.shape} {q_enc.dtype}")

        # -------- (5) build eval NPZ from GT source --------
        t0 = _t()
        if gt_mode == "benchmark":
            gt_cfg = GroundTruthLoadConfig(mode="benchmark", basedir=gt_basedir, dataset_name=gt_dataset_name, k=gt_k)
        elif gt_mode == "npy":
            if gt_I_path is None or gt_D_path is None:
                raise ValueError("[earlystop] gt_mode='npy' requires gt_I_path and gt_D_path")
            gt_cfg = GroundTruthLoadConfig(mode="npy", gt_I_path=gt_I_path, gt_D_path=gt_D_path)
        elif gt_mode == "npz":
            if gt_npz_path is None:
                raise ValueError("[earlystop] gt_mode='npz' requires gt_npz_path")
            gt_cfg = GroundTruthLoadConfig(
                mode="npz",
                gt_npz_path=gt_npz_path,
                npz_key_I=gt_npz_key_I,
                npz_key_D=gt_npz_key_D,
            )
        else:
            raise ValueError(f"[earlystop] Unknown gt_mode='{gt_mode}'")

        eval_out = build_eval_npz_from_groundtruth_source(
            gt_cfg=gt_cfg,
            q_enc=q_enc,
            base_enc=base_enc,
            thr=thr,
            out_npz_path=eval_npz,
            extra_meta={"note": "early_stop_function_evaluation (thr from self.config)"},
        )
        timing["build_eval_npz_sec"] = float(_t() - t0)

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.eval.build_eval_npz",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                    "run_time_sec": float(timing["build_eval_npz_sec"]),
                    "inputs": {"q_enc_out": q_enc_out, "base_enc_path": base_enc_path},
                    "outputs": {"eval_npz": eval_npz},
                    "params": {
                        "gt_mode": gt_mode,
                        "gt_dataset_name": gt_dataset_name,
                        "gt_k": int(gt_k),
                        "gt_I_path": gt_I_path,
                        "gt_D_path": gt_D_path,
                        "gt_npz_path": gt_npz_path,
                        "gt_npz_key_I": gt_npz_key_I,
                        "gt_npz_key_D": gt_npz_key_D,
                    },
                    "summary": {"eval_out": eval_out},
                },
            )

        print("[earlystop] eval_out =", eval_out)

        # -------- (6) visualize train/eval and early-stop sweep --------
        vis_out = None
        selected_row = None

        if do_visualize:
            t0 = _t()
            vis_out = visualize_train_eval_and_earlystop(
                train_npz=train_npz,
                eval_npz=eval_npz,
                out_dir=out_dir,
                earlystop_enable=earlystop_enable,
                earlystop_mode=earlystop_mode,
                earlystop_train_y_transform=earlystop_train_y_transform,
                earlystop_eval_y_transform=earlystop_eval_y_transform,
                shiftb_targets=shiftb_targets,
            )
            timing["visualize_sec"] = float(_t() - t0)

            if stage_json_path:
                _update_stage_json(
                    stage_json_path,
                    stage=f"{stage_name}.eval.visualize",
                    payload={
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                        "run_time_sec": float(timing["visualize_sec"]),
                        "inputs": {"train_npz": train_npz, "eval_npz": eval_npz},
                        "outputs": {"out_dir": out_dir},
                        "output_model":{"model":os.path.join(out_dir, "earlystop_shiftb_t0p002500_params.json")},
                        "params": {
                            "earlystop_enable": bool(earlystop_enable),
                            "earlystop_mode": str(earlystop_mode),
                            "earlystop_train_y_transform": (
                                str(earlystop_train_y_transform) if earlystop_train_y_transform is not None else None
                            ),
                            "earlystop_eval_y_transform": (
                                str(earlystop_eval_y_transform) if earlystop_eval_y_transform is not None else None
                            ),
                            "shiftb_targets": [float(x) for x in shiftb_targets] if shiftb_targets is not None else None,
                        },
                        "summary": {"vis_out_keys": list(vis_out.keys()) if isinstance(vis_out, dict) else None},
                    },
                )

            # pick selected row
            try:
                sweep = vis_out.get("shiftb_sweep", {}).get("eval_sweep", [])
                by_target = {}
                for row in sweep:
                    try:
                        by_target[float(row["target"])] = row
                    except Exception:
                        pass

                if shiftb_targets is not None and len(shiftb_targets) > 0:
                    t_primary = float(shiftb_targets[0])
                    selected_row = by_target.get(t_primary, None)

                if selected_row is None and len(sweep) > 0:
                    selected_row = sweep[0]

                for row in sweep:
                    print(
                        f"target={row['target']:.6f}  b={row['b']:.6g}  "
                        f"train_violation={row['train_violation']:.6f}  eval_violation={row['eval_violation']:.6f}"
                    )
            except Exception:
                pass

        # -------- (7) write early-stop params back to config.json --------
        earlystop_params = {
            "enable": bool(earlystop_enable),
            "mode": str(earlystop_mode),
            "train_y_transform": (str(earlystop_train_y_transform) if earlystop_train_y_transform is not None else None),
            "eval_y_transform": (str(earlystop_eval_y_transform) if earlystop_eval_y_transform is not None else None),
            "shiftb_targets": [float(x) for x in shiftb_targets] if shiftb_targets is not None else None,
            "selected": (
                {
                    "target": float(selected_row["target"]),
                    "b": float(selected_row["b"]),
                    "train_violation": float(selected_row.get("train_violation", 0.0)),
                    "eval_violation": float(selected_row.get("eval_violation", 0.0)),
                }
                if selected_row is not None else None
            ),
            "note": "written by early_stop_function_evaluation (train_eval)",
        }

        if target_autohash_config_path is not None:
            t0 = _t()
            update_earlystop_config_in_autohash_json(
                json_path=target_autohash_config_path,
                earlystop_params=earlystop_params,
                round_ndigits=round_ndigits,
                make_backup=make_backup,
            )
            timing["writeback_config_sec"] = float(_t() - t0)

            if stage_json_path:
                _update_stage_json(
                    stage_json_path,
                    stage=f"{stage_name}.writeback",
                    payload={
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t())),
                        "run_time_sec": float(timing["writeback_config_sec"]),
                        "outputs": {"target_autohash_config_path": target_autohash_config_path},
                        "summary": {"round_ndigits": round_ndigits, "make_backup": bool(make_backup)},
                    },
                )

        # -------- cleanup --------
        if cleanup_tmp:
            for p in [q_enc_out, train_npz, eval_npz]:
                try:
                    if p and os.path.isfile(p):
                        os.remove(p)
                except Exception:
                    pass

        # -------- finalize timing --------
        t_all1 = time.time()
        timing["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all1))
        timing["run_time_sec"] = float(t_all1 - t_all0)

        ret = {
            "mode": "train_eval",
            "database_file": database_file,
            "query_file": query_file,
            "base_enc_path": base_enc_path,
            "q_enc_out": q_enc_out,
            "train_npz": train_npz,
            "eval_npz": eval_npz,
            "work_dir": work_dir,
            "out_dir": out_dir,
            "thr_len": int(len(thr)),
            "train_out": train_out,
            "eval_out": eval_out,
            "vis_out": vis_out,
            "earlystop_params_written": earlystop_params,
            "best_torch_params_written_to": target_autohash_config_path,
            "base_loader": base_loader,
            "base_expected_d": base_expected_d,
            "base_d_hint": base_d_hint,
            "timing": timing,
        }

        if stage_json_path:
            # overall summary stage
            _update_stage_json(
                stage_json_path,
                stage=stage_name,
                payload={
                    "started_at": started_at,
                    "finished_at": timing["finished_at"],
                    "run_time_sec": timing["run_time_sec"],
                    "mode": "train_eval",
                    "inputs": {
                        "database_file": database_file,
                        "query_file": query_file,
                        "base_enc_path": base_enc_path,
                    },
                    "outputs": {
                        "train_npz": train_npz,
                        "eval_npz": eval_npz,
                        "out_dir": out_dir,
                        "written_config": target_autohash_config_path,
                    },
                    "summary": {
                        "selected": earlystop_params.get("selected", None),
                    },
                    "timing_breakdown": timing,
                },
            )

        return ret


    def build_index(
    self,
    *,
    x_npy_path: str,
    enc_out: str = "/dev/shm/encoded_1b_22_f32.npy",

    # ---------- encode params ----------
    encode_batch_size: int = 262144,
    encode_chunk_rows: int = 8_000_000,
    encode_num_threads: int = 24,
    encode_prefetch: bool = True,
    encode_flush_every_chunks: int = 0,
    encode_verbose: bool = True,

    # ---------- threshold ----------
    threshold: np.ndarray | None = None,   # None -> take from self.config["build_index"]["margin_position"]

    # ---------- eval params ----------
    eval_chunk_rows: int = 8_000_000,
    eval_numba_threads: int = 24,
    build_next_inputs: bool = True,
    next_out_dir: str = "/dev/shm/csr_index",
    keep_codes: bool = False,
    eval_verbose: bool = True,

    # ---------- NEW: stage logging ----------
    stage_json_path: str | None = None,    # if set, append/merge results by stage into this JSON
    stage_name: str = "build_index",       # key under which to write this stage
):
        """
        Wrapper for:
        (1) self.generate_encoder_vectors(...)
        (2) evaluate_bucket_balance_with_threshold(enc_out, threshold=thr, ...)

        Adds timing:
        - started_at / finished_at / run_time_sec (whole build_index)
        - encode_time_sec
        - eval_time_sec

        If stage_json_path is provided:
        - merge-write JSON by stage_name (does NOT overwrite other stages)
        - writes per-stage timing + key output paths for downstream stages
        """
     
        

        

        # ---------------- timing: whole function ----------------
        t0 = time.time()
        started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))

        # ---------- (0) resolve threshold ----------
        if threshold is None:
            if "build_index" not in self.config:
                raise KeyError("[build_index] self.config missing 'build_index'")
            threshold = self.config["build_index"].get("margin_position", None)
            if threshold is None:
                raise KeyError("[build_index] self.config['build_index']['margin_position'] missing")

        thr = np.asarray(threshold, dtype=np.float32).reshape(-1)

        # ---------- (1) encode ----------
        t_enc0 = time.time()
        _ = self.generate_encoder_vectors(
            x_npy_path=x_npy_path,
            out_npy_path=enc_out,
            batch_size=encode_batch_size,
            chunk_rows=encode_chunk_rows,
            num_threads=encode_num_threads,
            prefetch=encode_prefetch,
            flush_every_chunks=encode_flush_every_chunks,
            verbose=encode_verbose,
        )
        t_enc1 = time.time()
        encode_time_sec = float(t_enc1 - t_enc0)

        # ---------- stage write: encode only (optional) ----------
        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.encode",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_enc0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_enc1)),
                    "run_time_sec": encode_time_sec,
                    "inputs": {"x_npy_path": x_npy_path},
                    "outputs": {"enc_out": enc_out},
                    "params": {
                        "encode_batch_size": int(encode_batch_size),
                        "encode_chunk_rows": int(encode_chunk_rows),
                        "encode_num_threads": int(encode_num_threads),
                        "encode_prefetch": bool(encode_prefetch),
                        "encode_flush_every_chunks": int(encode_flush_every_chunks),
                    },
                },
            )

        # ---------- (2) evaluate bucket balance with fixed thr ----------
        os.makedirs(next_out_dir, exist_ok=True)

        t_eval0 = time.time()
        out = evaluate_bucket_balance_with_threshold(
            enc_out,
            threshold=thr,
            chunk_rows=eval_chunk_rows,
            numba_threads=eval_numba_threads,
            build_next_inputs=build_next_inputs,
            next_out_dir=next_out_dir,
            keep_codes=keep_codes,
            verbose=eval_verbose,
        )
        t_eval1 = time.time()
        eval_time_sec = float(t_eval1 - t_eval0)

        # ---------- stage write: eval only (optional) ----------
        if stage_json_path:
            # keep it lightweight: store key summary + timing + paths (not huge arrays)
            rep = out.get("report", {}) if isinstance(out, dict) else {}
            next_inputs = out.get("next_inputs", None) if isinstance(out, dict) else None

            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.eval",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_eval0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_eval1)),
                    "run_time_sec": eval_time_sec,
                    "inputs": {"enc_out": enc_out},
                    "outputs": {
                        "next_out_dir": next_out_dir,
                        "next_inputs": next_inputs,  # contains file paths + pass timings if build_next_inputs=True
                    },
                    "params": {
                        "eval_chunk_rows": int(eval_chunk_rows),
                        "eval_numba_threads": int(eval_numba_threads),
                        "build_next_inputs": bool(build_next_inputs),
                        "keep_codes": bool(keep_codes),
                    },
                    "summary": {
                        "max_bucket": int(rep.get("max_bucket", 0)) if isinstance(rep, dict) else None,
                        "p50": int(rep.get("p50", 0)) if isinstance(rep, dict) else None,
                        "p99": int(rep.get("p99", 0)) if isinstance(rep, dict) else None,
                        "p999": int(rep.get("p999", 0)) if isinstance(rep, dict) else None,
                        "nonempty": int(rep.get("nonempty", 0)) if isinstance(rep, dict) else None,
                        "empty_ratio": float(rep.get("empty_ratio", 1.0)) if isinstance(rep, dict) else None,
                    },
                },
            )

        # ---------------- timing: finalize ----------------
        t1 = time.time()
        finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
        run_time_sec = float(t1 - t0)

        # Merge timing info into output (keep any existing timing info too)
        timing_existing = out.get("timing", {}) if isinstance(out, dict) else {}
        if not isinstance(timing_existing, dict):
            timing_existing = {"_orig_timing": timing_existing}

        timing = {
            **timing_existing,
            "started_at": started_at,
            "finished_at": finished_at,
            "run_time_sec": run_time_sec,
            "encode_time_sec": encode_time_sec,
            "eval_time_sec": eval_time_sec,

            # handy context (optional)
            "enc_out": enc_out,
            "next_out_dir": next_out_dir,
            "encode_batch_size": int(encode_batch_size),
            "encode_chunk_rows": int(encode_chunk_rows),
            "encode_num_threads": int(encode_num_threads),
            "eval_chunk_rows": int(eval_chunk_rows),
            "eval_numba_threads": int(eval_numba_threads),
        }

        if isinstance(out, dict):
            out["timing"] = timing

        # ---------- stage write: whole build_index (optional) ----------
        if stage_json_path:
            # keep it small; do not dump giant fields
            rep = out.get("report", {}) if isinstance(out, dict) else {}
            next_inputs = out.get("next_inputs", None) if isinstance(out, dict) else None
            _update_stage_json(
                stage_json_path,
                stage=stage_name,
                payload={
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "run_time_sec": run_time_sec,
                    "timing": {
                        "encode_time_sec": encode_time_sec,
                        "eval_time_sec": eval_time_sec,
                    },
                    "inputs": {
                        "x_npy_path": x_npy_path,
                        "threshold_source": ("arg" if threshold is not None else "config.build_index.margin_position"),
                    },
                    "outputs": {
                        "enc_out": enc_out,
                        "next_out_dir": next_out_dir,
                        "next_inputs": next_inputs,
                    },
                    "summary": {
                        "max_bucket": int(rep.get("max_bucket", 0)) if isinstance(rep, dict) else None,
                        "nonempty": int(rep.get("nonempty", 0)) if isinstance(rep, dict) else None,
                        "empty_ratio": float(rep.get("empty_ratio", 1.0)) if isinstance(rep, dict) else None,
                    },
                    "params": {
                        "encode_batch_size": int(encode_batch_size),
                        "encode_chunk_rows": int(encode_chunk_rows),
                        "encode_num_threads": int(encode_num_threads),
                        "eval_chunk_rows": int(eval_chunk_rows),
                        "eval_numba_threads": int(eval_numba_threads),
                        "build_next_inputs": bool(build_next_inputs),
                        "keep_codes": bool(keep_codes),
                    },
                },
            )

        return out

        
    def build_csr_from_next_dir(
    self,
    *,
    base_dir: str,
    vectors_path: str,
    d: int,
    next_out_dir: str,

    # -------- database / loader options (match your script) --------
    vectors_format: str = "auto",          # "auto" | "npy_any" | "u8_auto" | "f32_auto" | "raw_f32bin"
    npy_cast_dtype: str | None = None,     # only for vectors_format="npy_any"
    N_for_raw_vectors: int | None = None,  # only required for vectors_format="raw_f32bin"

    # -------- outputs for runtime --------
    out_codes_dtype=np.uint8,              # np.uint8 or np.float32 ...
    pos_block: int = 2_000_000,
    io_buffer_mb: int = 16,

    # -------- filenames under next_out_dir --------
    indices_u32_name: str = "indices.uint32.bin",
    offsets_u64_name: str = "offsets.uint64.npy",

    # -------- optional override output paths --------
    out_codes_path: str | None = None,
    out_ids_path: str | None = None,
    out_offsets_bin: str | None = None,
    out_bucket_sizes_bin: str | None = None,
    out_manifest: str | None = None,

    verbose: bool = True,

    # ---------- NEW: stage logging ----------
    stage_json_path: str | None = None,    # if set, append/merge results by stage into this JSON
    stage_name: str = "csr_build",         # key under which to write this stage
):
        """
        Build final CSR artifacts using indices/offsets produced by evaluate_* stage.
        Supports multiple database formats via vectors_format.

        Required in next_out_dir:
        - indices.uint32.bin
        - offsets.uint64.npy

        Timing:
        Adds wall-clock timing information to the returned dict:
            out["timing"] = {
            "started_at": ...,
            "finished_at": ...,
            "run_time_sec": ...,
            "build_csr_sec": ...,
            }

        Stage logging (NEW):
        If stage_json_path is provided, merge-write:
            root["stages"][<stage_name>] = {...}
        Also write a sub-stage:
            root["stages"][<stage_name>+".build"] = {...}
        """
        # ---------------- overall timing ----------------
        t_all0 = time.time()
        timing = {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all0)),
        }

        indices_u32_path = os.path.join(next_out_dir, indices_u32_name)
        offsets_path = os.path.join(next_out_dir, offsets_u64_name)

        if not os.path.isfile(indices_u32_path):
            raise FileNotFoundError(f"[csr] missing indices file: {indices_u32_path}")
        if not os.path.isfile(offsets_path):
            raise FileNotFoundError(f"[csr] missing offsets file: {offsets_path}")

        cfg = CSRBuildConfig(
            base_dir=base_dir,
            vectors_path=vectors_path,
            d=int(d),
            indices_u32_path=indices_u32_path,
            offsets_path=offsets_path,

            vectors_format=str(vectors_format),
            npy_cast_dtype=npy_cast_dtype,
            N_for_raw_vectors=N_for_raw_vectors,

            out_codes_dtype=out_codes_dtype,
            pos_block=int(pos_block),
            io_buffer_mb=int(io_buffer_mb),

            out_codes_path=out_codes_path,
            out_ids_path=out_ids_path,
            out_offsets_bin=out_offsets_bin,
            out_bucket_sizes_bin=out_bucket_sizes_bin,
            out_manifest=out_manifest,
        )

        if verbose:
            print("[csr] vectors_path    =", vectors_path)
            print("[csr] vectors_format  =", vectors_format)
            print("[csr] d               =", d)
            print("[csr] next_out_dir     =", next_out_dir)
            print("[csr] indices          =", indices_u32_path)
            print("[csr] offsets          =", offsets_path)

        # write a "prepare" stage snapshot (optional)
        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.prepare",
                payload={
                    "started_at": timing["started_at"],
                    "inputs": {
                        "base_dir": base_dir,
                        "vectors_path": vectors_path,
                        "d": int(d),
                        "next_out_dir": next_out_dir,
                        "indices_u32_path": indices_u32_path,
                        "offsets_path": offsets_path,
                    },
                    "params": {
                        "vectors_format": str(vectors_format),
                        "npy_cast_dtype": npy_cast_dtype,
                        "N_for_raw_vectors": N_for_raw_vectors,
                        "out_codes_dtype": str(getattr(out_codes_dtype, "__name__", out_codes_dtype)),
                        "pos_block": int(pos_block),
                        "io_buffer_mb": int(io_buffer_mb),
                        "out_codes_path": out_codes_path,
                        "out_ids_path": out_ids_path,
                        "out_offsets_bin": out_offsets_bin,
                        "out_bucket_sizes_bin": out_bucket_sizes_bin,
                        "out_manifest": out_manifest,
                    },
                },
            )

        # ---------------- build csr timing ----------------
        t0 = time.time()
        out = build_csr_artifacts(cfg)
        t1 = time.time()
        timing["build_csr_sec"] = float(t1 - t0)

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.build",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
                    "run_time_sec": float(timing["build_csr_sec"]),
                    "summary": (out if isinstance(out, dict) else {"result": out}),
                },
            )

        # ---------------- finalize timing ----------------
        t_all1 = time.time()
        timing["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all1))
        timing["run_time_sec"] = float(t_all1 - t_all0)

        # Attach timing into output for logging/JSON.
        if isinstance(out, dict):
            out["timing"] = timing
        else:
            out = {"result": out, "timing": timing}

        if verbose:
            print("[csr] done. manifest_path =", out.get("manifest_path"))
            print(f"[csr] timing: build_csr_sec={timing['build_csr_sec']:.3f}s  total={timing['run_time_sec']:.3f}s")

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=stage_name,
                payload={
                    "started_at": timing["started_at"],
                    "finished_at": timing["finished_at"],
                    "run_time_sec": timing["run_time_sec"],
                    "inputs": {
                        "base_dir": base_dir,
                        "vectors_path": vectors_path,
                        "d": int(d),
                        "next_out_dir": next_out_dir,
                    },
                    "outputs": {
                        "manifest_path": out.get("manifest_path"),
                        "out_codes_path": out.get("out_codes_path"),
                        "out_ids_path": out.get("out_ids_path"),
                        "out_offsets_bin": out.get("out_offsets_bin"),
                        "out_bucket_sizes_bin": out.get("out_bucket_sizes_bin"),
                    },
                    "timing_breakdown": timing,
                },
            )

        return out
    
#     def build_subcodes_from_next_dir(
#     self,
#     *,
#     base_dir: str,
#     vectors_path: str,
#     d: int,
#     next_out_dir: str,

#     # -------- database / loader options (match your script) --------
#     vectors_format: str = "auto",          # "auto" | "npy_any" | "u8_auto" | "f32_auto" | "raw_f32bin"
#     npy_cast_dtype: str | None = None,     # only for vectors_format="npy_any"
#     N_for_raw_vectors: int | None = None,  # only required for vectors_format in {"raw_f32bin","u8_auto"}

#     # -------- outputs for runtime --------
#     out_codes_dtype=np.float32,            # np.uint8 or np.float32 ...
#     pos_block: int = 2_000_000,
#     io_buffer_mb: int = 16,

#     # -------- filenames under next_out_dir (subCSR inputs) --------
#     sub_ids_name: str = "sub_ids.i64.bin",
#     sub_offsets_name: str = "sub_offsets.u64.bin",

#     # -------- optional override output paths --------
#     out_sub_codes_path: str | None = None,
#     out_manifest: str | None = None,

#     # (optional) also override other cluster-csr outputs
#     out_sub_ids_path: str | None = None,
#     out_sub_offsets_bin: str | None = None,
#     out_sub_sizes_bin: str | None = None,
#     out_big_sub_ids_bin: str | None = None,

#     # big-sub selection (same concept as big bucket ids)
#     big_bucket_threshold: int = 200_000,
#     big_bucket_ids_dtype=np.int32,
#     bucket_sizes_dtype=np.uint32,

#     verbose: bool = True,

#     # ---------- stage logging ----------
#     stage_json_path: str | None = None,
#     stage_name: str = "sub_codes_build",
# ) -> dict:
#         """
#         Cluster-CSR builder (sub_codes) driven by subCSR in next_out_dir.

#         Required in next_out_dir:
#         - sub_ids.i64.bin
#         - sub_offsets.u64.bin

#         Output (under base_dir by default, or overridden):
#         - sub_codes.<dtype>.csr.bin
#         - sub_ids.int64.csr.bin
#         - sub_offsets.uint64.bin
#         - sub_sizes.bin
#         - big_sub_ids.bin (optional; produced if offsets available, always here)
#         - subcsr_manifest.json (or overridden out_manifest)

#         This delegates to csr_build_v2.build_csr_artifacts(cfg) with cfg.csr_kind="cluster".
#         """

        
        
#         # ---------------- overall timing ----------------
#         t_all0 = time.time()
#         timing = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all0))}

#         sub_ids_path = os.path.join(next_out_dir, sub_ids_name)
#         sub_offsets_path = os.path.join(next_out_dir, sub_offsets_name)

#         if not os.path.isfile(sub_ids_path):
#             raise FileNotFoundError(f"[subcsr] missing sub_ids file: {sub_ids_path}")
#         if not os.path.isfile(sub_offsets_path):
#             raise FileNotFoundError(f"[subcsr] missing sub_offsets file: {sub_offsets_path}")

#         os.makedirs(base_dir, exist_ok=True)

#         # default output paths (match csr_build_v2 naming)
#         dt = np.dtype(out_codes_dtype)
#         suffix = "f32" if dt == np.float32 else ("u8" if dt == np.uint8 else dt.name)

#         if out_sub_codes_path is None:
#             out_sub_codes_path = os.path.join(base_dir, f"sub_codes.{suffix}.csr.bin")
#         if out_sub_ids_path is None:
#             out_sub_ids_path = os.path.join(base_dir, "sub_ids.int64.csr.bin")
#         if out_sub_offsets_bin is None:
#             out_sub_offsets_bin = os.path.join(base_dir, "sub_offsets.uint64.bin")
#         if out_sub_sizes_bin is None:
#             out_sub_sizes_bin = os.path.join(base_dir, "sub_sizes.bin")
#         if out_big_sub_ids_bin is None:
#             out_big_sub_ids_bin = os.path.join(base_dir, "big_sub_ids.bin")
#         if out_manifest is None:
#             out_manifest = os.path.join(base_dir, "subcsr_manifest.json")

#         if verbose:
#             print("[subcsr] vectors_path      =", vectors_path)
#             print("[subcsr] vectors_format    =", vectors_format)
#             print("[subcsr] d                 =", d)
#             print("[subcsr] next_out_dir       =", next_out_dir)
#             print("[subcsr] sub_ids            =", sub_ids_path)
#             print("[subcsr] sub_offsets        =", sub_offsets_path)
#             print("[subcsr] out_sub_codes      =", out_sub_codes_path)
#             print("[subcsr] out_sub_ids        =", out_sub_ids_path)
#             print("[subcsr] out_sub_offsets    =", out_sub_offsets_bin)
#             print("[subcsr] out_sub_sizes      =", out_sub_sizes_bin)
#             print("[subcsr] out_big_sub_ids    =", out_big_sub_ids_bin)
#             print("[subcsr] out_manifest       =", out_manifest)
#             print("[subcsr] out_codes_dtype    =", str(dt))
#             print("[subcsr] pos_block          =", int(pos_block), "io_buffer_mb=", int(io_buffer_mb))
#             print("[subcsr] big_threshold      =", int(big_bucket_threshold))

#         # stage prepare snapshot (guard None)
#         if stage_json_path:
#             _update_stage_json(
#                 stage_json_path,
#                 stage=f"{stage_name}.prepare",
#                 payload={
#                     "started_at": timing["started_at"],
#                     "inputs": {
#                         "base_dir": base_dir,
#                         "vectors_path": vectors_path,
#                         "d": int(d),
#                         "next_out_dir": next_out_dir,
#                         "sub_ids_path": sub_ids_path,
#                         "sub_offsets_path": sub_offsets_path,
#                     },
#                     "params": {
#                         "vectors_format": str(vectors_format),
#                         "npy_cast_dtype": npy_cast_dtype,
#                         "N_for_raw_vectors": N_for_raw_vectors,
#                         "out_codes_dtype": str(dt),
#                         "pos_block": int(pos_block),
#                         "io_buffer_mb": int(io_buffer_mb),
#                         "out_sub_codes_path": out_sub_codes_path,
#                         "out_sub_ids_path": out_sub_ids_path,
#                         "out_sub_offsets_bin": out_sub_offsets_bin,
#                         "out_sub_sizes_bin": out_sub_sizes_bin,
#                         "out_big_sub_ids_bin": out_big_sub_ids_bin,
#                         "out_manifest": out_manifest,
#                         "big_bucket_threshold": int(big_bucket_threshold),
#                     },
#                 },
#             )

#         # ---------------- build via csr_build_v2 (cluster mode) ----------------
#         t0 = time.time()

#         cfg = CSRBuildConfig(
#             base_dir=base_dir,
#             vectors_path=vectors_path,
#             d=int(d),

#             # bucket CSR inputs not used in cluster mode
#             indices_u32_path=None,
#             offsets_path=None,

#             # loader
#             vectors_format=str(vectors_format),
#             npy_cast_dtype=npy_cast_dtype,
#             N_for_raw_vectors=N_for_raw_vectors,

#             # output dtype + perf knobs
#             out_codes_dtype=out_codes_dtype,
#             pos_block=int(pos_block),
#             io_buffer_mb=int(io_buffer_mb),

#             # -------- cluster CSR mode --------
#             csr_kind="cluster",
#             sub_ids_path=sub_ids_path,
#             sub_offsets_path=sub_offsets_path,
#             sub_codes_path=None,  

#             # output overrides (so build_csr_artifacts writes to your chosen names)
#             out_codes_path=out_sub_codes_path,
#             out_ids_path=out_sub_ids_path,
#             out_offsets_bin=out_sub_offsets_bin,
#             out_bucket_sizes_bin=out_sub_sizes_bin,
#             out_big_bucket_ids_bin=out_big_sub_ids_bin,
#             out_manifest=out_manifest,

#             # metadata params
#             bucket_sizes_dtype=bucket_sizes_dtype,
#             big_bucket_ids_dtype=big_bucket_ids_dtype,
#             big_bucket_threshold=int(big_bucket_threshold),
#         )

#         out = build_csr_artifacts(cfg)

#         t1 = time.time()
#         timing["build_sub_codes_sec"] = float(t1 - t0)

#         # stage build snapshot
#         if stage_json_path:
#             _update_stage_json(
#                 stage_json_path,
#                 stage=f"{stage_name}.build",
#                 payload={
#                     "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
#                     "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
#                     "run_time_sec": float(timing["build_sub_codes_sec"]),
#                     "summary": out,
#                 },
#             )

#         # ---------------- finalize timing ----------------
#         t_all1 = time.time()
#         timing["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all1))
#         timing["run_time_sec"] = float(t_all1 - t_all0)

#         # normalize output fields to match your old return contract
#         ret = {
#             "manifest_path": out.get("manifest_path", out_manifest),
#             "out_sub_codes_path": out.get("codes_path", out_sub_codes_path),
#             "out_sub_ids_path": out.get("ids_path", out_sub_ids_path),
#             "out_sub_offsets_bin": out.get("offsets_bin_path", out_sub_offsets_bin),
#             "out_sub_sizes_bin": out.get("bucket_sizes_path", out_sub_sizes_bin),
#             "out_big_sub_ids_bin": out.get("big_bucket_ids_path", out_big_sub_ids_bin),
#             "Nsub": int(out.get("N", 0)),
#             "nsub": int(out.get("nlist") or 0),
#             "d": int(d),
#             "dtype": str(dt),
#             "timing": timing,
#         }

#         if verbose:
#             print("[subcsr] done. manifest_path =", ret["manifest_path"])
#             print(f"[subcsr] timing: build_sub_codes_sec={timing['build_sub_codes_sec']:.3f}s  total={timing['run_time_sec']:.3f}s")

#         if stage_json_path:
#             _update_stage_json(
#                 stage_json_path,
#                 stage=stage_name,
#                 payload={
#                     "started_at": timing["started_at"],
#                     "finished_at": timing["finished_at"],
#                     "run_time_sec": timing["run_time_sec"],
#                     "outputs": {
#                         "manifest_path": ret["manifest_path"],
#                         "out_sub_codes_path": ret["out_sub_codes_path"],
#                     },
#                     "timing_breakdown": timing,
#                 },
#             )

#         return ret
    
#     def build_subcodes_from_next_dir(
#     self,
#     *,
#     base_dir: str,
#     vectors_path: str,
#     d: int,
#     next_out_dir: str,

#     # -------- database / loader options --------
#     vectors_format: str = "auto",          # "auto" | "npy_any" | "u8_auto" | "f32_auto" | "raw_f32bin"
#     npy_cast_dtype: str | None = None,     # only for vectors_format="npy_any"
#     N_for_raw_vectors: int | None = None,  # only required for vectors_format in {"raw_f32bin","u8_auto"}

#     # -------- outputs for runtime --------
#     out_codes_dtype=np.float32,            # np.uint8 or np.float32 ...
#     pos_block: int = 2_000_000,
#     io_buffer_mb: int = 16,

#     # -------- filenames under next_out_dir (subCSR inputs) --------
#     sub_ids_name: str = "sub_ids.i64.bin",
#     sub_offsets_name: str = "sub_offsets.u64.bin",

#     # -------- optional override output paths --------
#     out_sub_codes_path: str | None = None,
#     out_manifest: str | None = None,

#     # (optional) also override other cluster-csr outputs
#     out_sub_ids_path: str | None = None,
#     out_sub_offsets_bin: str | None = None,
#     out_sub_sizes_bin: str | None = None,

#     verbose: bool = True,

#     # ---------- stage logging ----------
#     stage_json_path: str | None = None,
#     stage_name: str = "sub_codes_build",
# ) -> dict:
#         """
#         Cluster-CSR builder (sub_codes) driven by subCSR in next_out_dir.

#         Required in next_out_dir:
#         - sub_ids.i64.bin
#         - sub_offsets.u64.bin

#         Output:
#         - sub_codes.<dtype>.csr.bin
#         - sub_ids.int64.csr.bin
#         - sub_offsets.uint64.bin
#         - sub_sizes.bin
#         - subcsr_manifest.json (or overridden out_manifest)

#         NOTE: big_sub_ids/bin + manifest.big_buckets are intentionally DISABLED here.
#         """

#         t_all0 = time.time()
#         timing = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all0))}

#         sub_ids_path = os.path.join(next_out_dir, sub_ids_name)
#         sub_offsets_path = os.path.join(next_out_dir, sub_offsets_name)

#         if not os.path.isfile(sub_ids_path):
#             raise FileNotFoundError(f"[subcsr] missing sub_ids file: {sub_ids_path}")
#         if not os.path.isfile(sub_offsets_path):
#             raise FileNotFoundError(f"[subcsr] missing sub_offsets file: {sub_offsets_path}")

#         os.makedirs(base_dir, exist_ok=True)

#         dt = np.dtype(out_codes_dtype)
#         suffix = "f32" if dt == np.float32 else ("u8" if dt == np.uint8 else dt.name)

#         if out_sub_codes_path is None:
#             out_sub_codes_path = os.path.join(base_dir, f"sub_codes.{suffix}.csr.bin")
#         if out_sub_ids_path is None:
#             out_sub_ids_path = os.path.join(base_dir, "sub_ids.int64.csr.bin")
#         if out_sub_offsets_bin is None:
#             out_sub_offsets_bin = os.path.join(base_dir, "sub_offsets.uint64.bin")
#         if out_sub_sizes_bin is None:
#             out_sub_sizes_bin = os.path.join(base_dir, "sub_sizes.bin")
#         if out_manifest is None:
#             out_manifest = os.path.join(base_dir, "subcsr_manifest.json")

#         if verbose:
#             print("[subcsr] vectors_path      =", vectors_path)
#             print("[subcsr] vectors_format    =", vectors_format)
#             print("[subcsr] d                 =", d)
#             print("[subcsr] next_out_dir       =", next_out_dir)
#             print("[subcsr] sub_ids            =", sub_ids_path)
#             print("[subcsr] sub_offsets        =", sub_offsets_path)
#             print("[subcsr] out_sub_codes      =", out_sub_codes_path)
#             print("[subcsr] out_sub_ids        =", out_sub_ids_path)
#             print("[subcsr] out_sub_offsets    =", out_sub_offsets_bin)
#             print("[subcsr] out_sub_sizes      =", out_sub_sizes_bin)
#             print("[subcsr] out_manifest       =", out_manifest)
#             print("[subcsr] out_codes_dtype    =", str(dt))
#             print("[subcsr] pos_block          =", int(pos_block), "io_buffer_mb=", int(io_buffer_mb))

#         if stage_json_path:
#             _update_stage_json(
#                 stage_json_path,
#                 stage=f"{stage_name}.prepare",
#                 payload={
#                     "started_at": timing["started_at"],
#                     "inputs": {
#                         "base_dir": base_dir,
#                         "vectors_path": vectors_path,
#                         "d": int(d),
#                         "next_out_dir": next_out_dir,
#                         "sub_ids_path": sub_ids_path,
#                         "sub_offsets_path": sub_offsets_path,
#                     },
#                     "params": {
#                         "vectors_format": str(vectors_format),
#                         "npy_cast_dtype": npy_cast_dtype,
#                         "N_for_raw_vectors": N_for_raw_vectors,
#                         "out_codes_dtype": str(dt),
#                         "pos_block": int(pos_block),
#                         "io_buffer_mb": int(io_buffer_mb),
#                         "out_sub_codes_path": out_sub_codes_path,
#                         "out_sub_ids_path": out_sub_ids_path,
#                         "out_sub_offsets_bin": out_sub_offsets_bin,
#                         "out_sub_sizes_bin": out_sub_sizes_bin,
#                         "out_manifest": out_manifest,
#                         "big_disabled": True,
#                     },
#                 },
#             )

#         # ---------------- build via csr_build_v2 (cluster mode) ----------------
#         t0 = time.time()

#         cfg = CSRBuildConfig(
#             base_dir=base_dir,
#             vectors_path=vectors_path,
#             d=int(d),

#             # bucket CSR inputs not used in cluster mode
#             indices_u32_path=None,
#             offsets_path=None,

#             # loader
#             vectors_format=str(vectors_format),
#             npy_cast_dtype=npy_cast_dtype,
#             N_for_raw_vectors=N_for_raw_vectors,

#             # output dtype + perf knobs
#             out_codes_dtype=out_codes_dtype,
#             pos_block=int(pos_block),
#             io_buffer_mb=int(io_buffer_mb),

#             # -------- cluster CSR mode --------
#             csr_kind="cluster",
#             sub_ids_path=sub_ids_path,
#             sub_offsets_path=sub_offsets_path,
#             sub_codes_path=None,

#             # output overrides
#             out_codes_path=out_sub_codes_path,
#             out_ids_path=out_sub_ids_path,
#             out_offsets_bin=out_sub_offsets_bin,
#             out_bucket_sizes_bin=out_sub_sizes_bin,
#             out_manifest=out_manifest,

#             # -------- disable big buckets --------
#             out_big_bucket_ids_bin=None,
#             big_bucket_threshold=0,
#             big_bucket_ids_dtype=None,
#             bucket_sizes_dtype=np.uint32,
#         )

#         out = build_csr_artifacts(cfg)

#         t1 = time.time()
#         timing["build_sub_codes_sec"] = float(t1 - t0)

#         if stage_json_path:
#             _update_stage_json(
#                 stage_json_path,
#                 stage=f"{stage_name}.build",
#                 payload={
#                     "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
#                     "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
#                     "run_time_sec": float(timing["build_sub_codes_sec"]),
#                     "summary": out,
#                 },
#             )

#         t_all1 = time.time()
#         timing["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all1))
#         timing["run_time_sec"] = float(t_all1 - t_all0)

#         ret = {
#             "manifest_path": out.get("manifest_path", out_manifest),
#             "out_sub_codes_path": out.get("codes_path", out_sub_codes_path),
#             "out_sub_ids_path": out.get("ids_path", out_sub_ids_path),
#             "out_sub_offsets_bin": out.get("offsets_bin_path", out_sub_offsets_bin),
#             "out_sub_sizes_bin": out.get("bucket_sizes_path", out_sub_sizes_bin),
#             "Nsub": int(out.get("N", 0)),
#             "nsub": int(out.get("nlist") or 0),
#             "d": int(d),
#             "dtype": str(dt),
#             "timing": timing,
#             "big_disabled": True,
#         }

#         if verbose:
#             print("[subcsr] done. manifest_path =", ret["manifest_path"])
#             print(f"[subcsr] timing: build_sub_codes_sec={timing['build_sub_codes_sec']:.3f}s  total={timing['run_time_sec']:.3f}s")

#         if stage_json_path:
#             _update_stage_json(
#                 stage_json_path,
#                 stage=stage_name,
#                 payload={
#                     "started_at": timing["started_at"],
#                     "finished_at": timing["finished_at"],
#                     "run_time_sec": timing["run_time_sec"],
#                     "outputs": {
#                         "manifest_path": ret["manifest_path"],
#                         "out_sub_codes_path": ret["out_sub_codes_path"],
#                     },
#                     "timing_breakdown": timing,
#                 },
#             )

#         return ret
    def build_subcodes_from_next_dir(
        self,
        *,
        base_dir: str,
        vectors_path: str,
        d: int,
        next_out_dir: str,

        # -------- database / loader options --------
        vectors_format: str = "auto",          # "auto" | "npy_any" | "u8_auto" | "f32_auto" | "raw_f32bin"
        npy_cast_dtype: str | None = None,     # only for vectors_format="npy_any"
        N_for_raw_vectors: int | None = None,  # only required for vectors_format in {"raw_f32bin","u8_auto"}

        # -------- outputs for runtime --------
        out_codes_dtype=None,                  # np.uint8 / np.float32; default keeps your old behavior if you pass it
        pos_block: int = 2_000_000,
        io_buffer_mb: int = 16,

        # -------- filenames under next_out_dir (subCSR inputs) --------
        sub_ids_name: str = "sub_ids.i64.bin",
        sub_offsets_name: str = "sub_offsets.u64.bin",

        # -------- optional override output paths --------
        out_sub_codes_path: str | None = None,
        out_manifest: str | None = None,

        # (optional) also override other cluster-csr outputs
        out_sub_ids_path: str | None = None,
        out_sub_offsets_bin: str | None = None,
        out_sub_sizes_bin: str | None = None,

        # NEW: copy bucket<->cluster sidecars from next_out_dir/subcsr to base_dir root
        copy_subcsr_sidecars_to_root: int = 1,

        verbose: bool = True,

        # ---------- stage logging ----------
        stage_json_path: str | None = None,
        stage_name: str = "sub_codes_build",
    ) -> dict:
        """
        Cluster-CSR builder (sub_codes) driven by subCSR in next_out_dir.

        Required in next_out_dir:
        - sub_ids.i64.bin
        - sub_offsets.u64.bin

        Output:
        - sub_codes.<dtype>.csr.bin
        - sub_ids.int64.csr.bin
        - sub_offsets.uint64.bin
        - sub_sizes.bin
        - subcsr_manifest.json (or overridden out_manifest)

        NOTE:
        - big_sub_ids/bin + manifest.big_buckets are intentionally DISABLED here.
        - Optionally copies subcsr sidecars (bucket<->cluster mapping) from next_out_dir/subcsr/ into base_dir root,
        and refreshes manifest.sidecars.
        """

        # ---------------- overall timing ----------------
        t_all0 = time.time()
        timing = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all0))}

        next_out_dir = str(next_out_dir)
        base_dir = str(base_dir)

        sub_ids_path = os.path.join(next_out_dir, sub_ids_name)
        sub_offsets_path = os.path.join(next_out_dir, sub_offsets_name)

        if not os.path.isfile(sub_ids_path):
            raise FileNotFoundError(f"[subcsr] missing sub_ids file: {sub_ids_path}")
        if not os.path.isfile(sub_offsets_path):
            raise FileNotFoundError(f"[subcsr] missing sub_offsets file: {sub_offsets_path}")

        os.makedirs(base_dir, exist_ok=True)

        # dtype handling
        if out_codes_dtype is None:
            # keep old default if caller didn't pass it; adjust if you want a different default
            out_codes_dtype = np.float32
        dt = np.dtype(out_codes_dtype)
        suffix = "f32" if dt == np.float32 else ("u8" if dt == np.uint8 else dt.name)

        # default output paths (match csr_build_v2 naming)
        if out_sub_codes_path is None:
            out_sub_codes_path = os.path.join(base_dir, f"sub_codes.{suffix}.csr.bin")
        if out_sub_ids_path is None:
            out_sub_ids_path = os.path.join(base_dir, "sub_ids.int64.csr.bin")
        if out_sub_offsets_bin is None:
            out_sub_offsets_bin = os.path.join(base_dir, "sub_offsets.uint64.bin")
        if out_sub_sizes_bin is None:
            out_sub_sizes_bin = os.path.join(base_dir, "sub_sizes.bin")
        if out_manifest is None:
            out_manifest = os.path.join(base_dir, "subcsr_manifest.json")

        if verbose:
            print("[subcsr] vectors_path      =", vectors_path)
            print("[subcsr] vectors_format    =", vectors_format)
            print("[subcsr] d                 =", d)
            print("[subcsr] next_out_dir       =", next_out_dir)
            print("[subcsr] sub_ids            =", sub_ids_path)
            print("[subcsr] sub_offsets        =", sub_offsets_path)
            print("[subcsr] out_sub_codes      =", out_sub_codes_path)
            print("[subcsr] out_sub_ids        =", out_sub_ids_path)
            print("[subcsr] out_sub_offsets    =", out_sub_offsets_bin)
            print("[subcsr] out_sub_sizes      =", out_sub_sizes_bin)
            print("[subcsr] out_manifest       =", out_manifest)
            print("[subcsr] out_codes_dtype    =", str(dt))
            print("[subcsr] pos_block          =", int(pos_block), "io_buffer_mb=", int(io_buffer_mb))
            print("[subcsr] big_disabled       = True")

        # stage prepare snapshot (guard None)
        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.prepare",
                payload={
                    "started_at": timing["started_at"],
                    "inputs": {
                        "base_dir": base_dir,
                        "vectors_path": vectors_path,
                        "d": int(d),
                        "next_out_dir": next_out_dir,
                        "sub_ids_path": sub_ids_path,
                        "sub_offsets_path": sub_offsets_path,
                    },
                    "params": {
                        "vectors_format": str(vectors_format),
                        "npy_cast_dtype": npy_cast_dtype,
                        "N_for_raw_vectors": N_for_raw_vectors,
                        "out_codes_dtype": str(dt),
                        "pos_block": int(pos_block),
                        "io_buffer_mb": int(io_buffer_mb),
                        "out_sub_codes_path": out_sub_codes_path,
                        "out_sub_ids_path": out_sub_ids_path,
                        "out_sub_offsets_bin": out_sub_offsets_bin,
                        "out_sub_sizes_bin": out_sub_sizes_bin,
                        "out_manifest": out_manifest,
                        "big_disabled": True,
                        "copy_subcsr_sidecars_to_root": int(copy_subcsr_sidecars_to_root),
                    },
                },
            )

        # ---------------- build via csr_build_v2 (cluster mode) ----------------
        t0 = time.time()

        cfg = CSRBuildConfig(
            base_dir=base_dir,
            vectors_path=vectors_path,
            d=int(d),

            # bucket CSR inputs not used in cluster mode
            indices_u32_path=None,
            offsets_path=None,

            # loader
            vectors_format=str(vectors_format),
            npy_cast_dtype=npy_cast_dtype,
            N_for_raw_vectors=N_for_raw_vectors,

            # output dtype + perf knobs
            out_codes_dtype=out_codes_dtype,
            pos_block=int(pos_block),
            io_buffer_mb=int(io_buffer_mb),

            # -------- cluster CSR mode --------
            csr_kind="cluster",
            sub_ids_path=sub_ids_path,
            sub_offsets_path=sub_offsets_path,
            sub_codes_path=None,

            # output overrides
            out_codes_path=out_sub_codes_path,
            out_ids_path=out_sub_ids_path,
            out_offsets_bin=out_sub_offsets_bin,
            out_bucket_sizes_bin=out_sub_sizes_bin,
            out_manifest=out_manifest,

            # -------- disable big buckets --------
            out_big_bucket_ids_bin=None,
            big_bucket_threshold=0,
            big_bucket_ids_dtype=None,

            bucket_sizes_dtype=np.uint32,
        )

        out = build_csr_artifacts(cfg)

        t1 = time.time()
        timing["build_sub_codes_sec"] = float(t1 - t0)

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=f"{stage_name}.build",
                payload={
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
                    "run_time_sec": float(timing["build_sub_codes_sec"]),
                    "summary": out,
                },
            )

        # ---------------- copy subcsr sidecars into base_dir ----------------
        # Produced by bucket_cluster_pipeline under: next_out_dir/subcsr/
        if int(copy_subcsr_sidecars_to_root) != 0:
            subcsr_dir = Path(next_out_dir)
            if subcsr_dir.is_dir():
                sidecars = [
                    "sub_bucket_id.u32.bin",        # cluster -> bucket
                    "bucket_sub_offsets.u64.bin",   # bucket -> cluster range
                    "sub_centroids.f32.bin",        # cluster centroids
                    "sub_cluster_id.u16.bin",       # optional per-vector local cluster id
                    "sub_cluster_id.u32.bin",       # optional alt dtype
                ]
                copied = []
                for name in sidecars:
                    src = subcsr_dir / name
                    if src.is_file():
                        dst = Path(base_dir) / name
                        try:
                            shutil.copy2(src, dst)  # overwrite OK
                            copied.append(name)
                            if verbose:
                                print(f"[subcsr][copy] {src} -> {dst}")
                        except Exception as e:
                            print(f"[warn] failed to copy {src} -> {dst}: {e}")
                    else:
                        if verbose:
                            print(f"[subcsr][copy-skip] missing: {src}")

                # refresh manifest.sidecars so downstream runtime can discover them from manifest
                if copied and os.path.isfile(out_manifest):
                    try:
                        with open(out_manifest, "r", encoding="utf-8") as f:
                            mani = json.load(f)
                        sc = mani.get("sidecars")
                        if not isinstance(sc, dict):
                            mani["sidecars"] = {}

                        p = Path(base_dir)

                        if (p / "sub_bucket_id.u32.bin").is_file():
                            mani["sidecars"]["sub_bucket_id"] = {
                                "path": str(p / "sub_bucket_id.u32.bin"),
                                "dtype": "uint32",
                                "layout": "row_major",
                            }
                        if (p / "bucket_sub_offsets.u64.bin").is_file():
                            mani["sidecars"]["bucket_sub_offsets"] = {
                                "path": str(p / "bucket_sub_offsets.u64.bin"),
                                "dtype": "uint64",
                                "layout": "row_major",
                            }
                        if (p / "sub_centroids.f32.bin").is_file():
                            mani["sidecars"]["sub_centroids"] = {
                                "path": str(p / "sub_centroids.f32.bin"),
                                "dtype": "float32",
                                "layout": "row_major",
                            }
                        if (p / "sub_cluster_id.u16.bin").is_file():
                            mani["sidecars"]["sub_cluster_id"] = {
                                "path": str(p / "sub_cluster_id.u16.bin"),
                                "dtype": "uint16",
                                "layout": "row_major",
                            }
                        elif (p / "sub_cluster_id.u32.bin").is_file():
                            mani["sidecars"]["sub_cluster_id"] = {
                                "path": str(p / "sub_cluster_id.u32.bin"),
                                "dtype": "uint32",
                                "layout": "row_major",
                            }

                        with open(out_manifest, "w", encoding="utf-8") as f:
                            json.dump(mani, f, indent=2)
                        if verbose:
                            print(f"[subcsr] refreshed manifest sidecars: {out_manifest}")
                    except Exception as e:
                        print(f"[warn] failed to refresh manifest sidecars: {e}")
            else:
                if verbose:
                    print(f"[subcsr][copy-skip] no subcsr dir: {subcsr_dir}")

        # ---------------- finalize timing ----------------
        t_all1 = time.time()
        timing["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_all1))
        timing["run_time_sec"] = float(t_all1 - t_all0)

        ret = {
            "manifest_path": out.get("manifest_path", out_manifest),
            "out_sub_codes_path": out.get("codes_path", out_sub_codes_path),
            "out_sub_ids_path": out.get("ids_path", out_sub_ids_path),
            "out_sub_offsets_bin": out.get("offsets_bin_path", out_sub_offsets_bin),
            "out_sub_sizes_bin": out.get("bucket_sizes_path", out_sub_sizes_bin),
            "Nsub": int(out.get("N", 0)),
            "nsub": int(out.get("nlist") or 0),
            "d": int(d),
            "dtype": str(dt),
            "timing": timing,
            "big_disabled": True,
        }

        if verbose:
            print("[subcsr] done. manifest_path =", ret["manifest_path"])
            print(f"[subcsr] timing: build_sub_codes_sec={timing['build_sub_codes_sec']:.3f}s  total={timing['run_time_sec']:.3f}s")

        if stage_json_path:
            _update_stage_json(
                stage_json_path,
                stage=stage_name,
                payload={
                    "started_at": timing["started_at"],
                    "finished_at": timing["finished_at"],
                    "run_time_sec": timing["run_time_sec"],
                    "outputs": {
                        "manifest_path": ret["manifest_path"],
                        "out_sub_codes_path": ret["out_sub_codes_path"],
                    },
                    "timing_breakdown": timing,
                },
            )

        return ret


    def prepare_cpp_inputs_and_run(
            self,
            *,
            next_out_dir: str,
            vec_npy: str,
            D: int,
            exe: str = "./bucket_cluster_pipeline_v6_8_gt",
            out_dir: str | None = None,
            out_sub_dir: str | None = None,

            # input filenames in next_out_dir (from build_index/evaluate stage)
            indices_u32_name: str = "indices.uint32.bin",
            offsets_u64_npy_name: str = "offsets.uint64.npy",

            # generated filenames (written under next_out_dir unless overridden)
            offsets_u64_bin_name: str = "offsets.uint64.bin",
            ids_i64_csr_bin_name: str = "ids.int64.csr.bin",

            # performance / io
            chunk_elems: int = 50_000_000,  # number of ids per chunk when converting (tune for memory)
            threads: int = 24,
            blas_threads: int = 1,

            # cpp flags
            stage_shm: int = 0,
            emit_bucketed_vectors: int = 0,
            emit_bucketed_ids: int = 1,
            emit_tiny_centroid: int = 0,

            # NEW: save bucket<->cluster sidecars into next_out_dir root as well
            copy_subcsr_sidecars_to_root: int = 1,

            verbose: bool = True,
        ):
        """
        1) Read next_out_dir/offsets.uint64.npy -> write next_out_dir/offsets.uint64.bin
        2) Read next_out_dir/indices.uint32.bin -> write next_out_dir/ids.int64.csr.bin (int64 cast)
        3) Compute nlist from offsets (len-1)
        4) Run bucket_cluster_pipeline_v6_8_gt with these prepared inputs
        5) (optional) Copy subcsr sidecar files back to next_out_dir root so results packing won't miss them

        Returns dict with paths + nlist.
        """

        next_out_dir = str(next_out_dir)
        in_dir = Path(next_out_dir)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"next_out_dir not found: {next_out_dir}")

        indices_u32_path = in_dir / indices_u32_name
        offsets_npy_path = in_dir / offsets_u64_npy_name
        if not indices_u32_path.is_file():
            raise FileNotFoundError(f"missing: {indices_u32_path}")
        if not offsets_npy_path.is_file():
            raise FileNotFoundError(f"missing: {offsets_npy_path}")

        offsets_bin_path = in_dir / offsets_u64_bin_name
        ids_i64_path = in_dir / ids_i64_csr_bin_name

        # ---------- (1) offsets npy -> raw bin ----------
        offsets = np.load(offsets_npy_path, mmap_mode="r")
        offsets = np.asarray(offsets, dtype=np.uint64).reshape(-1)
        if offsets.size < 2:
            raise ValueError(f"bad offsets: size={offsets.size}")
        nlist = int(offsets.size - 1)
        N = int(offsets[-1])

        if verbose:
            print("[prep] offsets_npy =", offsets_npy_path)
            print("[prep] nlist =", nlist, "N =", N)
            print("[prep] writing offsets_bin =", offsets_bin_path)

        offsets.tofile(offsets_bin_path)

        # ---------- (2) indices.u32 -> ids.i64 ----------
        file_bytes = indices_u32_path.stat().st_size
        num_u32 = file_bytes // 4
        if num_u32 != N:
            print(f"[warn] indices_u32 count={num_u32} != offsets[-1]={N} (check inputs)")
            N_use = num_u32
        else:
            N_use = N

        if verbose:
            print("[prep] indices_u32 =", indices_u32_path, "count =", num_u32)
            print("[prep] writing ids_i64 =", ids_i64_path)

        with open(indices_u32_path, "rb") as fin, open(ids_i64_path, "wb") as fout:
            remaining = N_use
            wrote = 0
            while remaining > 0:
                take = min(remaining, chunk_elems)
                buf = np.fromfile(fin, dtype=np.uint32, count=take)
                if buf.size == 0:
                    break
                buf64 = buf.astype(np.int64, copy=False)
                buf64.tofile(fout)
                wrote += int(buf.size)
                remaining -= int(buf.size)
                if verbose and (wrote % (chunk_elems * 10) == 0):
                    print(f"[prep] ids_i64 wrote {wrote}/{N_use} ({100.0*wrote/max(N_use,1):.2f}%)")

        # ---------- (3) run cpp ----------
        if out_dir is None:
            out_dir = str(in_dir / "out")
        if out_sub_dir is None:
            out_sub_dir = str(in_dir / "subcsr")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_sub_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            exe,
            "--vec_npy", str(vec_npy),
            "--D", str(int(D)),
            "--offsets_u64_bin", str(offsets_bin_path),
            "--nlist", str(int(nlist)),
            "--ids_i64_csr_bin", str(ids_i64_path),
            "--out_dir", str(out_dir),
            "--out_sub_dir", str(out_sub_dir),
            "--stage_shm", str(int(stage_shm)),
            "--emit_bucketed_vectors", str(int(emit_bucketed_vectors)),
            "--emit_bucketed_ids", str(int(emit_bucketed_ids)),
            "--emit_tiny_centroid", str(int(emit_tiny_centroid)),
            "--threads", str(int(threads)),
            "--blas_threads", str(int(blas_threads)),
        ]

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(threads))
        env["OPENBLAS_NUM_THREADS"] = str(int(blas_threads))
        env["MKL_NUM_THREADS"] = str(int(blas_threads))

        if verbose:
            print("[cpp] running:", " ".join(cmd))

        # assume you already have this method in your codebase
        _run_streaming(cmd, env=env)

        # ---------- (4) copy subcsr sidecars back to root ----------
        # Prevent losing bucket<->cluster mapping when you later "pack results"
        if int(copy_subcsr_sidecars_to_root) != 0:
            subdir = Path(out_sub_dir)
            if subdir.is_dir():
                sidecars = [
                    "sub_bucket_id.u32.bin",        # cluster -> bucket
                    "bucket_sub_offsets.u64.bin",   # bucket -> cluster range
                    "sub_centroids.f32.bin",        # centroids
                    "sub_cluster_id.u16.bin",       # cluster id per subcluster
                    "sub_cluster_id.u32.bin",       # optional alt dtype
                ]
                for name in sidecars:
                    src = subdir / name
                    if src.is_file():
                        dst = in_dir / name
                        try:
                            shutil.copy2(src, dst)  # overwrite ok
                            if verbose:
                                print(f"[copy] {src} -> {dst}")
                        except Exception as e:
                            print(f"[warn] failed to copy {src} -> {dst}: {e}")
                    else:
                        if verbose:
                            print(f"[copy-skip] missing: {src}")
            else:
                if verbose:
                    print(f"[copy-skip] out_sub_dir not found: {subdir}")

        return {
            "next_out_dir": next_out_dir,
            "nlist": int(nlist),
            "N": int(N_use),
            "offsets_u64_bin": str(offsets_bin_path),
            "ids_i64_csr_bin": str(ids_i64_path),
            "out_dir": str(out_dir),
            "out_sub_dir": str(out_sub_dir),
            "copied_subcsr_sidecars_to_root": bool(int(copy_subcsr_sidecars_to_root) != 0),
        }