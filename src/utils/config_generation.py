import json
import os
import shutil
import math

def to_float_list(vec, round_ndigits=None):
    """Convert list/tuple/np.ndarray/torch.Tensor -> flat list[float] with optional rounding."""
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        import torch
    except ImportError:
        torch = None

    if np is not None and isinstance(vec, np.ndarray):
        lst = vec.reshape(-1).astype(float).tolist()
    elif torch is not None and isinstance(vec, torch.Tensor):
        lst = vec.detach().reshape(-1).cpu().tolist()
    else:
        lst = list(vec)

    out = []
    for v in lst:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"Non-finite value in vector: {v}")
        out.append(fv)

    if round_ndigits is not None:
        out = [round(x, round_ndigits) for x in out]
    return out

# def update_margin_position_vector(json_path, vector, round_ndigits=None, make_backup=False):
#     """
#     Overwrite 'build_index.margin_position' with a VECTOR in a single JSON file.

#     Args:
#         json_path (str): Path to JSON file to update.
#         vector: List/NumPy array/Torch tensor to write.
#         round_ndigits (int|None): If set, round values for smaller JSON.
#         make_backup (bool): If True, save a '.bak' copy before overwriting.

#     Returns:
#         dict: {json_path: updated_json_dict}
#     """
#     if not os.path.isfile(json_path):
#         raise FileNotFoundError(f"Not a file: {json_path}")

#     margin_vec = _to_float_list(vector, round_ndigits=round_ndigits)

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     if "build_index" not in data:
#         raise KeyError(f"'build_index' section not found in {json_path}")

#     if make_backup:
#         with open(json_path + ".bak", "w") as f:
#             json.dump(data, f, indent=2)

#     data["build_index"]["margin_position"] = margin_vec

#     with open(json_path, "w") as f:
#         json.dump(data, f, indent=2)

#     print(f"Updated margin_position (len={len(margin_vec)}) in {json_path}")
#     print(margin_vec)
#     return {json_path: data}



def update_margin_position_vector(
    json_path,
    vector,
    round_ndigits=None,
    make_backup=False,
    *,
    model_path: str | None = None,          # ✅ NEW: optionally overwrite build_index.model_path
    model_selector: str | None = None,      # ✅ NEW: optionally store selection type (e.g. "topk:1")
    score_max: int | None = None,           # ✅ NEW: store score used for selection
    stats: dict | None = None,              # ✅ NEW: store p99/p999/nonempty/empty_ratio etc.
):
    """
    Overwrite 'build_index.margin_position' with a VECTOR in a single JSON file.

    Optional extras:
      - also overwrite build_index.model_path
      - record model selection metadata

    Args:
        json_path (str): Path to JSON file to update.
        vector: List/NumPy array/Torch tensor to write.
        round_ndigits (int|None): If set, round values for smaller JSON.
        make_backup (bool): If True, save a '.bak' copy before overwriting.
        model_path (str|None): if provided, set build_index.model_path
        model_selector (str|None): if provided, set build_index.model_selector
        score_max (int|None): if provided, set build_index.model_select_score_max
        stats (dict|None): if provided, set build_index.model_select_stats

    Returns:
        dict: {json_path: updated_json_dict}
    """
    # if not os.path.isfile(json_path):
    #     raise FileNotFoundError(f"Not a file: {json_path}")

    margin_vec = to_float_list(vector, round_ndigits=round_ndigits)

    with open(json_path, "r") as f:
        data = json.load(f)

    if "build_index" not in data or not isinstance(data["build_index"], dict):
        raise KeyError(f"'build_index' section not found or not dict in {json_path}")

    if make_backup:
        with open(json_path + ".bak", "w") as f:
            json.dump(data, f, indent=2)

    # ---- main update ----
    data["build_index"]["margin_position"] = margin_vec

    # ---- optional updates ----
    if model_path is not None:
        data["build_index"]["model_path"] = str(model_path)

    if model_selector is not None:
        data["build_index"]["model_selector"] = str(model_selector)

    if score_max is not None:
        data["build_index"]["model_select_score_max"] = int(score_max)

    if stats is not None:
        # keep stats json-friendly
        clean_stats = {}
        for k, v in stats.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                clean_stats[k] = v
            else:
                # fallback stringify (should not happen often)
                clean_stats[k] = str(v)
        data["build_index"]["model_select_stats"] = clean_stats

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated margin_position (len={len(margin_vec)}) in {json_path}")
    print("thr[:8] =", margin_vec[:8])

    if model_path is not None:
        print(f"Updated model_path = {model_path}")
    if score_max is not None:
        print(f"Stored selection score_max = {score_max}")

    return {json_path: data}


def _pick_weights_path(source_data, model_selector=None):
    r0 = source_data.get("results", [{}])[0]

    # 1) selector mode
    if model_selector:
        if model_selector == "stable_best":
            p = r0.get("stable_best_path", None)
            if p:
                return p

        if model_selector.startswith("topk:"):
            k = int(model_selector.split(":")[1])
            best_models = r0.get("best_models", [])
            if best_models and 1 <= k <= len(best_models):
                return best_models[k - 1].get("path", None)

        raise ValueError(f"Unknown model_selector={model_selector}")

    # 2) default priority
    if r0.get("stable_best_path"):
        return r0["stable_best_path"]

    best_models = r0.get("best_models", [])
    if best_models and best_models[0].get("path"):
        return best_models[0]["path"]

    # 3) old format fallback
    if r0.get("weights_path"):
        return r0["weights_path"]

    raise KeyError("Cannot find weights path in results json (no stable_best_path/best_models/weights_path).")


def transform_json(
    source_json_path,
    target_json_name,
    base_path,
    vector_dim=96,
    model_selector=None,
    keep_results_for_selection=True,   # ✅ NEW
):
    source_file_name = os.path.splitext(os.path.basename(source_json_path))[0]
    output_folder = os.path.join(base_path, source_file_name)
    os.makedirs(output_folder, exist_ok=True)

    source_copy_path = os.path.join(output_folder, f"{source_file_name}_original.json")
    shutil.copy2(source_json_path, source_copy_path)

    target_json_path = os.path.join(output_folder, target_json_name)

    with open(source_json_path, "r") as f:
        source_data = json.load(f)

    weights_path = _pick_weights_path(source_data, model_selector=model_selector)

    mean_encoded = source_data.get("results", [{}])[0].get("mean_encoded", None)

    # ✅ IMPORTANT: keep original structure so AutoHash can see results/best_models
    target_data = {
        "config": source_data["config"],          # ✅ keep training config
        "results": source_data.get("results", []),# ✅ keep best_models/stable_best_path
        "timestamp": source_data.get("timestamp", None),

        # ---- build/eval part (your existing usage) ----
        "experiment_name": "AutoHash_evaluation",
        "data_path": [
            "/path/to/big-ann-benchmarks/data/deep1b/base.1B.npy"
        ],
        "build_index": {
            "model_path": weights_path,
            "vector_dim": vector_dim,
            "hidden_dim": source_data["config"]["encoding_dim"],
            "margin_position": mean_encoded,
            "batch_size": source_data["config"]["training"]["batch_size"]
        },
    }

    # 如果你不想带 results，就删掉它
    if not keep_results_for_selection:
        target_data.pop("config", None)
        target_data.pop("results", None)
        target_data.pop("timestamp", None)

    with open(target_json_path, "w") as f:
        json.dump(target_data, f, indent=2)

    print(f"Source JSON copied to {source_copy_path}")
    print(f"Transformed JSON saved to {target_json_path}")
    print(f"Selected weights: {weights_path}")

    return target_json_path


if __name__ == "__main__":
    # Example usage
    source_json_path = "/path/to/training_data_root/deep1b_results/results_20250625_153025.json"  # Replace with actual path to source JSON
    target_json_name = "Autohash_config.json"  # Name of the transformed JSON file
    base_path = "/path/to/training_data_root/1B_dataset_experiments/config"  # Replace with the base path of the other project
    vector_dim = 96  # Modify this value as needed
    transform_json(source_json_path, target_json_name, base_path, vector_dim=vector_dim)