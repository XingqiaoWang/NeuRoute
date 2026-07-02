import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC  = os.path.join(ROOT, "src")
BIGANN = os.path.join(SRC, "Bigann")

sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)
sys.path.insert(0, BIGANN)

import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from indexing_model_bigann import Indexing_Model


def load_data(data_path, training_set_ratio):
    """
    Load embeddings from .npy and build TensorDataset splits.

    NOTE:
    - Keep dataset on CPU. Move to GPU per batch in the training loop.
    """
    embeddings = np.load(data_path)
    print(f"[Data] embeddings shape={embeddings.shape} dtype={embeddings.dtype}")

    input_dim = embeddings.shape[1]
    instance_amount = embeddings.shape[0]
    training_set_amount = int(instance_amount * training_set_ratio)

    embeddings_tensor = torch.from_numpy(embeddings).float()

    train_dataset = TensorDataset(embeddings_tensor[:training_set_amount])
    val_dataset = TensorDataset(embeddings_tensor[training_set_amount:])

    return train_dataset, val_dataset, input_dim


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def _state_dict_to_cpu(state_dict):
    """
    Make a CPU copy of a model state_dict to reduce GPU memory pressure.
    """
    cpu_sd = {}
    for k, v in state_dict.items():
        cpu_sd[k] = v.detach().cpu().clone()
    return cpu_sd


def _insert_topk(best_list, candidate, top_k):
    """
    Maintain a sorted Top-K list by val_loss (ascending).
    best_list entries: dict with keys: val_loss, epoch, state_dict_cpu
    """
    best_list.append(candidate)
    best_list.sort(key=lambda x: x["val_loss"])
    if len(best_list) > top_k:
        best_list.pop()  # drop worst


def run_training_loop(config_path):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    encoding_dim = config.get("encoding_dim")

    # Only load what you need
    param = config.get("parameter", {})
    threshold1 = param.get("threshold1")
    gamma = param.get("gamma")
    # Do NOT load/use threshold2 or lambda_1

    training = config.get("training", {})
    data_path = training.get("data_path")
    model_path = training.get("model_path")
    training_set_ratio = training.get("training_set_ratio")
    batch_size = training.get("batch_size")
    num_epochs = training.get("num_epochs")
    lr = training.get("learning_rate")
    validation_interval = training.get("validation_interval")

    # Save Top-K best models
    save_top_k = int(training.get("save_top_k", 5))

    # -------------------------
    # DataLoader knobs (ONLY change here)
    # -------------------------
    num_workers = training.get("num_workers", 8)
    prefetch_factor = training.get("prefetch_factor", 4)
    pin_memory = training.get("pin_memory", True)
    persistent_workers = training.get("persistent_workers", True)

    print(
        f"[DataLoader] batch_size={batch_size} num_workers={num_workers} "
        f"prefetch_factor={prefetch_factor} pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers}"
    )
    print(f"[Checkpoint] save_top_k={save_top_k}")

    # Load data (CPU)
    train_dataset, val_dataset, input_dim = load_data(data_path, training_set_ratio)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )

    # Model
    model = Indexing_Model(input_dim, encoding_dim, metric="euclidean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Keep Top-K best models (by val loss)
    best_topk = []  # list of {val_loss, epoch, state_dict_cpu}

    losses = []
    results = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0

        for batch in train_loader:
            inputs = batch[0].to(device, non_blocking=True)

            optimizer.zero_grad()
            encoded = model(inputs)

            # NOTE: threshold2 removed, lambda_1 removed
            l1 = model.similarity_loss(inputs, encoded, threshold1, gamma)

            loss = l1
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        # Validation
        if epoch % validation_interval == 0:
            model.eval()
            val_total_loss_sum = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    inputs = val_batch[0].to(device, non_blocking=True)
                    encoded = model(inputs)

                    l1 = model.similarity_loss(inputs, encoded, threshold1, gamma)
                    val_total_loss_sum += float(l1.item())

            val_total_loss = val_total_loss_sum / max(1, len(val_loader))

            epoch_results = {
                "epoch": epoch,
                "train_total_loss": total_loss / max(1, len(train_loader)),
                "val_total_loss": val_total_loss,
            }
            print(epoch_results)
            losses.append(epoch_results)

            # Update Top-K list
            candidate = {
                "val_loss": float(val_total_loss),
                "epoch": int(epoch),
                "state_dict_cpu": _state_dict_to_cpu(model.state_dict()),
            }
            _insert_topk(best_topk, candidate, top_k=save_top_k)

    end_time = time.time()
    run_time = end_time - start_time

    os.makedirs(model_path, exist_ok=True)

    # --------------------------------------------------------
    # Weight filename only keeps requested parameters
    # --------------------------------------------------------
    base_prefix = os.path.join(
        model_path,
        f"encoding_dim{encoding_dim}_lr{lr}_thresh{threshold1}_gamma_{gamma}"
    )

    saved_meta = []

    # Save Top-K models (ranked)
    for rank, item in enumerate(best_topk, start=1):
        epoch = item["epoch"]
        vloss = item["val_loss"]

        weight_filename = f"{base_prefix}_best{rank}_epoch{epoch}_valloss{vloss:.6f}.pt"
        torch.save(item["state_dict_cpu"], weight_filename)

        saved_meta.append(
            {"rank": rank, "epoch": epoch, "val_loss": vloss, "path": weight_filename}
        )

    # Save stable "best" model name (exact requested pattern)
    stable_best_filename = f"{base_prefix}.pt"
    if len(best_topk) > 0:
        torch.save(best_topk[0]["state_dict_cpu"], stable_best_filename)
    else:
        torch.save(_state_dict_to_cpu(model.state_dict()), stable_best_filename)

    results.append(
        {
            "run_time": run_time,
            "losses": losses,
            "save_top_k": save_top_k,
            "best_models": saved_meta,
            "stable_best_path": stable_best_filename,
            "learning_rate": lr,
            "dataloader": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
            },
        }
    )
    return results


if __name__ == "__main__":
    from datetime import datetime

    results_dir = "./01192026_results"
    config_path = "./config/experiment_bigann_eu.json"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"results_{timestamp}.json")

    config = json.load(open(config_path, "r"))
    results = run_training_loop(config_path)

    with open(output_file, "w") as f:
        json.dump(
            {"config": config, "results": results, "timestamp": timestamp},
            f,
            indent=2,
        )

    print(f"Results written to {output_file}")
