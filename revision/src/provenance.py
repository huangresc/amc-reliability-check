import hashlib
import json
import os
import platform

import numpy as np
import torch


def sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def sha1_json(obj):
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha1_source_keys(keys):
    """Fingerprint of WHICH samples a split selected.

    `keys` is an (n, 3) array of (mod_id, snr, local_index) triples, where
    local_index is the position inside that (mod, snr) block of the pickle.  The
    rows are sorted before hashing so the result depends on the selection, not
    on the order the blocks happened to be concatenated in.

    This is the field that identifies a partition.  It supersedes
    `data.train_idx_sha1` in build_nuisance below, which is sha1 of
    list(range(n_train)) because both loaders hand it positions inside the
    already-split arrays -- it records the SIZE of the training set and nothing
    about its contents, so two different splits of the same size collide.
    """
    keys = np.asarray(keys)
    if keys.size == 0:
        return sha1_json([])
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(keys[order], dtype="<i4").tobytes())
    return h.hexdigest()


def build_partition(partition):
    """Manifest block describing the data split.  See sha1_source_keys."""
    return {
        "rule": partition["rule"],
        "seed": partition["seed"],
        "n_groups": partition["n_groups"],
        "n_train": partition["n_train"],
        "n_test": partition["n_test"],
        "train_source_sha1": sha1_source_keys(partition["train_source_keys"]),
        "test_source_sha1": sha1_source_keys(partition["test_source_keys"]),
    }


def build_nuisance(cfg, train_idx, test_idx, model, x_sample):
    """两个被判为可比的运行，此块必须逐字相同。禁止出现架构与增强。"""
    d = {
        "framework": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_available": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_used": str(next(model.parameters()).device),
        "device_used_inputs": str(x_sample.device),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "python": platform.python_version(),
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "betas": cfg["betas"],
        "eps": cfg["eps"],
        "weight_decay": cfg["weight_decay"],
        "scheduler": cfg["scheduler"],
        "loss": cfg["loss"],
        "batch_size": cfg["batch_size"],
        "total_epochs": cfg["total_epochs"],
        "seed": cfg["seed"],
        "dropout": cfg["dropout"],
        "normalization": cfg["normalization"],
        "use_bn": cfg["use_bn"],
        "num_workers": cfg.get("num_workers", 0),
        "pin_memory": cfg.get("pin_memory", False),
        "data": {
            "dataset": cfg["dataset"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            # LEGACY, size-only: both loaders set these to np.arange(n), so these
            # are sha1(range(n_train)) and collide across different splits of
            # the same size.  Kept byte-identical so that the 185 manifests
            # already on disk, and the analysis scripts that filter on this
            # value, stay valid.  The real partition fingerprint is
            # manifest["partition"] (build_partition above), which is
            # deliberately OUTSIDE the nuisance block so that nuisance_hash --
            # the comparability key -- does not move.
            "train_idx_sha1": sha1_json(sorted(train_idx.tolist())),
            "test_idx_sha1": sha1_json(sorted(test_idx.tolist())),
        },
        "eval": {
            "eval_batch_size": cfg["eval_batch_size"],
            "model_eval_called": True,
            "tta": False,
            "metric": "acc_v1",
        },
    }
    # Recorded only when an explicit data-order seed is in force. Runs that
    # leave it None inherit the ambient RNG stream (the historical behaviour),
    # so omitting the key keeps every pre-existing nuisance hash unchanged
    # while giving order-controlled runs a distinguishable identity.
    if cfg.get("shuffle_seed") is not None:
        d["shuffle_seed"] = cfg["shuffle_seed"]
    return d


def build_treatment(cfg):
    return {
        "head": cfg["head_spec"],
        "use_aug": cfg["use_aug"],
        "aug_spec": cfg["aug_spec"],
        "normalization": cfg["normalization"],
    }


def save_ckpt(model, optimizer, run_dir, epoch):
    """原子写 + 只追加的 sha1 台账（同一epoch只记录一次）"""
    path = os.path.join(run_dir, f"ckpt_ep{epoch:03d}.pt")
    tmp = path + ".tmp"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, tmp)
    os.replace(tmp, path)
    d = sha1_file(path)

    sha1_txt = os.path.join(run_dir, "ckpt_sha1.txt")
    already_exists = False
    if os.path.exists(sha1_txt):
        with open(sha1_txt, 'r') as f:
            for line in f:
                if line.startswith(f"{epoch}\t"):
                    already_exists = True
                    break

    if not already_exists:
        with open(sha1_txt, "a") as f:
            f.write(f"{epoch}\t{d}\t{os.path.basename(path)}\n")
    else:
        with open(sha1_txt, 'r') as f:
            lines = f.readlines()
        with open(sha1_txt, 'w') as f:
            for line in lines:
                if line.startswith(f"{epoch}\t"):
                    f.write(f"{epoch}\t{d}\t{os.path.basename(path)}\n")
                else:
                    f.write(line)

    return d


def assert_gpu(model, x):
    """硬断言：不在GPU上直接退出"""
    assert next(model.parameters()).device.type == "cuda", "MODEL NOT ON GPU - ABORT"
    assert x.device.type == "cuda", "BATCH NOT ON GPU - ABORT"