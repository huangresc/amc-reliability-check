import hashlib
import json
import os
import platform
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


def build_nuisance(cfg, train_idx, test_idx, model, x_sample):
    """两个被判为可比的运行，此块必须逐字相同。禁止出现架构与增强。"""
    return {
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