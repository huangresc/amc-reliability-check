import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def normalize_global(X_train, X_test):
    """全局标准化：用训练集的均值和标准差"""
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    return X_train, X_test


def normalize_per_sample(X_train, X_test):
    """逐样本标准化：每个样本独立标准化"""
    # X shape: [N, 2, 128]
    # 对每个样本的2x128矩阵做标准化
    def _norm(X):
        mean = X.mean(axis=(1, 2), keepdims=True)
        std = X.std(axis=(1, 2), keepdims=True)
        return (X - mean) / (std + 1e-8)
    return _norm(X_train), _norm(X_test)


def load_data(data_path='rml2016_data/RML2016.10a_dict.pkl', seed=42,
              normalization='none'):
    """
    normalization: 'none' | 'global' | 'per_sample'
    使用官方划分：每个 (mod, snr) 组合取一半训练，一半测试。
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    mod_types = sorted(set([key[0] for key in data.keys()]))
    mod_to_idx = {mod: i for i, mod in enumerate(mod_types)}
    idx_to_mod = {i: mod for mod, i in mod_to_idx.items()}

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    # Provenance: the (mod_id, snr, local_index) triple of every selected sample.
    # local_index is the position inside that (mod, snr) block of the pickle, so
    # the triples identify WHICH samples went where, not merely how many.  The
    # arrays below are concatenated in iteration order; the triples are sorted
    # before hashing, so the fingerprint does not depend on that order.
    train_src_list, test_src_list = [], []
    rng = np.random.RandomState(seed)

    for (mod, snr), samples in data.items():
        n = samples.shape[0]          # 每个组合 1000 个样本
        indices = rng.permutation(n)
        half = n // 2
        train_idx = indices[:half]
        test_idx = indices[half:]

        for local, dest in ((train_idx, train_src_list), (test_idx, test_src_list)):
            keys = np.empty((local.size, 3), dtype=np.int32)
            keys[:, 0] = mod_to_idx[mod]
            keys[:, 1] = int(snr)
            keys[:, 2] = local
            dest.append(keys)

        X_train_list.append(samples[train_idx])
        y_train_list.append(np.full(half, mod_to_idx[mod]))

        X_test_list.append(samples[test_idx])
        y_test_list.append(np.full(n - half, mod_to_idx[mod]))

    X_train = np.concatenate(X_train_list, axis=0).astype('float32')
    y_train = np.concatenate(y_train_list, axis=0).astype('int64')
    X_test = np.concatenate(X_test_list, axis=0).astype('float32')
    y_test = np.concatenate(y_test_list, axis=0).astype('int64')

    # 归一化：只用训练集统计量
    if normalization == 'global':
        X_train, X_test = normalize_global(X_train, X_test)
    elif normalization == 'per_sample':
        X_train, X_test = normalize_per_sample(X_train, X_test)
    # 'none': 不做处理

    X_train_t = torch.from_numpy(X_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test)

    train_idx = np.arange(len(X_train))
    test_idx = np.arange(len(X_test))

    return {
        'X_train_t': X_train_t,
        'y_train_t': y_train_t,
        'X_test_t': X_test_t,
        'y_test_t': y_test_t,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'mod_types': mod_types,
        'idx_to_mod': idx_to_mod,
        # What the split actually did.  `train_idx`/`test_idx` above are
        # np.arange(n) -- positions in the ALREADY SPLIT arrays, which carry no
        # information about which samples were selected.  This block does.
        'partition': {
            'rule': 'official_per_modsnr_half_half',
            'seed': int(seed),
            'n_groups': len(data),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'train_source_keys': (np.concatenate(train_src_list)
                                  if train_src_list else np.zeros((0, 3), np.int32)),
            'test_source_keys': (np.concatenate(test_src_list)
                                 if test_src_list else np.zeros((0, 3), np.int32)),
        },
    }


def make_loader(X_t, y_t, batch_size=256, shuffle=True, num_workers=0,
                shuffle_seed=None):
    """shuffle_seed=None keeps the original behaviour: the sampler draws its
    per-epoch seed from the global torch RNG, so the mini-batch order is a
    function of everything that consumed that RNG earlier in the process.
    Passing an int isolates the data order from the rest of the RNG stream,
    which is what the order-vs-initialisation experiment needs."""
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )