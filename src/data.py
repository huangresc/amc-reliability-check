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
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    X_list, y_list = [], []
    mod_types = sorted(set([key[0] for key in data.keys()]))
    mod_to_idx = {mod: i for i, mod in enumerate(mod_types)}
    idx_to_mod = {i: mod for mod, i in mod_to_idx.items()}
    for (mod, snr), samples in data.items():
        X_list.append(samples)
        y_list.append(np.full(samples.shape[0], mod_to_idx[mod]))

    X = np.concatenate(X_list, axis=0).astype('float32')
    y = np.concatenate(y_list, axis=0).astype('int64')

    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # 归一化
    if normalization == 'global':
        X_train, X_test = normalize_global(X_train, X_test)
    elif normalization == 'per_sample':
        X_train, X_test = normalize_per_sample(X_train, X_test)
    # 'none': 不做处理

    train_idx = np.arange(len(X_train))
    test_idx = np.arange(len(X_train), len(X_train) + len(X_test))

    X_train_t = torch.from_numpy(X_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test)

    return {
        'X_train_t': X_train_t,
        'y_train_t': y_train_t,
        'X_test_t': X_test_t,
        'y_test_t': y_test_t,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'mod_types': mod_types,
        'idx_to_mod': idx_to_mod,
    }


def make_loader(X_t, y_t, batch_size=256, shuffle=True, num_workers=4):
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )