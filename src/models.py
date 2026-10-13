import torch
import torch.nn as nn


class M2_GAPHead(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


class M4_FlattenFC(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * 32, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# M5 与 M2 架构相同，只是训练时用相位增强
M5_GAPHead_PhaseAug = M2_GAPHead


def random_phase_rotation(x):
    batch_size = x.shape[0]
    theta = torch.rand(batch_size, device=x.device) * 2 * 3.141592653589793 - 3.141592653589793
    cos_t = torch.cos(theta).view(-1, 1, 1, 1)
    sin_t = torch.sin(theta).view(-1, 1, 1, 1)
    I = x[:, :, 0:1, :]
    Q = x[:, :, 1:2, :]
    return torch.cat([I * cos_t - Q * sin_t, I * sin_t + Q * cos_t], dim=2)


MODEL_REGISTRY = {
    'M2_GAPHead': M2_GAPHead,
    'M4_FlattenFC': M4_FlattenFC,
    'M5_GAPHead_PhaseAug': M5_GAPHead_PhaseAug,
}
def discrete_phase_rotation(x):
    """ULNN风格：4个离散旋转 (0, π/2, π, 3π/2)"""
    batch_size = x.shape[0]
    theta_choices = torch.tensor([0, 1.5707963, 3.1415927, 4.7123890],
                                  device=x.device)
    idx = torch.randint(0, 4, (batch_size,), device=x.device)
    theta = theta_choices[idx]
    cos_t = torch.cos(theta).view(-1, 1, 1, 1)
    sin_t = torch.sin(theta).view(-1, 1, 1, 1)
    I = x[:, :, 0:1, :]
    Q = x[:, :, 1:2, :]
    return torch.cat([I * cos_t - Q * sin_t, I * sin_t + Q * cos_t], dim=2)


# M5b: 与M2架构相同，训练时用离散4旋转
M5b_GAPHead_PhaseAug4 = M2_GAPHead

MODEL_REGISTRY = {
    'M2_GAPHead': M2_GAPHead,
    'M4_FlattenFC': M4_FlattenFC,
    'M5_GAPHead_PhaseAug': M5_GAPHead_PhaseAug,
    'M5b_GAPHead_PhaseAug4': M5b_GAPHead_PhaseAug4,
}# Tier 2: 归一化变体（架构与M2/M4相同，只是数据归一化方式不同）
# 这些变体不需要新的模型类，只需要在 train.py 里指定 normalization

# 注册表扩展（用于 train.py）
MODEL_REGISTRY.update({
    'M2n_g': M2_GAPHead,
    'M4n_g': M4_FlattenFC,
    'M2n_p': M2_GAPHead,
    'M4n_p': M4_FlattenFC,
})