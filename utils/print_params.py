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


class M4_LMRCNN(nn.Module):
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


print("=" * 60)
print("M2_GAPHead 逐层参数量")
print("=" * 60)
m2 = M2_GAPHead()
total = 0
for name, p in m2.named_parameters():
    print(f"{name:40s} {p.numel():>10,}")
    total += p.numel()
print(f"{'总计':40s} {total:>10,}")

print()
print("=" * 60)
print("M4_LMRCNN 逐层参数量")
print("=" * 60)
m4 = M4_LMRCNN()
total = 0
for name, p in m4.named_parameters():
    print(f"{name:40s} {p.numel():>10,}")
    total += p.numel()
print(f"{'总计':40s} {total:>10,}")