"""Minimal network training step using entirely artificial signals."""
import torch
from torch import nn
from ecg_gpu import ECGImageLayer, RenderConfig, Choice, Uniform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)
images = ECGImageLayer(fs=500, config=RenderConfig(height=128, width=192),
    parameters={'layout': Choice(['3x4+r3', '12x1']), 'angle': Uniform(-.1, .1)})
cnn = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))
pipeline = nn.Sequential(images, cnn).to(device)
optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-3)
signals = torch.zeros(2, 12, 5000, device=device)
classes = torch.tensor([0, 1], device=device)
optimizer.zero_grad(set_to_none=True)
loss = nn.functional.cross_entropy(pipeline(signals), classes)
loss.backward()
optimizer.step()
print({'device': str(device), 'loss': float(loss.detach()), 'artificial_signals': True})
