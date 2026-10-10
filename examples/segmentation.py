"""Train a small U-Net from signals with aligned synthetic 12-channel targets.

This verifies software integration on artificial signals. It is not a trained
digitizer or a clinical quality benchmark. Run: python examples/segmentation.py
"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, RenderConfig, Choice, Uniform
from ecg_gpu.demo import demo_signals


class TinyUNet(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, channels, 1))

    def forward(self, image):
        skip = self.encoder(image)
        low = self.bottleneck(F.max_pool2d(skip, 2))
        high = F.interpolate(low, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.decoder(torch.cat((skip, high), dim=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--compile-raster', action='store_true')
    options = parser.parse_args()
    if options.steps < 1:
        parser.error('Positive steps required')
    torch.set_num_threads(2)
    device = torch.device(options.device)
    synth = ECGSupervisionLayer(fs=500,
        config=RenderConfig(height=128, width=192, surface_resolution=32, shadow_samples=0),
        parameters={'layout': Choice(['3x4+r3', '6x2', '12x1']), 'angle': Uniform(-.12, .12)},
        supervision=SupervisionConfig(outputs=('lead_masks',)),
        compile_raster=options.compile_raster).to(device)
    model = TinyUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values, _, _ = demo_signals(fs=500, duration=10)
    signals = torch.tensor(values, device=device)[None].expand(2, -1, -1)
    before = model.encoder[0].weight.detach().clone()
    for step in range(options.steps):
        batch = synth(signals)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['image'])
        loss_map = F.binary_cross_entropy_with_logits(logits, batch['lead_masks'], reduction='none')
        loss = (loss_map * batch['valid']).sum() / (batch['valid'].sum() * logits.shape[1]).clamp_min(1)
        loss.backward()
        optimizer.step()
        print({'step': step, 'loss': float(loss.detach()), 'device': str(device)})
    if not torch.isfinite(loss) or torch.equal(before, model.encoder[0].weight):
        raise AssertionError('Nonfinite loss or no network weight update')
    print({'status': 'passed', 'output_shape': list(logits.shape), 'artificial_signals': True})


if __name__ == '__main__':
    main()
