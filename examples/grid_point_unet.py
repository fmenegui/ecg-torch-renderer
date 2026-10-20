"""Train a small full-resolution U-Net with center and subpixel offset heads.

Integration demonstration on artificial ECGs, not a trained point detector.
python examples/grid_point_unet.py --steps 3 --sigma 1.2
python examples/grid_point_unet.py --steps 3 --mask-shape disk --radius 2
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import (ECGSupervisionLayer, SupervisionConfig, GridPointConfig,
                     RenderConfig, grid_point_loss, decode_grid_points)
from ecg_gpu.demo import demo_signals
from segmentation import TinyUNet


def train(*, steps=3, sigma=1.2, mask_shape='gaussian', radius=None, device='cpu'):
    if type(steps) is not int or steps < 1:
        raise ValueError('Positive integer steps required')
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = GridPointConfig(sigma_pixels=sigma, mask_shape=mask_shape,
                             radius_pixels=radius, return_offsets=True)
    synth = ECGSupervisionLayer(fs=500, seed=17,
        config=RenderConfig(height=192, width=256, surface_resolution=32, shadow_samples=0),
        parameters={'grid_present':1., 'grid_strength':.4},
        supervision=SupervisionConfig(outputs=('grid_points',), grid_point_config=config)).to(device)
    network = TinyUNet(channels=3).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    values, _, _ = demo_signals(fs=500, duration=10)
    signal = torch.tensor(values, device=device)[None].expand(2, -1, -1)
    before = network.encoder[0].weight.detach().clone()
    for _ in range(steps):
        batch = synth(signal)
        prediction = network(batch['image'])
        losses = grid_point_loss(prediction[:, :1], prediction[:, 1:], batch, pos_weight=10.)
        optimizer.zero_grad(set_to_none=True)
        losses['loss'].backward()
        optimizer.step()
    if not torch.isfinite(losses['loss']) or torch.equal(before, network.encoder[0].weight):
        raise AssertionError('Finite loss and changed network weights required')
    heads = network.decoder[-1].weight.grad
    if heads is None or not torch.isfinite(heads).all() or not (heads.flatten(1).abs().sum(-1) > 0).all():
        raise AssertionError('Both heatmap and x/y offset heads must receive gradients')
    # This demonstrates the inference contract, not detector accuracy.
    decoded = decode_grid_points(prediction[:, :1].detach().sigmoid(), prediction[:, 1:].detach())
    return {'status':'passed', 'artificial_signals':True, 'steps':steps,
            'shape':list(prediction.shape), 'loss':float(losses['loss'].detach()),
            'offset_anchors':int(batch['grid_points_offset_valid'].sum()),
            'decoded_shape':list(decoded['xy'].shape), 'accuracy_benchmark':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=1.2)
    parser.add_argument('--mask-shape', choices=('gaussian', 'disk'), default='gaussian')
    parser.add_argument('--radius', type=float)
    args = parser.parse_args()
    print(train(steps=args.steps, sigma=args.sigma, mask_shape=args.mask_shape, radius=args.radius, device=args.device))
