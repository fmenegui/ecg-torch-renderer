"""Train a local Ultralytics detector from scratch without downloading weights.

Install optional dependency: pip install ultralytics
Export first: python examples/export_yolo.py --out build/yolo
Smoke: python examples/train_yolo.py --data build/yolo/dataset.yaml --epochs 1
This is an integration example, not a validated ECG detector.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--architecture', default='yolo11n.yaml',
                        help='Bundled Ultralytics YAML or local architecture YAML, never .pt weights')
    parser.add_argument('--loss-smoke', action='store_true',
                        help='Run one actual DetectionModel optimizer step without the torchvision validator')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--project', type=Path, default=Path('build/yolo-training'))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error('Positive epochs/batch and image size >=32 required')
    if Path(args.architecture).suffix not in ('.yaml', '.yml'):
        parser.error('Use a model architecture YAML to avoid pretrained weight downloads')
    if not args.data.is_file():
        parser.error('Export a local dataset.yaml before training')
    try:
        from ultralytics import YOLO
    except ImportError:
        parser.error('Optional dependency unavailable. Install with: pip install ultralytics')
    if args.loss_smoke:
        loss_smoke(args)
        return
    model = YOLO(args.architecture, task='detect')
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=0, pretrained=False, amp=False,
        plots=False, project=str(args.project.resolve()), name='ecg', exist_ok=False,
        # Vertical/horizontal flips change ECG time or polarity. Mosaic changes
        # page geometry. Disable them for this controlled integration example.
        fliplr=0., flipud=0., mosaic=0., mixup=0., copy_paste=0.)


def loss_smoke(args):
    """Read exported PNG/txt pairs and exercise YOLO's genuine training loss."""
    import torch
    import numpy as np
    from PIL import Image
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import YAML
    torch.set_num_threads(2)
    data = YAML.load(args.data)
    root = Path(data.get('path', args.data.parent))
    train = root / data['train']
    files = sorted(train.glob('*.png'))[:args.batch]
    if not files:
        raise ValueError('loss-smoke requires exported training PNG files')
    images, classes, boxes, batch_indices = [], [], [], []
    for index, file in enumerate(files):
        with Image.open(file) as image:
            array = np.array(image.convert('RGB').resize((args.imgsz, args.imgsz)))
        images.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255.)
        relative = file.relative_to(root / 'images')
        for line in (root / 'labels' / relative.with_suffix('.txt')).read_text().splitlines():
            values = line.split()
            classes.append([int(values[0])])
            boxes.append([float(v) for v in values[1:]])
            batch_indices.append(index)
    device = torch.device(args.device)
    batch = {'img': torch.stack(images).to(device),
             'batch_idx': torch.tensor(batch_indices, device=device),
             'cls': torch.tensor(classes, device=device).reshape(-1, 1),
             'bboxes': torch.tensor(boxes, device=device).reshape(-1, 4)}
    model = DetectionModel(args.architecture, nc=len(data['names']), verbose=False).to(device).train()
    model.args = get_cfg()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = next(model.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss, components = model(batch)
    loss.sum().backward()
    optimizer.step()
    if not torch.isfinite(loss).all() or torch.equal(before, next(model.parameters())):
        raise AssertionError('YOLO loss nonfinite or model weights unchanged')
    components = ({key: value.detach().cpu().tolist() if torch.is_tensor(value) else value
                   for key, value in components.items()} if isinstance(components, dict)
                  else components.detach().cpu().tolist())
    print({'status': 'passed', 'mode': 'one-loss-optimizer-step', 'classes': len(data['names']),
           'loss_components': components, 'pretrained': False,
           'device': str(device), 'clinical_validation': False})


if __name__ == '__main__':
    main()
