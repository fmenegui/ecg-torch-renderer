"""Adapters for explicitly downloaded Heng/Tang research artifacts.

No notebook execution, sys.path changes, automatic pretrained downloads,
permissive checkpoint loading or zero-valued failure substitutions.
External definitions are selected by name from hash-verified source files.
"""

import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .geometry import compose_grid_to_source, sample_source
from .signals import DigitizationResult, decode_waveform_rows, split_rows


def _definitions(path, names, namespace):
    """Load only selected definitions from a reviewed external Python source.

    This avoids notebook I/O and unrelated imports. It is NOT a sandbox.
    Callers must verify the artifact lock before loading any external code.
    """
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    missing = set(names)-definitions.keys()
    if missing:
        raise ValueError(f'Missing upstream definitions: {sorted(missing)}')
    for name in names:
        selected = ast.Module(body=[definitions[name]], type_ignores=[])
        exec(compile(selected, str(path), 'exec'), namespace)
    return namespace


def _load_weights(net, path):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if 'state_dict' in state:
        state = state['state_dict']
    if 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.output_type = ['infer']
    return net.eval()


def _environment():
    try:
        import cv2
        import cc3d
        import timm
        from scipy.interpolate import griddata
    except ImportError as exc:
        raise RuntimeError('Install ecg-torch-renderer[digitization] for reference inference') from exc
    return {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional,
            'np': np, 'cv2': cv2, 'cc3d': cc3d, 'timm': timm, 'griddata': griddata}


class HengTangDigitizer:
    """Heng stage0/1 and Tang's ConvNeXtV2 waveform checkpoint.

    The single profile uses run15/fold0 with optional four-flip TTA. The full
    profile follows the fifteen checkpoints and TTA of Tang notebook v15.
    Geometry can preserve the notebook's multi-resampling path or use the
    separately identified single-sampling experiment.
    """

    def __init__(self, assets, *, device='cpu', precision='float32', geometry='reference',
                 waveform_tta=False, profile='single', rectification_only=False):
        from .artifacts import verify_assets

        self.assets = Path(assets)
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('The reference adapter supports CPU and CUDA')
        if precision not in ('float32', 'bfloat16', 'float16'):
            raise ValueError('Unknown precision')
        if self.device.type == 'cpu' and precision == 'float16':
            raise ValueError('Use float32 or bfloat16 on CPU')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        if geometry not in ('reference', 'composed'):
            raise ValueError('geometry must be reference or composed')
        if profile not in ('single', 'full') or type(waveform_tta) is not bool:
            raise ValueError('A known profile and boolean waveform_tta are required')
        self.artifacts = verify_assets(self.assets, profile=profile,
                                       rectification_only=rectification_only)
        self.precision, self.geometry, self.waveform_tta = precision, geometry, waveform_tta
        self.profile = profile
        env = _environment()
        self.cv2 = env['cv2']
        for stage in (0, 1):
            ns = _definitions(self.assets/f'heng/stage{stage}_model.py',
                              ('MyDecoderBlock', 'MyUnetDecoder', 'encode_with_resnet', 'Net'), dict(env))
            net = ns['Net'](pretrained=False)
            _load_weights(net, self.assets/f'heng/weight/stage{stage}-last.checkpoint.pth')
            setattr(self, f'net{stage}', net.to(self.device))
        labels = ('None', 'I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5',
                  'III', 'aVF', 'V3', 'V6', 'II-rhythm')
        common = dict(env, WIDTH=1440, HEIGHT=1152, LABEL_TO_LEAD_NAME=dict(enumerate(labels)))
        common['gridpoint0001_xy'] = np.load(self.assets/'heng/640106434-0001.gridpoint_xy.npy', allow_pickle=False)
        self.common0 = _definitions(self.assets/'heng/stage0_common.py', ('make_ref_point',), dict(common))
        self.common0['REF_PT9'] = self.common0['make_ref_point']()
        _definitions(self.assets/'heng/stage0_common.py',
                     ('normalise_image', 'image_to_batch', 'marker_to_keypoint', 'output_to_predict'), self.common0)
        self.common1 = _definitions(self.assets/'heng/stage1_common.py',
            ('ROUND', 'interpolate_mapping', 'segment_to_endpoints_fitline', 'line_params',
             'compare_segment', 'canonical_order', 'canonical_y_order', 'canonical_x_order',
             'output_to_predict'), dict(common))
        if rectification_only:
            self.net2 = None
            self.hrnet = None
            return
        decoder = _definitions(self.assets/'third/models/decoders.py',
            ('CoordDecoderBlock', 'CoordUnetDecoder'), dict(env))
        ns = dict(env, CoordUnetDecoder=decoder['CoordUnetDecoder'])
        _definitions(self.assets/'third/models/stage2_net.py', ('Stage2ConvNeXtV2', 'Stage2HRNet'), ns)
        cfg = SimpleNamespace(backbone='convnextv2_tiny.fcmae_ft_in1k', pretrained=False,
            encoder_dim=[96, 192, 384, 768], decoder_dim=[256, 128, 64, 32],
            use_coord_conv=True, num_output_channels=4)
        self.net2 = _load_weights(ns['Stage2ConvNeXtV2'](cfg),
            self.assets/'third/run15/fold_0/epoch_0024.pth').to(self.device)
        self.hrnet_type = ns['Stage2HRNet']
        self.hrnet = None

    def _amp(self):
        if self.precision == 'float32':
            return nullcontext()
        return torch.autocast(self.device.type, dtype=getattr(torch, self.precision))

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _rectify(self, rotated, homography, grid):
        cv2 = self.cv2
        if self.geometry == 'composed':
            xy = compose_grid_to_source(torch.as_tensor(grid, dtype=torch.float32),
                                       homography, output_size=(1700, 4400))
            rect, valid = sample_source(rotated, xy)
            # Heng fills unsupported lattice nodes with [0, 0]. They do not
            # establish observed support even when that source pixel exists.
            support = torch.as_tensor(np.any(grid != 0, axis=-1), dtype=torch.float32)[None, None]
            support = torch.nn.functional.interpolate(support, size=(1700, 4400),
                mode='bilinear', align_corners=True)[0, 0] > 1-1e-6
            valid &= support
            return rect.clamp(0, 255).byte().numpy(), float(valid.float().mean())
        # Preserve the three sampling operations and historical align_corners
        # mismatch in Tang notebook v15. Corrected geometry is a distinct mode.
        scale = rotated.shape[1]/1440
        h = np.diag([scale, scale, 1.]) @ homography
        normalized = cv2.warpPerspective(rotated, h,
            (int(1440*scale), int(1152*scale)), flags=cv2.INTER_CUBIC)
        height, width = normalized.shape[:2]
        sparse = grid*scale/np.array([width-1, height-1])*2-1
        sparse = torch.from_numpy(np.ascontiguousarray(sparse.transpose(2, 0, 1))).float()[None]
        dense = torch.nn.functional.interpolate(sparse,
            size=(int(1700*scale), int(2200*scale)), mode='bilinear', align_corners=True)
        image = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()[None]
        rect = torch.nn.functional.grid_sample(image, dense.permute(0, 2, 3, 1),
            mode='bilinear', padding_mode='border', align_corners=False)
        rect = rect[0].permute(1, 2, 0).byte().numpy()
        return cv2.resize(rect, (4400, 1700), interpolation=cv2.INTER_CUBIC), None

    @torch.inference_mode()
    def rectify_image(self, image):
        """Run the published stage0/1 geometry without waveform extraction."""
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 32:
            raise ValueError('Reference input must be uint8 HWC RGB')
        started = time.perf_counter()
        batch = self.common0['image_to_batch'](rgb)
        # Process TTA sequentially to bound peak memory while retaining order.
        pieces = []
        with self._amp():
            for image0 in batch['image'].split(1):
                pieces.append(self.net0({'image': image0.to(self.device)}))
        output0 = {key: torch.cat([p[key] for p in pieces]).float() for key in pieces[0]}
        rotated, keypoints = self.common0['output_to_predict'](rgb, batch, output0)
        orientation = output0['orientation'].float().cpu().numpy().tolist()
        points = np.asarray([p[:2] for p in keypoints], dtype=np.float32)
        if (points != 0).all(axis=1).sum() < 4:
            raise ValueError('Fewer than four document references were detected')
        normalized, homography, inliers = self.common0['normalise_image'](rotated, points)
        if homography is None or not np.isfinite(homography).all() or inliers.sum() < 4:
            raise ValueError('Invalid document homography')
        self._sync()
        stage0_s = time.perf_counter()-started
        del output0, pieces, batch
        t1 = time.perf_counter()
        batch1 = {'image': torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))[None]}
        with self._amp():
            output1 = self.net1(batch1)
        grid, _ = self.common1['output_to_predict'](normalized, batch1, output1)
        if not np.isfinite(grid).all() or np.count_nonzero(np.any(grid != 0, axis=-1)) < grid.shape[0]*grid.shape[1]//2:
            raise ValueError('Insufficient rectification grid support')
        del output1, batch1
        rect, valid_fraction = self._rectify(rotated, homography, grid)
        self._sync()
        stage1_s = time.perf_counter()-t1
        return {'rectified': rect, 'grid_normalized_xy': grid, 'homography': homography,
                'rotated': rotated, 'geometric_valid_fraction': valid_fraction,
                'homography_inliers': int(inliers.sum()),
                'stage0_orientation_probabilities': orientation,
                'seconds': {'stage0': stage0_s, 'stage1': stage1_s}}

    @torch.inference_mode()
    def predict(self, image, *, fs, progress=None):
        """Return twelve calibrated leads plus learned geometry diagnostics."""
        if self.net2 is None:
            raise RuntimeError('This instance was created for rectification only')
        if isinstance(fs, bool) or not isinstance(fs, (int, float)) or not math.isfinite(fs) or fs <= 0 or fs != int(fs):
            raise ValueError('Reference digitization requires positive integer fs')
        geometric = self.rectify_image(image)
        rect = geometric['rectified']
        t2 = time.perf_counter()
        image2 = torch.from_numpy(np.ascontiguousarray(rect[:1696, :4352].transpose(2, 0, 1)))[None].to(self.device)
        rows, observed, checkpoint_count = self._waveforms(image2, fs, progress)
        self._sync()
        stage2_s = time.perf_counter()-t2
        warnings = ['Kaggle-specific 3x4 + long-II layout and fixed calibration',
                    'Upstream training overlap with supplied images is not established']
        if self.profile == 'single':
            warnings.append('Single run15/fold0 checkpoint, not the full third-place ensemble')
        if not observed.all():
            warnings.append('Low-confidence waveform columns were interpolated')
        result = DigitizationResult(split_rows(rows, fs),
            provenance={'profile': f'heng-tang-{self.profile}', 'precision': self.precision,
                'device': str(self.device), 'geometry': self.geometry,
                'waveform_tta': self.profile == 'full' or self.waveform_tta,
                'waveform_tta_by_group': ({'run15': True, 'run13': False, 'run10': False}
                    if self.profile == 'full' else {'run15': self.waveform_tta}),
                'waveform_checkpoints': checkpoint_count,
                'artifact_lock_sha256': self.artifacts['lock_sha256']},
            diagnostics={'seconds': dict(geometric['seconds'], stage2=stage2_s),
                'row_observed_fraction': observed.mean(axis=1).tolist(),
                'geometric_valid_fraction': geometric['geometric_valid_fraction'],
                'homography_inliers': geometric['homography_inliers'],
                'stage0_orientation_probabilities': geometric['stage0_orientation_probabilities']}, warnings=warnings)
        return result, {'rectified': rect, 'grid_normalized_xy': geometric['grid_normalized_xy'],
                        'homography': geometric['homography'], 'rows_mv': rows, 'observed_columns': observed}

    def _waveforms(self, image, fs, progress):
        if self.profile == 'single':
            groups = [(15, [24], 1., self.waveform_tta)]
        else:
            groups = [(15, [24, 23, 28, 25, 26], .5, True),
                      (13, [29, 32, 30, 30, 32], .3, False),
                      (10, [26, 29, 28, 19, 28], .2, False)]
        combined = None
        masks = []
        count = 0
        for run, epochs, weight, tta in groups:
            net = self.net2
            if run == 10:
                if self.hrnet is None:
                    cfg = SimpleNamespace(backbone='hrnet_w32.ms_in1k', pretrained=False,
                        encoder_dim=[64, 128, 256, 512], decoder_dim=[256, 128, 64, 32],
                        use_coord_conv=True, num_output_channels=4)
                    self.hrnet = self.hrnet_type(cfg).to(self.device)
                net = self.hrnet
            group = []
            for fold, epoch in enumerate(epochs):
                path = self.assets/f'third/run{run}/fold_{fold}/epoch_{epoch:04d}.pth'
                _load_weights(net, path)
                if progress is not None:
                    progress({'stage': 'waveform', 'run': run, 'fold': fold, 'tta': tta})
                probability = None
                dims = ((), (-1,), (-2,), (-1, -2)) if tta else ((),)
                for flip in dims:
                    with self._amp():
                        p = net({'image': image.flip(flip) if flip else image})['pixel'].float()
                    if flip:
                        p = p.flip(flip)
                    probability = p if probability is None else probability+p
                probability = probability[0].cpu().numpy()/len(dims)
                rows, observed = decode_waveform_rows(probability[..., 236:4160],
                    zero_mv=[703.5, 987.5, 1271.5, 1531.5], pixels_per_mv=79,
                    length=round(10*fs))
                group.append(rows)
                masks.append(observed)
                count += 1
                del probability, p
            average = np.mean(group, axis=0)*weight
            combined = average if combined is None else combined+average
        return combined, np.logical_and.reduce(masks), count
