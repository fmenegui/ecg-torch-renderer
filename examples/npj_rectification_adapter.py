"""Instrument the authentic npj Open-ECG-Digitizer rectification stage.

Upstream code and checkpoints are imported unchanged. This adapter exposes
the geometry already used by its default inference pipeline, before signal
extraction and lead-layout recognition. No gold enters inference.

SPDX-License-Identifier: CC-BY-SA-4.0
Pipeline instrumentation adapted from Ahus-AIM/Open-ECG-Digitizer,
src/model/inference_wrapper.py, at the commit pinned below. Changes expose
the coordinate map and bypass downstream layout and waveform modules.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as nnf
from torchvision.transforms import functional as tvf
import yaml


UPSTREAM_COMMIT = '97a15087d4abcda843da8c58ee74b1d8f47e6f9a'
WEIGHT_SHA256 = '17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b'
DEFAULT_ROOT = Path(__file__).resolve().parent/'Open-ECG-Digitizer'
SOURCE_HASHES = {
    'src/model/unet.py': '549a06ed70eb1bc04a833c3fc0ec71f3cb46ba3e9fe6ad08262fd9dcd87b55a2',
    'src/model/perspective_detector.py': 'a5b50c3acbbacc3a57d7111b5b97dac56f0fd70d52c39506ba06b82265fb1de5',
    'src/model/cropper.py': 'e00342396d80e1395a7bad7e847b30841c46d20d3086330bf9fd10a16ac65ef2',
    'src/model/pixel_size_finder.py': '82b18209b432daf6a1f4cb4cc3cfe6b9304737eabb53713b9492c3191158d197',
    'src/model/inference_wrapper.py': 'fc7ee888f4cb1065d3d6b80b1a73b7babd2f1cb989bce770515be9208b3d4fd2',
}
PRESETS = {
    'ahus': ('inference_wrapper_ahus_testset.yml', '3b5bc19ad09078599b7d056e1d9a4872f3c0bf67dc2aaf7210956aa0fad9dd45'),
    'george-moody-2024': ('inference_wrapper_george-moody-2024.yml', '2ea9416ae5e2751fc2084523fdb95253072157acda9fde8ddd44429b6650b181'),
    'generic': ('inference_wrapper.yml', '870e8b27f86687e6f4dd5295b8984793bc2d84a26410bb37ff6f7112e64df40a'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def homogeneous(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    value = np.column_stack((points, np.ones(len(points))))@matrix.T
    with np.errstate(invalid='ignore', divide='ignore'):
        return value[:, :2]/value[:, 2:]


def tv_forward_matrix(source_points, destination_points):
    """Pixel-index map reproducing torchvision's corner-coordinate sampler.

    Torchvision computes output-corner -> input-corner coefficients and
    evaluates them at output pixel centers x+0.5,y+0.5. Raw pixel indices
    therefore require half-pixel translations on both sides. The returned
    continuous map precedes BILINEAR sampling, as used by upstream Cropper.
    """
    coefficients = tvf._get_perspective_coeffs(
        source_points.tolist(), destination_points.tolist())
    backward = np.r_[np.asarray(coefficients, dtype=np.float32), 1.].reshape(3, 3)
    plus = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    minus = np.array([[1., 0., -.5], [0., 1., -.5], [0., 0., 1.]])
    return minus@np.linalg.inv(backward)@plus


class NPJMap:
    """Estimated source-pixel -> cropped-output mm, with native eligibility."""

    def __init__(self, source_to_pixels, mm_per_pixel, source_shape, output_shape):
        self.source_to_pixels = np.asarray(source_to_pixels, dtype=np.float64)
        self.mm_per_pixel = np.asarray(mm_per_pixel, dtype=np.float64)
        self.source_to_mm = np.diag([*self.mm_per_pixel, 1.])@self.source_to_pixels
        self.source_shape = tuple(source_shape)
        self.output_shape = tuple(output_shape)

    def predict(self, source_queries):
        source_queries = np.asarray(source_queries, dtype=np.float64)
        pixel = homogeneous(self.source_to_pixels, source_queries)
        prediction = pixel*self.mm_per_pixel
        sh, sw = self.source_shape
        oh, ow = self.output_shape
        eligible = (np.isfinite(pixel).all(1) & np.isfinite(prediction).all(1)
                    & (source_queries[:, 0] >= 0) & (source_queries[:, 0] <= sw-1)
                    & (source_queries[:, 1] >= 0) & (source_queries[:, 1] <= sh-1)
                    & (pixel[:, 0] >= 0) & (pixel[:, 0] <= ow-1)
                    & (pixel[:, 1] >= 0) & (pixel[:, 1] <= oh-1))
        return prediction, eligible

    __call__ = predict


class NPJPerspectiveRectifier:
    """Unchanged official-preset geometry, with no downstream layout requirement."""

    def __init__(self, upstream_root=DEFAULT_ROOT, seed=1973, preset='ahus'):
        self.root = Path(upstream_root)
        self.seed = seed
        self.preset = preset
        config_name, expected_config_hash = PRESETS[preset]
        config_path = self.root/'src/config'/config_name
        self.config_hash = digest(config_path)
        if self.config_hash != expected_config_hash:
            raise ValueError('Authentic upstream preset hash mismatch')
        self.config = yaml.safe_load(config_path.read_text())['MODEL']['KWARGS']
        for path, expected in SOURCE_HASHES.items():
            if digest(self.root/path) != expected:
                raise ValueError('Authentic upstream source hash mismatch: '+path)
        names = ['unet', 'perspective_detector', 'cropper', 'pixel_size_finder']
        self.source_hashes = SOURCE_HASHES.copy()
        modules = {}
        for name in names:
            path = self.root/'src/model'/f'{name}.py'
            self.source_hashes[str(path.relative_to(self.root))] = digest(path)
            modules[name] = module_from_file('m3_npj_'+name, path)
        weights = self.root/'weights/unet_weights_07072025.pt'
        if digest(weights) != WEIGHT_SHA256:
            raise ValueError('Authentic upstream checkpoint hash mismatch')
        component_config = self.config['config']
        self.model = modules['unet'].UNet(**component_config['SEGMENTATION_MODEL']['KWARGS']).eval()
        state = torch.load(weights, weights_only=True, map_location='cpu')
        if isinstance(state, tuple):
            state = state[0]
        self.model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
        self.detector = modules['perspective_detector'].PerspectiveDetector(**component_config['PERSPECTIVE_DETECTOR']['KWARGS'])
        self.cropper = modules['cropper'].Cropper(**component_config['CROPPER']['KWARGS'])
        self.scaler = modules['pixel_size_finder'].PixelSizeFinder(**component_config['PIXEL_SIZE_FINDER']['KWARGS'])

    @staticmethod
    def sparse_probability(value):
        value = torch.clamp(value-value.mean(), min=0)
        return value/(value.max()+1e-9)

    @torch.no_grad()
    def infer(self, image):
        """Return map, native normalized RGB, and inference-only metadata.

        The stages reproduce the selected official inference wrapper preset with
        device='cpu'. Official wrapper downstream text/trace modules are not
        loaded. Full-page preservation is not added to this native comparator.
        """
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError('Expected RGB HxWx3 image')
        image_tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float()
        minimum, maximum = image_tensor.min(), image_tensor.max()
        if maximum == minimum:
            raise ValueError('Constant input has undefined native min-max normalization')
        image_tensor = (image_tensor-minimum)/(maximum-minimum)
        height, width = image_tensor.shape[-2:]
        new_height, new_width = height, width
        if min(height, width) < 512:
            scale = 512/min(height, width)
            new_height, new_width = int(height*scale), int(width*scale)
            image_tensor = nnf.interpolate(image_tensor, size=(new_height, new_width),
                                           mode='bilinear', align_corners=False)
        elif max(height, width) > 3000:
            scale = 3000/max(height, width)
            new_height, new_width = int(height*scale), int(width*scale)
            image_tensor = nnf.interpolate(image_tensor, size=(new_height, new_width),
                                           mode='bilinear', align_corners=False, antialias=True)
        sx, sy = new_width/width, new_height/height
        source_to_resized = np.array([[sx, 0., (sx-1)/2], [0., sy, (sy-1)/2], [0., 0., 1.]])
        segmentation_start = time.perf_counter()
        probabilities = torch.softmax(self.model(image_tensor), dim=1)
        signal = self.sparse_probability(probabilities[:, [2]])
        grid = self.sparse_probability(probabilities[:, [0]])
        segmentation_seconds = time.perf_counter()-segmentation_start
        del probabilities
        geometry_start = time.perf_counter()
        parameters = self.detector(grid)
        source_points = self.cropper(signal, parameters)
        destination_points = self.cropper._calculate_destination_points(
            new_height, new_width, source_points)
        pixel_matrix = tv_forward_matrix(source_points, destination_points)@source_to_resized
        aligned_signal = self.cropper.apply_perspective(signal, source_points, fill_value=0)
        aligned_grid = self.cropper.apply_perspective(grid, source_points, fill_value=0)
        aligned_image = self.cropper.apply_perspective(image_tensor, source_points, fill_value=0)
        rotated = aligned_image.shape[2] > aligned_image.shape[3]
        if rotated:
            h_before = aligned_image.shape[2]
            aligned_image = torch.rot90(aligned_image, k=3, dims=(2, 3))
            aligned_signal = torch.rot90(aligned_signal, k=3, dims=(2, 3))
            aligned_grid = torch.rot90(aligned_grid, k=3, dims=(2, 3))
            pixel_matrix = np.array([[0., -1., h_before-1], [1., 0., 0.], [0., 0., 1.]])@pixel_matrix
        combined = aligned_signal+aligned_grid
        profile = combined.squeeze().sum(dim=combined.dim()-3)
        profile = torch.clamp(profile-profile.mean(), min=0)
        nonzero = (profile > 0).nonzero(as_tuple=True)[0]
        if nonzero.numel():
            y1, y2 = int(nonzero[0].item()), int(nonzero[-1].item())
        else:
            y1, y2 = 0, combined.shape[2]-1
        aligned_grid = aligned_grid[:, :, y1:y2+1]
        aligned_image = aligned_image[:, :, y1:y2+1]
        pixel_matrix = np.array([[1., 0., 0.], [0., 1., -y1], [0., 0., 1.]])@pixel_matrix
        mm_x, mm_y = map(float, self.scaler(aligned_grid))
        if not np.isfinite([mm_x, mm_y]).all() or min(mm_x, mm_y) <= 0:
            raise ValueError('Native scale estimation failed')
        geometry_seconds = time.perf_counter()-geometry_start
        mapping = NPJMap(pixel_matrix, [mm_x, mm_y], [height, width], aligned_image.shape[-2:])
        metadata = {
            'comparator': 'npj authentic official-preset perspective stage, instrumented',
            'upstream_repository': 'https://github.com/Ahus-AIM/Open-ECG-Digitizer',
            'upstream_commit': UPSTREAM_COMMIT, 'upstream_source_sha256': self.source_hashes,
            'official_preset': self.preset, 'official_config_sha256': self.config_hash,
            'geometry_parameters': self.config['config']['CROPPER']['KWARGS'],
            'scale_parameters': self.config['config']['PIXEL_SIZE_FINDER']['KWARGS'],
            'checkpoint_sha256': WEIGHT_SHA256, 'device': 'cpu', 'seed': self.seed,
            'torch_version': torch.__version__, 'num_threads': torch.get_num_threads(),
            'source_shape': [height, width], 'resized_shape': [new_height, new_width],
            'source_points_in_resized_pixel_indices': source_points.tolist(),
            'destination_points_corner_convention': destination_points.tolist(),
            'source_to_output_pixels': pixel_matrix.tolist(),
            'output_shape': list(aligned_image.shape[-2:]), 'clockwise_portrait_rotation': rotated,
            'cropped_rows_in_rotated_canvas': [y1, y2], 'mm_per_pixel_xy': [mm_x, mm_y],
            'physical_origin': 'native cropped output pixel center (0,0), arbitrary grid phase',
            'native_interpolation': 'torchvision perspective BILINEAR default',
            'gold_used_for_inference': False, 'apply_dewarping': False,
            'downstream_layout_and_waveform_stages_executed': False,
            'segmentation_seconds': segmentation_seconds, 'geometry_seconds': geometry_seconds,
            'seconds': time.perf_counter()-started,
            'coordinate_convention': 'pixel-index centers with exact torchvision half-pixel corner composition',
            'source_rgb_minmax_normalized': True, 'full_page_preservation_added': False,
        }
        output = aligned_image[0].permute(1, 2, 0).cpu().numpy()
        return mapping, output, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--upstream-root', default=DEFAULT_ROOT, type=Path)
    parser.add_argument('--preset', choices=tuple(PRESETS), default='ahus')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with np.load(args.input) as archive:
        image = archive['image'].copy()
    rectifier = NPJPerspectiveRectifier(args.upstream_root, preset=args.preset)
    mapping, output, metadata = rectifier.infer(image)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'geometry.npz', source_to_output_pixels=mapping.source_to_pixels,
                        source_to_mm=mapping.source_to_mm, mm_per_pixel=mapping.mm_per_pixel,
                        source_shape=mapping.source_shape, output_shape=mapping.output_shape,
                        normalized_rectified_rgb=output)
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
