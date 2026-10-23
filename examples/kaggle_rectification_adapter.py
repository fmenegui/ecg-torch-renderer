"""CPU wrapper for the pinned third-place Kaggle author's actual rectifier.

The author's source and weights stay external. This wrapper supplies its missing
reference asset, disables pretrained downloads and uses float32 on CPU. Model
definitions, thresholds, topology assignment, cubic fill and RGB sampling are
called from the original source. No gold data enters this module.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np
import torch


PINS = {
    'source_commit': 'f2b4ac2823007e3fb387a7e188e748ad832bb2b7',
    'source_file_sha256': '130768d580e130706774bbc6fd97c337cf118a3d962af6e67beb0332d9303b3d',
    'stage0_sha256': 'b7360dfce9c49305b54272cf242e93e5728823f439a8e5ad59c03b073988cfc9',
    'stage1_sha256': '313f1e5dafdc212977e4586ce118453866830be0fa32664e36054f7f4568a46a',
    'reference_sha256': 'e695f580bd0de4df2fe2e2ba8c89809af819127b2e5936c2716830f421cef4ad',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError('Unsupported report value '+type(value).__name__)


def rotate_xy(points, shape, k):
    """Pixel-center coordinates corresponding exactly to np.rot90(image,k)."""
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    height, width = shape
    return np.column_stack(((x, y), (y, width-1-x),
                            (width-1-x, height-1-y), (height-1-y, x))[k % 4])


class NativeKaggleField:
    """Invert the native backward bilinear grid, without fitting a replacement.

    Cells that are nonconvex or have conflicting inverse images cannot supply
    an unambiguous physical source-to-target coordinate. Their count is exposed.
    This validity is evaluator geometry, not a confidence supplied by the model.
    """
    def __init__(self, grid, homography, rotation, source_shape, report):
        self.report = report
        self.rotation = int(rotation)
        self.source_shape = tuple(source_shape)
        self.homography = homography
        if grid is None:
            self.grid = None
            return
        height, width = source_shape
        rotated_width = height if rotation % 2 else width
        scale = rotated_width/1440.
        normalized_shape = (int(1152*scale), int(1440*scale))
        self.normalized_shape = normalized_shape
        self.scale = scale
        self.high_homography = np.diag([scale, scale, 1.])@homography
        # This explicitly RETAINS the author's W-1 / align_corners=False rule.
        nh, nw = normalized_shape
        self.grid = np.asarray(grid, np.float64)*scale*np.array([nw/(nw-1), nh/(nh-1)])-.5
        a, p10 = self.grid[:-1, :-1], self.grid[:-1, 1:]
        p01, p11 = self.grid[1:, :-1], self.grid[1:, 1:]
        vertices = np.stack([a, p10, p11, p01], axis=2).reshape(-1, 4, 2)
        edges = np.roll(vertices, -1, axis=1)-vertices
        following = np.roll(edges, -1, axis=1)
        cross = edges[..., 0]*following[..., 1]-edges[..., 1]*following[..., 0]
        finite = np.isfinite(vertices).all((1, 2))
        convex = ((cross > 1e-8).all(1) | (cross < -1e-8).all(1)) & finite
        self.cell_ids = np.argwhere(np.ones((43, 56), dtype=bool))[convex]
        self.a = a.reshape(-1, 2)[convex]
        self.b = (p10-a).reshape(-1, 2)[convex]
        self.c = (p01-a).reshape(-1, 2)[convex]
        self.d = (p11-p10-p01+a).reshape(-1, 2)[convex]
        self.lo = vertices.min(1)[convex]
        self.hi = vertices.max(1)[convex]
        self.report['inverse_map'] = {
            'method': 'Newton inversion of native bilinear cells, before inverse global homography',
            'nominal_grid_pitch_mm': 5.,
            'gold_used': False, 'new_geometry_fit': False,
            'total_cells': 43*56, 'convex_cells': int(convex.sum()),
            'nonconvex_or_nonfinite_cells': int((~convex).sum()),
            'native_model_confidence': False,
            'ambiguity_policy': 'reject conflicting inverse positions, accept shared edges only if coordinates agree',
            'pixel_convention': 'author normalization retained, effective sample=x*W/(W-1)-0.5',
            'image_resampling_quantization_modeled': False,
        }

    def forward(self, query):
        query = np.asarray(query, dtype=np.float64)
        result = np.full((len(query), 2), np.nan)
        eligible = np.zeros(len(query), dtype=bool)
        if self.grid is None or not len(self.a):
            return result, eligible
        rotated = rotate_xy(query, self.source_shape, self.rotation)
        homogeneous = np.column_stack([rotated, np.ones(len(rotated))])@self.high_homography.T
        target = homogeneous[:, :2]/homogeneous[:, 2:]
        nh, nw = self.normalized_shape
        bounds = (np.isfinite(target).all(1) & (target[:, 0] >= -1e-7)
                  & (target[:, 0] <= nw-1+1e-7) & (target[:, 1] >= -1e-7)
                  & (target[:, 1] <= nh-1+1e-7))
        for start in range(0, len(query), 128):
            stop = min(start+128, len(query))
            q = target[start:stop]
            match = ((q[:, None] >= self.lo[None]-1e-7).all(2)
                     & (q[:, None] <= self.hi[None]+1e-7).all(2)
                     & bounds[start:stop, None])
            qi, ci = np.nonzero(match)
            if not len(qi):
                continue
            a, b, c, d = self.a[ci], self.b[ci], self.c[ci], self.d[ci]
            uv = np.full((len(qi), 2), .5)
            desired = q[qi]
            for _ in range(16):
                u, v = uv[:, :1], uv[:, 1:]
                residual = a+b*u+c*v+d*u*v-desired
                ju, jv = b+d*v, c+d*u
                det = ju[:, 0]*jv[:, 1]-ju[:, 1]*jv[:, 0]
                safe = np.where(np.abs(det) > 1e-12, det, np.nan)
                du = (residual[:, 0]*jv[:, 1]-residual[:, 1]*jv[:, 0])/safe
                dv = (ju[:, 0]*residual[:, 1]-ju[:, 1]*residual[:, 0])/safe
                uv -= np.column_stack([du, dv])
            u, v = uv[:, :1], uv[:, 1:]
            error = np.linalg.norm(a+b*u+c*v+d*u*v-desired, axis=1)
            inside = ((uv >= -1e-6).all(1) & (uv <= 1+1e-6).all(1)
                      & np.isfinite(error) & (error < 1e-5))
            # Multiple adjacent cells on a shared edge are the same coordinate.
            physical = (self.cell_ids[ci][:, ::-1]+uv)*5.
            for local in np.unique(qi[inside]):
                values = physical[inside & (qi == local)]
                if np.max(np.linalg.norm(values-values[0], axis=1)) <= 1e-4:
                    result[start+local] = values.mean(0)
                    eligible[start+local] = True
        return result, eligible


class KaggleThirdPlaceAdapter:
    def __init__(self, source_repo, assets, *, threads=2):
        self.source_repo, self.assets = Path(source_repo), Path(assets)
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        reference = self.assets/'640106434-0001.gridpoint_xy.npy'
        for name in ('stage0', 'stage1'):
            p = self.assets/'weight'/f'{name}-last.checkpoint.pth'
            if sha256(p) != PINS[name+'_sha256']:
                raise ValueError('Checkpoint hash mismatch: '+str(p))
        if sha256(reference) != PINS['reference_sha256']:
            raise ValueError('Reference grid hash mismatch')
        supplied = self.source_repo/'frog'/reference.name
        supplied.parent.mkdir(exist_ok=True)
        if not supplied.exists():
            shutil.copyfile(reference, supplied)
        if sha256(supplied) != PINS['reference_sha256']:
            raise ValueError('Author reference asset path contains different data')
        module_path = self.source_repo/'data_preparation/prepare_inputs.py'
        if sha256(module_path) != PINS['source_file_sha256']:
            raise ValueError('Author implementation source hash mismatch')
        spec = importlib.util.spec_from_file_location('kaggle_third_place_original', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.stage0 = module.Stage0Net(pretrained=False).cpu().eval()
        self.stage1 = module.Stage1Net(pretrained=False).cpu().eval()
        self.load_reports = {}
        for name, model in [('stage0', self.stage0), ('stage1', self.stage1)]:
            state = torch.load(self.assets/'weight'/f'{name}-last.checkpoint.pth',
                               map_location='cpu', weights_only=True)['state_dict']
            loaded = model.load_state_dict(state, strict=True)
            model.output_type = ['infer']
            self.load_reports[name] = {'missing_keys': loaded.missing_keys,
                                       'unexpected_keys': loaded.unexpected_keys,
                                       'parameters': sum(p.numel() for p in model.parameters())}
            del state
        self.provenance = {
            'implementation': 'author third-place prepare_inputs.py, CPU execution wrapper',
            **PINS, 'source_file_sha256': sha256(module_path),
            'runtime': {'torch': torch.__version__, 'opencv': cv2.__version__,
                        'threads': threads, 'device': 'cpu', 'dtype': 'float32'},
            'adaptations': ['Supply missing author frog/reference grid asset from official model dataset',
                            'Disable ImageNet preload, strictly load all author checkpoint keys',
                            'CPU float32 replaces CUDA float16 autocast',
                            'Call original stages directly without its multiprocessing/file-list driver'],
            'algorithm_changes': False, 'gold_used': False, 'models': self.load_reports,
            'model_dataset_license': 'Unknown in Kaggle metadata, do not redistribute weights',
        }

    def infer(self, image, output_dir=None):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected RGB uint8 HWC image')
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        m = self.module
        report = {'provenance': self.provenance, 'status': 'running',
                  'input_shape': list(image.shape), 'native_rectified_output': None}
        arrays, start = {}, time.perf_counter()
        try:
            np.random.seed(1973)
            with torch.inference_mode():
                batch0 = m.image_to_batch(image)
                raw0 = self.stage0(batch0)
                rotated, keypoints, k = m.output_to_predict_stage0(image, batch0, raw0)
                report['rotation_k'] = int(k)
                report['lead_keypoints_before_homography'] = keypoints
                report['stage0_model_and_decode_seconds'] = time.perf_counter()-start
                normalized, keypoints, homo = m.normalise_by_homography(rotated, keypoints)
            report['stage0_seconds'] = time.perf_counter()-start
            report['rotation_k'] = int(k)
            report['lead_keypoints'] = keypoints
            if homo is None or not np.isfinite(homo).all():
                raise ValueError('Author stage0 did not produce a finite homography')
            arrays.update(homography=homo, rotation=np.array([k]),
                          reference_grid=m.gridpoint0001_xy)
            del batch0, raw0
            gc.collect()
            t1 = time.perf_counter()
            captured = {}
            original_interpolate = m.interpolate_mapping
            def capture_grid(grid):
                captured['before_interpolation'] = grid.copy()
                return original_interpolate(grid)
            m.interpolate_mapping = capture_grid
            try:
                with torch.inference_mode():
                    batch1 = {'image': torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)}
                    raw1 = self.stage1(batch1)
                    grid = m.output_to_predict_stage1(normalized, batch1, raw1)
            finally:
                m.interpolate_mapping = original_interpolate
            report['stage1_seconds'] = time.perf_counter()-t1
            report['grid_before_interpolation_observed_nodes'] = int(np.any(captured['before_interpolation'] != 0, axis=2).sum())
            report['grid_after_interpolation_zero_nodes'] = int(np.all(grid == 0, axis=2).sum())
            arrays.update(gridpoint_xy=grid, grid_before_interpolation=captured['before_interpolation'])
            del raw1, batch1
            gc.collect()
            # Execute the author's actual high-resolution image implementation.
            with tempfile.TemporaryDirectory(prefix='ecg-kaggle-') as temporary:
                td = Path(temporary)
                cv2.imwrite(str(td/'input.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                np.save(td/'homo.npy', homo)
                np.save(td/'rotation.npy', np.array([k]))
                np.save(td/'grid.npy', grid)
                natural, scale = m.compute_fixed_size_rectified_img(td/'input.png', td/'homo.npy', td/'grid.npy', td/'rotation.npy')
                if natural is None:
                    raise ValueError('Author high-resolution rectification failed')
                rectified = cv2.resize(natural, (4400, 1700), interpolation=cv2.INTER_CUBIC)
                report['native_rectified_output'] = [4400, 1700]
                report['native_grid_span_mm'] = [280., 215.]
                report['native_nominal_pixel_density_xy_per_mm'] = [4400/280, 1700/215]
                report['native_density_is_isotropic'] = False
                report['natural_scale'] = scale
                if output_dir is not None:
                    cv2.imwrite(str(output_dir/'rectified-native.png'), rectified)
            report['status'] = 'completed'
            field = NativeKaggleField(grid, homo, k, image.shape[:2], report)
            arrays['effective_grid_normalized_highres'] = field.grid
        except Exception as exc:
            report['status'] = 'failed'
            report['error'] = type(exc).__name__+': '+str(exc)
            field = NativeKaggleField(None, None, 0, image.shape[:2], report)
        report['inference_and_RGB_seconds'] = time.perf_counter()-start
        serial = json.loads(json.dumps(report, default=json_value, allow_nan=False))
        report.clear()
        report.update(serial)
        if output_dir is not None:
            np.savez_compressed(output_dir/'native_geometry.npz', **arrays)
            (output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        return field


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    adapter = KaggleThirdPlaceAdapter(args.source_repo, args.assets)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.input is None:
        (args.out/'load-report.json').write_text(json.dumps(adapter.provenance, indent=2)+'\n')
        print(json.dumps(adapter.load_reports))
    else:
        image = cv2.cvtColor(cv2.imread(str(args.input)), cv2.COLOR_BGR2RGB)
        field = adapter.infer(image, args.out)
        print(json.dumps({'status': field.report['status'], 'seconds': field.report['inference_and_RGB_seconds']}))
