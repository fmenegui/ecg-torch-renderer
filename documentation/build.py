"""Build the static documentation and derive reference data from the package."""
from dataclasses import asdict, fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / os.environ.get('ECG_DOCS_OUTPUT', 'docs')
sys.path.insert(0, str(ROOT))


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows])


def build():
    import torch
    import ecg_gpu
    from ecg_gpu import ECGImageLayer, ECGRenderer, ECGSupervisionLayer, RenderConfig, SupervisionConfig, profile_layer, capture_scenario
    torch.set_num_threads(2)
    renderer = ECGRenderer(RenderConfig(height=64, width=96, surface_resolution=32, shadow_samples=0))
    parameters = renderer.sample_parameters(1, generator=torch.Generator().manual_seed(0))
    metadata = {
        'version': ecg_gpu.__version__,
        'render_config': asdict(RenderConfig()),
        'supervision_config': asdict(SupervisionConfig()),
        'parameter_shapes': {key: {'shape': ['B', *tensor.shape[1:]], 'dtype': str(tensor.dtype)}
                             for key, tensor in parameters.items()},
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((ROOT / 'ecg_gpu').glob('*.py'))},
    }
    api = {
        'ECGImageLayer': ECGImageLayer, 'ECGImageLayer.forward': ECGImageLayer.forward,
        'ECGImageLayer.render_batch': ECGImageLayer.render_batch,
        'ECGSupervisionLayer': ECGSupervisionLayer, 'ECGSupervisionLayer.forward': ECGSupervisionLayer.forward,
        'ECGSupervisionLayer.supervise': ECGSupervisionLayer.supervise,
        'ECGRenderer': ECGRenderer, 'ECGRenderer.forward': ECGRenderer.forward,
        'profile_layer': profile_layer,
        'capture_scenario': capture_scenario,
    }
    metadata['signatures'] = {name: str(inspect.signature(obj)) for name, obj in api.items()}
    perf = json.loads((ROOT / 'PERFORMANCE_CPU.json').read_text())
    scene_perf = json.loads((ROOT / 'PERFORMANCE_SCENE_CPU.json').read_text())
    t4 = json.loads((ROOT / 'PERFORMANCE_T4_V040.json').read_text())['paired_comparison']
    gallery = json.loads((OUTPUT / 'gallery.json').read_text())
    verified = json.loads((ROOT / 'VERIFICATION.json').read_text()) if (ROOT / 'VERIFICATION.json').exists() else {}
    config_notes = {
        'height': 'Output rows, at least 64', 'width': 'Output columns, at least 64',
        'antialias_samples': 'Subpixel samples, 1..16', 'lead_chunk_size': 'Printed slots per chunk, 1..15',
        'max_duration_s': 'Maximum printed duration in seconds',
        'surface_resolution': 'Relief-grid width, 32..256', 'inverse_iterations': 'Coarse-grid steps (minimum 2), or dense output steps, 2..12',
        'shadow_samples': 'Height-field shadow samples, 0..32',
        'geometry_solver': 'coarse_to_fine with two pixel refinements, or dense reference',
    }
    summary = verified.get('release_validation', {})
    current_path = ROOT / 'research/SWARM_VERIFICATION.json'
    if current_path.exists():
        current = json.loads(current_path.read_text())
        if current.get('version') == ecg_gpu.__version__:
            counts = current['tests']
            summary = {**summary, 'portable_tests':
                       f"{counts['passed']} passed, {counts['skipped_cuda']} CUDA skips ({counts['run']} tests run)"}
    replacements = {
        'layout_count': str(len(renderer.layout_specs)), 'figure_count': str(gallery['figures']),
        'layout_table': table(['Layout', 'Short rows', 'Rhythm strips'], [
            [f'`{spec.name}`', ', '.join(str(len(row)) for row in spec.rows), len(spec.rhythm)]
            for spec in renderer.layout_specs]),
        'config_table': table(['Field', 'Default', 'Notes'], [
            [f'`{field.name}`', f'`{getattr(RenderConfig(), field.name)!r}`', config_notes.get(field.name, '')]
            for field in fields(RenderConfig)]),
        'supervision_table': table(['Field', 'Default'], [
            [f'`{field.name}`', f'`{getattr(SupervisionConfig(), field.name)!r}`'] for field in fields(SupervisionConfig)]),
        'api_signatures': '\n\n'.join('### ' + name + '\n\n```text\n' + name + signature + '\n```'
                                      for name, signature in metadata['signatures'].items()),
        'parameters_table': table(['Parameter', 'Shape', 'Dtype'], [
            [f'`{key}`', '`[' + ','.join(map(str, value['shape'])) + ']`', f"`{value['dtype']}`"]
            for key, value in metadata['parameter_shapes'].items()]),
        'performance_table': table(['Preset', 'B', '0.2.1 ms', 'Eager ms', 'Compiled ms', 'Compiled images/s'], [
            [f"{case['preset']} ({case['width']}×{case['height']})", case['batch_size'],
             f"{case['backends']['v021']['median_ms']:.1f}", f"{case['backends']['eager']['median_ms']:.1f}",
             f"{case['backends']['compiled']['median_ms']:.1f}", f"{case['backends']['compiled']['images_s']:.2f}"]
            for case in perf['cases']]),
        'scene_performance_table': table(['Preset', 'B', 'Eager ms', 'Raster ms', 'Raster + scene ms', 'Scene images/s'], [
            [f"{case['preset']} ({case['width']}×{case['height']})", case['batch_size'],
             *[f"{case['backends'][name]['median_ms']:.1f}" for name in ['eager','compiled','compiled_scene']],
             f"{case['backends']['compiled_scene']['images_s']:.2f}"] for case in scene_perf['cases']]),
        't4_performance_table': table(['Preset', 'B', 'Eager ms', 'Compiled ms', 'Compiled images/s'], [
            [f"{case['preset']} ({case['width']}×{case['height']})", case['batch_size'],
             f"{case['backends']['eager']['median_ms']:.1f}", f"{case['backends']['compiled']['median_ms']:.1f}",
             f"{case['backends']['compiled']['images_s']:.2f}"] for case in t4['cases']]),
        'verification_summary': table(['Check', 'Result'], [
            ['Portable library suite', summary.get('portable_tests', 'See VERIFICATION.json')],
            ['Paper audit', 'All eight families, dense numerical reference, see PAPER_AUDIT_CPU.json'],
            ['U-Net example', 'CPU compiled raster and scene, 3 optimization steps, finite loss and changed weights'],
            ['Real patient records used here', '0'], ['CUDA execution', 'Not executed in this CPU environment'],
        ]),
    }
    payload = {'pages': json.loads((ROOT / 'documentation/pages.json').read_text()),
               'replacements': replacements, 'version': ecg_gpu.__version__, 'gallery': gallery}
    staging = ROOT / 'build/documentation'
    staging.mkdir(parents=True, exist_ok=True)
    (staging / 'data.json').write_text(json.dumps(payload, indent=2) + '\n')
    (OUTPUT / 'api-data.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for filename in ['PERFORMANCE_CPU.json', 'PERFORMANCE_SCENE_CPU.json', 'PERFORMANCE_T4_V040.json', 'PAPER_AUDIT_CPU.json', 'PAPER_VISUAL_COMPARISON.json', 'VERIFICATION.json', 'FEATURES.json']:
        shutil.copy2(ROOT / filename, OUTPUT / filename)
    node = os.environ.get('ECG_DOCS_NODE') or shutil.which('node')
    if not node:
        raise RuntimeError('Node.js is required to compile Markdown. Run npm install in documentation first.')
    subprocess.run([node, str(ROOT / 'documentation/build.mjs'), str(staging / 'data.json')], check=True, cwd=ROOT)


if __name__ == '__main__':
    build()
