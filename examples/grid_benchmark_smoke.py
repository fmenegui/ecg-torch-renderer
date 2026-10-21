"""Exercise experiment export/replay on artificial signals, not an accuracy run."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.grid_benchmark import BenchmarkConfig, load_sources, run, rescore
from ecg_gpu.schema_v05 import LEADS


def smoke():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        signals = root/'signals.npz'
        # Random arrays are software-test inputs only, not physiological ECGs.
        np.savez(signals, signals=np.random.default_rng(4).normal(0, .05, (3, 12, 101)).astype('float32'),
            fs=100., lead_names=LEADS, units='mV', source_ids=['a', 'b', 'c'], group_ids=['p1', 'p2', 'p3'],
            splits=['train', 'validation', 'test'])
        records, _ = load_sources(signals)
        assert len(records) == 3
        Image.new('RGB', (96, 64), 'white').save(root/'external.png')
        external = root/'external.json'
        item = {'id': 'artificial-external', 'image': 'external.png',
                'sha256': hashlib.sha256((root/'external.png').read_bytes()).hexdigest(),
                'source_id': 'external-test', 'group_id': 'external-test'}
        external.write_text(json.dumps([item, dict(item, id='artificial-annotated', points_xy=[[10., 10.]],
            annotation_region=[5, 5, 15, 15], annotation_method='Artificial software-test annotation')]))
        config = BenchmarkConfig(height=64, width=96, views=2, crop_size=32, steps=2, batch_size=1, spp=1)
        first = run(signals, root/'first', config=config, external=external)
        second = run(signals, root/'second', config=config, cache=root/'first', ablations=True, external=external)
        third = rescore(root/'first', root/'rescored', external=external)
        assert len(first['models']) == len(third['models']) == 3
        assert len(second['models']) == 7
        assert first['status'] == second['status'] == third['status'] == 'completed'
        for a, b, c in zip(first['models'], second['models'], third['models']):
            assert a['history'] == b['history']
            assert a['by_backend'] == b['by_backend'] == c['by_backend']
            assert a['external'] == b['external'] == c['external']
        return {'status': 'passed', 'models': 3, 'cache_replay_equal': True, 'accuracy_benchmark': False}


if __name__ == '__main__':
    print(json.dumps(smoke()))
