"""Replay geometry on frozen detections without rerunning a neural network."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.rectify_denoising_grid import write_geometry_output


def preview(rgb, points, output):
    with np.load(points) as data:
        xy, scores, chosen, ids, cells = [data[k] for k in ('xy', 'scores', 'selected', 'ids', 'cells')]
    local = {'selected_indices': chosen.tolist(), 'ids': ids.tolist(),
             'cells': cells.tolist(), 'partial_status': 'partial' if len(cells) else 'rejected'}
    np.savez_compressed(output/'points.npz', xy=xy, scores=scores, selected=chosen, ids=ids, cells=cells)
    result = write_geometry_output(rgb, xy, scores, local, output, pixels_per_mm=2.)
    result['detector_reexecuted'] = False
    result['points_source'] = str(points)
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def run(root):
    manifest = {r['key']: r for r in json.loads((root/'diagnostic/manifest.json').read_text())}
    evaluation = root/'complete-preserved-evaluation'
    selection = json.loads((evaluation/'selection.json').read_text())
    records = []
    for record in json.loads((evaluation/'test.json').read_text())['cases']:
        key = record['key']
        with np.load(root/'diagnostic'/manifest[key]['file']) as data:
            image = data['image']
        output = root/'content-preserving-previews/native'/key
        output.mkdir(parents=True, exist_ok=True)
        result = preview(image, root/'diagnostic-evaluation/topologies'/(key+'-clean85.npz'), output)
        records.append({'key': key, 'kind': 'native-retrospective', 'status': result['status'],
                        'folder': str(output), 'metrics': record['result']['eligibility'][str(selection['selected_radius'])],
                        'baseline': record['baseline']})
        print(key, result['status'], flush=True)
    for path in sorted((root/'comparison/inputs').glob('*.jpg')):
        key = path.stem
        output = root/'content-preserving-previews/comparison'/key
        output.mkdir(parents=True, exist_ok=True)
        result = preview(np.asarray(Image.open(path).convert('RGB')),
                         root/'comparison/ours'/key/'points.npz', output)
        records.append({'key': key, 'kind': 'cached-reference', 'status': result['status'], 'folder': str(output)})
        print(key, result['status'], flush=True)
    (root/'content-preserving-previews/manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(**vars(parser.parse_args()))
