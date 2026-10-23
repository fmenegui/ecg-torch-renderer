"""Package measured data and native outputs, excluding source/checkpoint copies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    audit = json.loads((args.evidence/'qa/diagnostic-results-review.json').read_text())
    if audit['status'] != 'pass' or audit['actual_case_method_results'] != 160:
        raise ValueError('Require independent audit of the complete diagnostic panel')
    entries = {}
    def add(path, name):
        path = Path(path)
        if not path.is_file():
            raise ValueError('Missing archive member '+str(path))
        if name in entries:
            raise ValueError('Duplicate archive member '+name)
        entries[name] = path
    for name in ('lock.json', 'protocol.json', 'schedule.json'):
        add(args.execution/name, 'comparison/'+name)
    add(args.evidence/'runtime-hardware.json', 'runtime-hardware.json')
    for name in ('m3-diagnostic.log', 'm3-development-smoke.log', 'm3-comparison-contract-tests.log', 'm3-scenarios.log'):
        add(args.execution.parent/name, 'logs/'+name)
    for name in ('saved-fidelity-note.md', 'saved-fidelity-audit.json'):
        add(args.evidence/'npj'/name, 'npj/'+name)
    add(args.evidence/'kaggle/author-normalized-wrapper-smoke/report.json', 'kaggle/positive-author-execution-report.json')
    add(args.evidence/'kaggle/analytic-controls.json', 'kaggle/analytic-controls.json')
    for path in (args.execution/'queries').glob('*.npz'):
        add(path, 'comparison/queries/'+path.name)
    rows = json.loads((args.execution/'schedule.json').read_text())['cases']
    keys = []
    input_stage = args.out.parent/'m3-original-inputs'
    input_stage.mkdir(exist_ok=True)
    for row in rows:
        folder = args.execution/'cases'/row['key']
        if not folder.exists():
            continue
        keys.append({'key': row['key'], 'split': row['split'], 'input_archive_sha256': row['sha256']})
        with np.load(args.bank/row['file'], allow_pickle=False) as data:
            original = data['image'].copy()
        image_path = input_stage/(row['key']+'.png')
        Image.fromarray(original).save(image_path)
        add(image_path, 'inputs/'+image_path.name)
        if (folder/'fresh-observations.npz').exists():
            add(folder/'fresh-observations.npz', 'comparison/cases/'+row['key']+'/fresh-observations.npz')
        for method_folder in sorted(p for p in folder.iterdir() if p.is_dir()):
            result_path = method_folder/'result.json'
            if not result_path.exists():
                raise ValueError('Partial case cannot be packaged as complete')
            result = json.loads(result_path.read_text())
            prefix = 'comparison/cases/'+row['key']+'/'+method_folder.name+'/'
            for path in method_folder.iterdir():
                if path.suffix in ('.json', '.npz', '.npy'):
                    add(path, prefix+path.name)
            # Preserve any produced native raster even if a later stage fails.
            # The result receipt, not the existence of a PNG, defines success.
            for name in ('rectified-original-rgb.png', 'rectified-native.png', 'original-preserved.png'):
                if (method_folder/name).exists():
                    add(method_folder/name, prefix+name)
    for path in args.summary.iterdir():
        if path.suffix in ('.json', '.csv'):
            add(path, 'summary/'+path.name)
    for label, directory in [('composition', args.evidence/'our-critique/composition-results'),
                             ('primitives', args.evidence/'dotter-gridder/component-results'),
                             ('sampling', args.evidence/'sampling'), ('qa', args.evidence/'qa')]:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.csv', '.npz', '.png', '.svg', '.md', '.log', '.txt'):
                add(path, label+'/'+str(path.relative_to(directory)))
    for path in args.out.parent.glob('m3-*.png'):
        add(path, 'figures/'+path.name)
    for path in args.out.parent.glob('m3-*.svg'):
        add(path, 'figures/'+path.name)
    add(args.evidence/'our-critique/structural-controls.json', 'structural-controls.json')
    add(args.evidence/'our-critique/composition-protocol.json', 'composition/protocol-pre-results.json')
    manifest = {'schema': 'M3-measurement-package-v1', 'diagnostic_case_method_results': 160,
                'diagnostic_captures': 32, 'development_smoke_captures': 1,
                'input_captures': keys, 'third_party_weights_or_source_included': False,
                'source_and_reproduction': 'https://github.com/fmenegui/ecg-torch-renderer/pull/22',
                'lock_sha256': audit['lock_sha256'],
                'members': [{'name': name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
                            for name, path in sorted(entries.items())]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2)+'\n')
        archive.writestr('README.txt', 'M3 diagnostic evidence, 2026-10-23.\n'
            '32 retrospective synthetic captures from eight patients, five native configurations.\n'
            'Development smoke is separate, no complete development aggregate.\n'
            'See summary/cases.csv and summary/summary.json, then qa/final-review.md.\n'
            'Original source/prediction bytes are locked. Commercial Dotter/Gridder not executed.\n'
            'Sampling and composition are separate analytical component controls, not new native benchmark arms.\n'
            'No external code/checkpoints included. Own code and attribution: PR22 linked in manifest.\n')
    receipt = {'path': str(args.out), 'bytes': args.out.stat().st_size,
               'sha256': digest(args.out), 'members': len(entries)+2,
               'lock_sha256': audit['lock_sha256']}
    args.out.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'bank', 'summary', 'evidence', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    build(p.parse_args())
