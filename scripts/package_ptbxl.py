"""Package the publisher's PTB-XL 100 Hz bytes, verified against our acquired copy."""
import argparse
import array
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

EXPECTED = 'b08d272ea3b43fe6bdf68b2f5086c4a0b89c2d0b82b2c2705cf33f58742f0a68'
METADATA = '7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def package(source, out):
    rows = list(csv.DictReader((source / 'ptbxl_database.csv').open()))
    assert len(rows) == 21799 and sha(source / 'ptbxl_database.csv') == METADATA
    out.mkdir(parents=True, exist_ok=True)
    inventory = []
    parts = []
    for start in range(0, len(rows), 5000):
        name = f'ptb-xl-1.0.3-records100-{start // 5000:03d}.zip'
        with zipfile.ZipFile(out / name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for row in rows[start:start + 5000]:
                stem = row['filename_lr']
                assert len(Path(stem).parts) == 3 and Path(stem).parts[0] == 'records100'
                assert '..' not in Path(stem).parts
                header = (source / (stem + '.hea')).read_text().splitlines()
                record, leads, fs, count = header[0].split()[:4]
                assert record == Path(stem).name
                assert (int(leads), int(fs), int(count)) == (12, 100, 1000)
                data = (source / (stem + '.dat')).read_bytes()
                assert len(data) == 24000
                samples = array.array('h')
                samples.frombytes(data)
                if sys.byteorder != 'little':
                    samples.byteswap()
                assert len(header[1:13]) == 12
                for i, line in enumerate(header[1:13]):
                    fields = line.split()
                    assert samples[i] == int(fields[5])
                    assert sum(samples[i::12]) % 65536 == int(fields[6]) % 65536
                for suffix in ('.hea', '.dat'):
                    path = source / (stem + suffix)
                    info = zipfile.ZipInfo(stem + suffix, date_time=(2022, 11, 7, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes(), compresslevel=4)
                    inventory.append({'path': stem + suffix, 'bytes': path.stat().st_size,
                                      'sha256': sha(path), 'archive': name})
        parts.append({'path': name, 'bytes': (out / name).stat().st_size,
                      'sha256': sha(out / name), 'records': len(rows[start:start + 5000])})
        print(json.dumps(parts[-1]), flush=True)
    for name in ('ptbxl_database.csv', 'scp_statements.csv', 'LICENSE.txt'):
        shutil.copy2(source / name, out / name)
        inventory.append({'path': name, 'bytes': (out / name).stat().st_size,
                          'sha256': sha(out / name), 'archive': None})
    payload = ''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n"
                      for r in sorted(inventory, key=lambda r: r['path'])).encode()
    assert len(inventory) == 43601 and hashlib.sha256(payload).hexdigest() == EXPECTED
    manifest = {'source': 'https://physionet.org/content/ptb-xl/1.0.3/',
                'download_source': 's3://physionet-open/ptb-xl/1.0.3/', 'version': '1.0.3',
                'signal_hz': 100, 'records': len(rows), 'source_inventory_sha256': EXPECTED,
                'archives': parts, 'files': inventory}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / 'datasets/ptbxl/README.md', out / 'README.md')
    shutil.copy2(root / 'scripts/verify_ptbxl.py', out / 'verify_ptbxl.py')
    sums = ''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file())
    (out / 'SHA256SUMS').write_text(sums)
    print(json.dumps({'status': 'original-bytes-verified', 'records': len(rows), 'files': len(inventory)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    package(args.source, args.out)
