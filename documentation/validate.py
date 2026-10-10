"""Check built documentation links, anchors, figures, recipes and code syntax."""
import ast
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.issues = []
        self.lang = None
        self.title = False
        self.main = False

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if 'id' in data:
            if data['id'] in self.ids:
                self.issues.append('Duplicate ID: ' + data['id'])
            self.ids.add(data['id'])
        if tag == 'html':
            self.lang = data.get('lang')
        if tag == 'title':
            self.title = True
        if tag == 'main':
            self.main = True
        if tag == 'img' and data.get('src') and not data.get('alt'):
            self.issues.append('Image lacks descriptive alt text')
        for key in ['href', 'src']:
            if key in data:
                self.links.append(data[key])


def validate():
    docs = ROOT / os.environ.get('ECG_DOCS_OUTPUT', 'docs')
    expected = json.loads((ROOT / 'documentation/pages.json').read_text())
    pages = {}
    failures = []
    for page in expected:
        path = docs / (page['id'] + '.html')
        parsed = Page()
        parsed.feed(path.read_text())
        pages[path.resolve()] = parsed
        failures.extend(str(path.name) + ': ' + issue for issue in parsed.issues)
        if parsed.lang != 'en' or not parsed.title or not parsed.main:
            failures.append(path.name + ': missing English document structure')
    links = 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            destination = (path.parent / unquote(url.path)).resolve() if url.path else path
            links += 1
            if not destination.exists():
                failures.append(path.name + ': broken local link ' + link)
            if url.fragment and destination.suffix == '.html' and destination in pages:
                if unquote(url.fragment) not in pages[destination].ids:
                    failures.append(path.name + ': missing anchor ' + link)
    gallery = json.loads((docs / 'gallery.json').read_text())
    if gallery['figures'] != len(gallery['examples']):
        failures.append('Gallery count differs from examples')
    ids = set()
    for item in gallery['examples']:
        if item['id'] in ids:
            failures.append('Duplicate gallery recipe ' + item['id'])
        ids.add(item['id'])
        for name in ['image', 'thumbnail']:
            path = docs / item[name]
            if not path.is_file():
                failures.append('Missing gallery asset ' + item[name])
        actual = hashlib.sha256((docs / item['image']).read_bytes()).hexdigest()
        if actual != item.get('image_sha256'):
            failures.append('Image hash differs for ' + item['id'])
    snippets = 0
    for path in (ROOT / 'documentation/pages').glob('*.md'):
        for index, code in enumerate(re.findall(r'```python\n(.*?)```', path.read_text(), flags=re.S)):
            try:
                ast.parse(code)
                snippets += 1
            except SyntaxError as error:
                failures.append(f'{path.name} snippet {index}: {error}')
    notebook = json.loads((ROOT / 'notebooks/colab_gpu_profile.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    metadata = json.loads((docs / 'api-data.json').read_text())
    for filename, expected_hash in metadata['source_sha256'].items():
        if hashlib.sha256((ROOT / 'ecg_gpu' / filename).read_bytes()).hexdigest() != expected_hash:
            failures.append('Stale API metadata for ' + filename)
    if failures:
        raise AssertionError('\n'.join(failures))
    report = {'status': 'passed', 'pages': len(pages), 'local_links': links,
              'gallery_figures': gallery['figures'], 'python_snippets_parsed': snippets,
              'notebook_python_cells_parse': True, 'figure_hashes_verified': True,
              'api_source_hashes_verified': True, 'browser_visual_qa': False}
    (docs / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    validate()
