# Documentation source

The guide is authored in English Markdown and compiled into static HTML with
Marked. API defaults, signatures and parameter shapes are derived from the
Python package. Gallery images are actual renderer outputs from artificial
signals, with reproducible recipes and file hashes.

```bash
python -m pip install -e .
npm install --prefix documentation
python documentation/build.py
python documentation/validate.py
python -m http.server 8000 --directory docs
```

Regenerate the image bank when renderer changes affect its appearance:

```bash
python documentation/gallery.py --device cpu --out docs
python documentation/supervision_gallery.py --device cpu --out docs
```

The first script renders feature examples. The second appends the aligned
BW/WB, channel and box views. Both accept CUDA and optional compiled raster.
Encoding happens after rendering. `pages.json` defines navigation. `pages/`
contains content, and `assets/` contains the design and browser enhancement.

Deploy the `docs/` output with any static host. Images and pages work without
JavaScript. Search, code copying, filtering, themes and the recipe dialog add
browser interactions. No third-party analytics or client services are required.
