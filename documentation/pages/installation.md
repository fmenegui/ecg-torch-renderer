# Installation

Install a PyTorch distribution appropriate for your environment first. CPU and CUDA builds have different runtime requirements. Follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Install from source

```bash
git clone https://github.com/fmenegui/ecg-torch-renderer.git
cd ecg-torch-renderer
python -m pip install -e .
```

Python 3.10 or newer is required. The runtime dependencies are Torch ≥2.6, NumPy ≥1.24 and Pillow ≥10.1. Optional WFDB input is available with:

```bash
python -m pip install -e '.[wfdb]'
```

## Verify the device

```python
import torch
import ecg_gpu

print(ecg_gpu.__version__)
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Move the module once with `.to(device)`. Move every input batch to the same device. Keep the renderer buffers in float32. Use AMP around the downstream network.

## Run a working example

```bash
python examples/sequential.py
python examples/segmentation.py --steps 3
python -m unittest discover -s tests -v
```

The examples use artificial signals and work on CPU. The segmentation example accepts `--device cuda --compile-raster --compile-scene` for your GPU run.

## Google Colab

Open the [GPU profiling notebook](https://colab.research.google.com/github/fmenegui/ecg-torch-renderer/blob/main/notebooks/colab_gpu_profile.ipynb), choose a GPU runtime, then run all cells. The notebook checks the selected runtime, generates images, benchmarks the backends and exports a results ZIP. Hardware availability and performance depend on the assigned runtime.

## Build this documentation

```bash
cd documentation
npm install
cd ..
python documentation/build.py
python documentation/validate.py
python -m http.server 8000 --directory docs
```

The site is static. It can be served from the generated `docs/` directory. Search, figure inspection and code copying run locally in the browser, with no telemetry or image uploads.
