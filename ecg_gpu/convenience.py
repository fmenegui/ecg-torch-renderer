"""Exploration and data-preparation helpers, outside the training hot path.

``ECGImageLayer`` and ``ECGSupervisionLayer`` remain the training interface.
These helpers cover the common first steps: turning arrays in any lead order
or unit into a ``[B,12,T]`` mV tensor, rendering a few images in one call,
converting results to PIL images and listing renderer parameters.
"""
import difflib

import torch

from .schema_v05 import LEADS

UNIT_SCALE = {'mv': 1., 'uv': .001, 'µv': .001, 'v': 1000.}


def did_you_mean(value, choices):
    """Short hint naming the closest valid choices, or an empty string."""
    originals = {}
    for choice in choices:
        originals.setdefault(str(choice).lower(), str(choice))
    matches = difflib.get_close_matches(str(value).lower(), list(originals), n=3, cutoff=.6)
    if not matches:
        return ''
    return ' Did you mean ' + ' or '.join(repr(originals[match]) for match in matches) + '?'


def lead_index(name):
    """Index of a lead name in the standard order, case-insensitive."""
    key = str(name).strip().upper()
    for index, lead in enumerate(LEADS):
        if lead.upper() == key:
            return index
    raise ValueError(f'Unknown lead name {name!r}. Standard leads are {LEADS}.' + did_you_mean(name, LEADS))


def prepare_signals(values, *, lead_names=None, units='mV', device=None, dtype=torch.float32):
    """Return a ``[B,12,T]`` mV tensor in standard lead order.

    ``values`` may be a NumPy array, nested list or tensor shaped ``[12,T]`` or
    ``[B,12,T]``. ``lead_names`` gives the order of the twelve input channels,
    for example WFDB ``sig_name``. ``units`` is mV, uV/µV or V.
    """
    signals = torch.as_tensor(values, dtype=dtype, device=device)
    if signals.ndim == 2:
        signals = signals[None]
    if signals.ndim != 3 or signals.shape[0] < 1 or signals.shape[1] != 12 or signals.shape[2] < 2:
        raise ValueError(f'Signals must be shaped [12,T] or [B,12,T] with T>=2; got {tuple(signals.shape)}.'
                         ' If channels are last, transpose with values.T or values.transpose(0, 2, 1)')
    if lead_names is not None:
        names = list(lead_names)
        if len(names) != 12:
            raise ValueError(f'lead_names must name the 12 input channels; got {len(names)}')
        order = [lead_index(name) for name in names]
        if sorted(order) != list(range(12)):
            raise ValueError(f'lead_names must contain each standard lead once; got {names}')
        # order[c] is the standard position of input channel c.
        inverse = [order.index(position) for position in range(12)]
        signals = signals[:, inverse]
    scale = UNIT_SCALE.get(str(units).strip().lower())
    if scale is None:
        raise ValueError(f'Unsupported units {units!r}. Use mV, uV or V')
    signals = signals * scale
    if not torch.isfinite(signals).all():
        raise ValueError('Signals contain NaN or infinite values. Fill or crop them before rendering')
    return signals.contiguous()


def describe_parameters(renderer=None):
    """Map each rendering parameter name to its per-record shape and dtype.

    Every name is accepted in ``ECGImageLayer(parameters=...)``,
    ``capture_scenario`` overrides and ``ECGRenderer.forward(parameters=...)``.
    """
    from .gpu_renderer import ECGRenderer, RenderConfig
    if renderer is None:
        renderer = ECGRenderer(RenderConfig(height=64, width=96, surface_resolution=32, shadow_samples=0))
    generator = torch.Generator(device=renderer.device).manual_seed(0)
    sampled = renderer.sample_parameters(1, generator=generator)
    return {name: {'shape': tuple(value.shape[1:]), 'dtype': str(value.dtype).replace('torch.', '')}
            for name, value in sorted(sampled.items())}


def render_ecg(signals, fs=500., *, scenario=None, severity=.6, height=768, width=1024,
               seed=0, device=None, lead_names=None, units='mV', parameters=None):
    """Render images in one call for exploration, returning ``[B,3,H,W]`` in [0,1].

    ``scenario`` is one of ``CAPTURE_SCENARIOS`` or None for the default
    mixed distribution. ``parameters`` overrides or adds rendering parameters
    (see ``describe_parameters``). The same seed reproduces the same capture.
    Build an ``ECGImageLayer`` once for training instead of calling this per batch.
    """
    from .gpu_module import ECGImageLayer
    from .gpu_renderer import RenderConfig
    from .scenarios import capture_scenario
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    batch = prepare_signals(signals, lead_names=lead_names, units=units, device=device)
    if scenario is None:
        config, options = RenderConfig(height=height, width=width), {}
    else:
        config, options = capture_scenario(scenario, height=height, width=width, severity=severity)
    options.update(parameters or {})
    layer = ECGImageLayer(fs, config=config, parameters=options, seed=seed, randomize=False).to(device)
    layer.eval()
    with torch.no_grad():
        return layer(batch)


def to_pil(images):
    """Convert ``[3,H,W]`` or ``[B,3,H,W]`` images in [0,1] to a list of PIL images."""
    from PIL import Image
    if not isinstance(images, torch.Tensor):
        raise TypeError('to_pil expects a torch tensor shaped [3,H,W] or [B,3,H,W]')
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError(f'Expected [B,3,H,W] or [B,1,H,W]; got {tuple(images.shape)}')
    pixels = (images.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    # A trailing channel of 3 gives RGB; a squeezed single channel gives L.
    return [Image.fromarray(item.permute(1, 2, 0).squeeze(-1).numpy()) for item in pixels]
