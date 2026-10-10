# Train an image model

Load signals, move them to the training device, render there, then pass RGB to your network. DataLoader workers can read and preprocess source files. Keep CUDA rendering in the training process.

## U-Net segmentation

```python
import torch
from torch.nn import functional as F
from ecg_gpu import ECGSupervisionLayer, RenderConfig

synth = ECGSupervisionLayer(
    fs=500, config=RenderConfig(height=384, width=512)
).to(device)
optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-4)

for signals in signal_loader:
    signals = signals.to(device, non_blocking=True)
    batch = synth(signals)
    optimizer.zero_grad(set_to_none=True)
    logits = unet(batch["image"])       # [B,12,H,W]
    loss_map = F.binary_cross_entropy_with_logits(
        logits, batch["lead_masks"], reduction="none"
    )
    support = batch["valid"]
    loss = (loss_map * support).sum() / (
        support.sum() * logits.shape[1]
    ).clamp_min(1)
    loss.backward()
    optimizer.step()
```

Every lead is an independent output channel. Keep logits unbounded during training and use `.sigmoid()` for predicted probabilities at evaluation. Trace pixels are sparse, so tune class weighting or an additional Dice objective against validation data. A finite loss and a successful backward step do not establish good digitization performance.

## Run the included U-Net example

```bash
python examples/segmentation.py --steps 3
python examples/segmentation.py --device cuda --compile-raster --steps 20
```

The complete example defines a small encoder, bottleneck, upsampling path and skip connection. It verifies that a 12-channel output receives gradients and that optimization changes network weights. It uses artificial waveforms.

## Classification with nn.Sequential

```python
from torch import nn
from ecg_gpu import ECGImageLayer

pipeline = nn.Sequential(
    ECGImageLayer(fs=500, config=RenderConfig(height=384, width=512)),
    image_network,
).to(device)
logits = pipeline(signals)
```

Use the RGB-only layer when your model does not need masks. It avoids allocating dense supervision targets.

## Mixed precision

```python
synth = synth.to(device)              # preserve float32 renderer buffers
with torch.autocast(device_type="cuda", dtype=torch.float16):
    batch = synth(signals)
    logits = unet(batch["image"])
    loss_map = F.binary_cross_entropy_with_logits(logits, batch["lead_masks"], reduction="none")
```

The renderer disables autocast internally and returns float32 geometry and pixels. The network can still use AMP. In a full fp16 training loop, use `torch.amp.GradScaler("cuda")` for backward and optimizer stepping. Do not call `.half()` on the renderer or on a Sequential pipeline containing it.

## Detection and waveform recovery

Use instance boxes and IDs to train a detector. Use `instance_windows_s` and `instance_amplitude_sign` to associate detected printed windows with source samples. Aggregate per-lead masks only when repeated occurrences should share one output channel.

The reference signal is not automatically a waveform-regression target for every crop. Supervise only the source windows that your image/task definition makes observable.

## Distributed training

Create one synthesizer per rank, on that rank's device. Give each rank a distinct seed, for example `base_seed + 1009 * rank`. Save renderer state alongside model and optimizer state. Restore per-rank randomization counters when exact continuation matters.

DDP can wrap the trainable image network. The synthesizer has buffers and no trainable parameters. For data loading, use pinned memory and nonblocking transfers when appropriate. Measure the full training step before selecting renderer batch size or resolution.
