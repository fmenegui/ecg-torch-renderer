# Detailed profiling

One command exports a benchmark report, operator tables, CUDA kernel events when available, allocator peaks and a compressed trace. Instrumented profile passes are separate from timed benchmark passes.

## Profile the renderer

```bash
ecg-renderer-profile --device cuda --preset balanced --batch-size 4 \
  --compile-raster --compile-scene --warmup 5 --steps 30 --profile-steps 3 \
  --out profile-renderer
```

## Profile a complete training step

```bash
ecg-renderer-profile --device cuda --preset balanced --batch-size 4 \
  --compile-raster --compile-scene --train-step --stacks --memory-timeline \
  --out profile-train
```

The CLI training workload includes a small artificial CNN, forward, loss, backward and AdamW. Use the Python API below for your actual U-Net or other production model.

## Profile your own workload

```python
from ecg_gpu import profile_layer

def train_step():
    batch = synth(signals)
    optimizer.zero_grad(set_to_none=True)
    logits = unet(batch["image"])
    loss_map = F.binary_cross_entropy_with_logits(logits, batch["lead_masks"], reduction="none")
    loss = (loss_map * batch["valid"]).sum() / (batch["valid"].sum() * logits.shape[1]).clamp_min(1)
    loss.backward()
    optimizer.step()
    return loss.detach()

report = profile_layer(
    synth, signals, "profile-unet",
    workload=train_step,
    workload_name="ECGSupervisionLayer + U-Net + backward + AdamW",
    steps=30, warmup=5, profile_steps=3,
)
```

Use a disposable model/optimizer/renderer instance when profiling. Warmup, timing and profiling all execute the workload and advance its state. The output directory must be empty.

## Read the artifacts

| File | Purpose |
| --- | --- |
| `report.json` | Runtime, source hashes, config, median/p90 latency, throughput and memory peaks |
| `operators.csv` | Operator calls, shapes, host/device time and allocation information |
| `kernels.csv` | Captured CUDA kernel names, call counts and event totals |
| `tables.txt` | Readable top-operator tables |
| `trace.json.gz` | Compressed Chrome/Perfetto trace |
| `memory-timeline.json.gz` | Optional allocation timeline, when supported |
| `<output-directory>.zip` | Portable results bundle |

Decompress the trace and open it in [Perfetto](https://ui.perfetto.dev/). Parent and child region totals overlap. CUDA kernel counts come only from complete trace events with category `kernel`. GPU user annotations, memcpy/memset events and CUDA API calls are excluded. Version 0.4.0 incorrectly included GPU annotation spans in kernel totals. Operator tables retain device type and annotation flags. Concurrent GPU events can overlap. Do not add these totals to estimate wall latency.

## Measurement boundaries

The first call is recorded separately. Warm benchmark calls exclude profiler instrumentation, encoding and DataLoader work unless your custom workload includes them. CUDA timing uses events and synchronization. CPU reports keep CUDA measurements null.

Allocator peaks are not DRAM traffic or total process GPU memory. Hardware bandwidth, SM occupancy and utilization counters are not collected. CUDA kernel activity needs a working CUPTI/runtime profiler setup. The report records whether events were actually captured.

Signals and patient metadata are not exported by the profiler. Your own workload remains responsible for any additional logging or files it writes.
