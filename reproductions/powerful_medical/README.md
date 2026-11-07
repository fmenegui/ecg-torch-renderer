# Dotter/Gridder public-description reconstruction

Sources:
[published paper](https://doi.org/10.1016/j.jelectrocard.2025.153900),
[preprint](https://www.medrxiv.org/content/10.1101/2024.08.31.24312876v1.full),
[US20240144417A1](https://patents.google.com/patent/US20240144417A1/en),
[US12614244B2](https://patents.google.com/patent/US12614244B2/en).

This code is an independent reconstruction from public descriptions. It is not
the commercial PMcardio source or a verified replacement for its implementation.

## Disclosure versus engineering choice

| Mechanism | Implemented | Remaining ambiguity |
| --- | --- | --- |
| Dotter | ResBlock U-Net, SiLU, BCE logits, Adam at 0.005, 256-pixel patches, 300-epoch adaptation entrypoint. | Exact widths, normalization, trained weights and original 176-image annotation bank are not available here. Current augmentation/sampling is explicitly reduced. |
| Dot decoding | Threshold, connected regions, intensity-weighted subpixel centroids and largest-component-size ranking. | Mask widths and confidence calibration are not established. |
| Gridder | Nearest-neighbour directions, probabilistic directional hypotheses, reciprocal edges, unit cycles and BFS integer labels. | Global four-cluster likelihood, thresholds and conflict rejection are our declared numerical interpretation. |
| Missing nodes | Local regression inside the observed hull, separately flagged. | Regression family, neighbourhood and commercial acceptance policy are unverified. |
| Undistortion | Per-cell four-corner projective inverse mapping and original RGB sampling. | The disclosure does not uniquely fix all interpolation, boundary or sampling conventions. |

`dotter.py` uses widths `[32,64,128,256]` and GroupNorm. These are explicit
choices, not parameters attributed to the company. `train.py` labels a run on
other annotations as adapted training. Its patch sampling and horizontal-flip
augmentation are not a reproduction of the original augmented patch bank.

`gridder.py` returns a relative chart. A periodic square grid does not establish
semantic page orientation, absolute origin, major/minor pitch or ECG gain and
speed. Inferred nodes stay distinct from observed nodes. The partial diagnostic
raster has a support mask and does not claim complete-image preservation.

```bash
python -m reproductions.powerful_medical.controls --out build/pm-controls

python -m reproductions.powerful_medical.train --annotations annotations.csv \
  --out /workspace/pm-dotter-adaptation

python -m reproductions.powerful_medical.infer --image photograph.png \
  --checkpoint /workspace/pm-dotter-adaptation/best.pt --out /workspace/pm-normalization

# Geometry-only diagnostic, explicitly using known points:
python -m reproductions.powerful_medical.run --image photograph.png \
  --points exact_points.npy --oracle-points --out build/pm-oracle-diagnostic
```

The annotation CSV requires `image,mask,image_sha256,mask_sha256,patient_id,split`.
Every split must have a verified independent patient identity. The original
annotation bank must not be silently substituted with our synthetic renderer.

## Executed controls and comparison boundary

The analytical controls use perfect points and, separately, known cell IDs.
Flat and projective mappings are checked against independent analytical maps.
Smooth within-cell curvature retains measurable approximation error. These
are mechanism tests. They do not measure learned Dotter accuracy.

The stronger projective control is a retained failure of our current global
direction interpretation. It retains only seven of 88 perfect points and does
not assign correct relative IDs throughout that component. Removing a complete
row also reduces retained support. These findings constrain this reconstruction,
and do not establish the behavior of the unavailable commercial implementation.

The paper reports full waveform performance using additional reconstruction
and lead assignment, including its Leader network. The reported PCC above
0.91, SNR above 12.5 dB, RMSE below 0.10 mV and mean failure rate 6.62% cannot
be reproduced by a Gridder-only run. The full system was not executed here.

To test those claims, reconcile the full original preprocessing, alignment,
lead assignment, failure definition, aggregation and PM-ECG-ID page semantics.
An independently reconstructed downstream extractor would be another declared
adaptation. The normalization component can meanwhile be compared using a
fixed common extractor and independently measured geometry, in separate tables.
