# Second independent supervision review

Date: 2026-10-18. Scope: `gpu_supervision.py`, printed-instance ownership,
semantic lead channels, pixel boxes and observation visibility. The review is
separate from the capture-realism assessment.

## Finding

The renderer already supplies the two complementary representations needed
for segmentation and detection. The review found no lead-identity or
per-instance box error in the exercised layouts. It added independent
regression evidence and a clear error for zero-height or zero-width masks.

| Target | Definition | Training use |
| --- | --- | --- |
| `lead_masks[B,12,H,W]` | Coverage of waveform ink grouped by canonical lead | Twelve-channel sigmoid segmentation |
| `lead_boxes[B,12,4]` | Envelope of every occurrence of one lead | Lead-level localization only |
| `instance_masks[B,15,H,W]` | One mask for each short or rhythm occurrence | Instance segmentation or strip extraction |
| `instance_boxes[B,15,4]` | Tight envelope of each printed occurrence | Individual-strip detection |
| `instance_lead_ids[B,15]` | Canonical lead ID for each fixed slot | Detection class, including repeated leads |
| `instance_active[B,15]` | Slot enabled by the layout before capture | Padding and source-presence information |
| `instance_box_valid[B,15]` | Target pixels above threshold survive capture | Detection validity filter |

Canonical class order is `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
Repeated rhythm strips have separate slots and may share the same class.
Cabrera's printed `-aVR` retains class ID 3 and publishes amplitude sign -1.
No physiological signal is modified by the supervision interface.

`SupervisionConfig.leads` reorders or subsets semantic channels and semantic
boxes only. Instance outputs retain all printed slots. Inactive slots still
have canonical IDs, so class ID alone is not a presence test. A detector must
use `instance_active & instance_box_valid`.

## Box and mask conventions

Boxes are integer pixel-edge `[x0,y0,x1,y1]` with exclusive right/bottom edges.
A one-pixel object at the bottom-right pixel has box `[W-1,H-1,W,H]`.
Empty supervision boxes contain `-1` and have false validity flags. The legacy
renderer metadata key `instance_boxes_xyxy` instead uses zero for empty slots.
Training adapters should consume the supervised batch keys with their validity
flags, rather than mixing these two interfaces.

The default coverage threshold is strictly `> 0.05`. Binary masks apply the
same threshold after visibility. Soft masks retain antialiasing. Lead masks
are multilabel, since two leads may cross and both own the same pixel. A
single integer semantic class image and mutually exclusive softmax loss
cannot preserve those crossings.

Targets describe waveform strokes, not lead-name text, calibration pulses,
cell backgrounds or the grid. Thus a flat trace can have a very short box.
If a downstream detector needs an allocated layout cell including text and
calibration, that is a different target definition and is not currently
provided by these keys.

`lead_boxes` can span disconnected short and long occurrences. This is a valid
lead-level envelope but an unsuitable target for detection of individual
printed strips. The regression suite includes a lead II repeated three times
as rhythm strips, verifying four distinct same-class detection objects.

## Visibility and geometry

Visible masks equal projected waveform coverage multiplied by the same
visibility map used for paper loss and explicit image occlusion. Tears, frame
cropping, opaque blocks, pen annotations and added text can remove target
pixels. An active printed slot may become completely invisible, leaving no
valid box. `visible_only=False` retains projected ink behind explicit
occluders, where its image-space projection exists.

Blur, sensor noise, thermal fading and JPEG degradation do not redefine the
geometric waveform target. The visibility map does not promise readability,
contrast or clinical recoverability. Heavily degraded samples may therefore
have correct geometric targets while offering little recoverable evidence.

Semantic coverage is reduced across same-lead occurrences before camera
resampling. Instance coverage is resampled separately. Bilinear interpolation
does not generally commute with a maximum, so exact semantic-versus-instance
union equality is asserted for clean geometry, not as a universal floating
point identity after warping.

## Independent checks

The new `tests/test_supervision_integrity.py` enumerates occupied coordinates
to derive a reference box. This differs from the production row/column
reduction algorithm. It checks every standard layout, inactive slots,
Cabrera class and sign, lead subsets, repeated same-class rhythms, sparse
masks, exact threshold edges and exclusive pixel bounds. Additional cases
exercise perspective, crop, tear, pen, partial and total occlusion, visible
versus amodal output, target immutability and no host transfers/scalar reads.

Validation command:

```bash
python -m unittest discover -s tests -p 'test_supervision*.py' -v
```

Observed result: **20 tests ran, 19 passed, 1 skipped** in 3.102 s on CPU.
The skipped test requires CUDA. No GPU execution or real-image downstream
generalization was established by this review. Target alignment is an
algorithmic contract and does not by itself establish photographic realism.
