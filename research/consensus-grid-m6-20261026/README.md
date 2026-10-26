# Image-only grid organization and ECG-text orientation

M5 showed that perfect point localization does not establish correct physical
grid coordinates. This cycle implements two separate opt-in improvements and
evaluates them on identical detector observations with the frozen M3 queries.

`organize_consensus_lattice` tests competing axial bases, permits different
horizontal/vertical pitch, follows local spacing changes and integrates integer
constraints in confidence order. Disconnected observed components may be aligned
only after bounded, redundant projective agreement. Missing nodes and cells are
not generated. All competing hypotheses and the selected structural score are
returned. Geometry alone leaves semantic page orientation unresolved.

`infer_text_orientation` reads original-image ECG labels in four quarter
rotations through explicitly requested local Tesseract English OCR. Two distinct
strong lead labels or two recognized header words, a minimum score and a margin
over the runner-up are required. Repeated labels do not create extra evidence.
An incomplete OCR execution, weak evidence or ambiguity leaves orientation
unresolved. Raw TSV alternatives, runtime hashes and the image hash are retained.
OCR preprocessing never replaces the original RGB sampling source.

## Use

Install the existing `digitization` extras and a local Tesseract executable with
English language data to request text orientation. Neither feature downloads a
network or modifies the original detector weights.

```bash
python examples/rectify_denoising_grid.py \
  --checkpoint /path/to/joint-1973.pt \
  --image /path/to/ecg.png --out build/consensus-ecg \
  --organizer consensus --page-orientation text --pixels-per-mm 4
```

The existing organizer and no semantic orientation correction remain the CLI
defaults. Physical pitch still assumes major 5 mm intersections. Neither the
OCR confidence nor the geometric support mask is a calibrated error probability.
The full-frame writer samples original RGB and maintains separate geometric
evidence. It preserves the original when its complete map is rejected.

## Evaluation contract

Development uses the 24 frozen validation captures from six patient groups.
The 32 already inspected M3 test captures from eight other patient groups are
a retrospective evaluation, not a new unseen holdout. The detector weights and
all query archives remain unchanged. Fresh validation detections are cached
once and reused in every arm. Only RGB enters OCR and only detector coordinates
and confidence enter grid inference. Gold fields and query references reach the
scorer after inference. No smoothing, waveform restoration or new training is
part of this comparison.

Four arms isolate the existing organizer, the consensus organizer, text
orientation with the existing organizer and the combination. Primary scores use
native orientation and one anchor-only translation at 0.5 mm. Secondary D4
registration is diagnostic only. Complete RGB output, coordinate eligibility
and partial-cell scores are evaluated separately. All scheduled failures must
stay in the denominator. Results and reproduction commands are added after the
locked executions complete.

The scope is synthetic development geometry. There is no claim of clinical
validation, improved waveform accuracy, automatic lead layout/calibration or
superiority to an external complete digitizer.
