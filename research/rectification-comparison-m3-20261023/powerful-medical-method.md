# Dotter, Gridder and Powerful Medical: methodological and experimental audit

Checked 23 October 2026. Scope is engineering comparison, without a legal opinion. No image was sent to Powerful Medical, no inference API was called and nobody was contacted.

## Identification and primary evidence

The relevant publication is Demolder et al., **High precision ECG digitization using artificial intelligence**, Journal of Electrocardiology 90 (2025), 153900, DOI [10.1016/j.jelectrocard.2025.153900](https://doi.org/10.1016/j.jelectrocard.2025.153900). The author preprint is [10.1101/2024.08.31.24312876](https://www.medrxiv.org/content/10.1101/2024.08.31.24312876v1.full).

Dotter detects grid intersections. Gridder infers their lattice relationship and missing points. Undistortion resamples individual cells. Leader belongs to subsequent trace extraction. These names do not denote four independent end-to-end competitors. The preprint describes U-Net with residual blocks, 256-pixel training patches, 300 epochs, Adam, logits BCE and SiLU. Its four-point cell transformation is described as linear. Public text does not uniquely specify every interpolation convention of the running product. [Paper PDF indexed methods](https://www.medrxiv.org/content/10.1101/2024.08.31.24312876v1.full.pdf).

The geometry family includes [US20240144417A1](https://patents.google.com/patent/US20240144417A1/en), [EP4302256B1](https://patents.google.com/patent/EP4302256B1/en) and the now-published [US12614244B2](https://patents.google.com/patent/US12614244B2/en). Patent description and claims were read, and application drawings 1A/B, 1G and 2J/K were rendered and inspected. A patent is evidence of a disclosed mechanism, not its implementation checksum or benchmark score.

## What the descriptions establish

The application gives an example threshold of 150, largest 2,520 spots, intensity-weighted coordinates, a 45-degree rotation, 20-nearest-neighbor edges, length/angle clustering, Gaussian edge scores, four-direction maximum-likelihood cycles, largest connected component and regression of missing vertices. Its expected grid is 60 by 42 points. Those are disclosed example settings, not evidence that every product image must have that grid size. The claims describe inverse sampling of source color into matched target cells. The template can use 64-pixel squares. [US application, Gridder description](https://patents.google.com/patent/US20240144417A1/en).

The granted European independent method claim associates original and target cells, determines each cell relationship and samples source data into the target. Its dependent claims describe graph construction, probabilistic edge selection, closed cycles, breadth-first component selection and regression for absent vertices. [European claims](https://patents.google.com/patent/EP4302256B1/en).

The US grant also specifies cell correspondence and source-to-target sampling. It does not provide a public executable Dotter checkpoint, calibrated geometric error bounds or the exact current server build. [US grant claims](https://patents.google.com/patent/US12614244B2/en).

## Methodological decisions

The table separates observations from our engineering inferences. Assertions about commercial internals are restricted to the cited descriptions or schema. No unavailable implementation has been assigned a simulated performance number. The full author preprint was read through its [author-uploaded copy](https://www.researchgate.net/publication/383740641_High_Precision_ECG_Digitization_Using_Artificial_Intelligence) after direct publisher access returned 403. Dotter used 176 annotated images, split 140/36. Leader used 232 images, split 185/47. These image counts are not independent patient counts.

| Decision | Disclosed evidence | Critique and proposed decision for our method |
|---|---|---|
| Detect major-grid intersections | Dotter mask and major-square reference | Keep independent point detection. Evaluate 1 mm/5 mm aliases explicitly, including disappearance of major lines and confusing dense minor grids. A high heatmap value does not settle physical pitch. |
| Decode localization | Example binary spots or intensity-weighted coordinates | Compare source-coordinate offset regression, centroids and native resolution under the same topology. Patch training does not establish how the product tiles or resizes at inference. Do not invent that detail. |
| Assign IDs | Gridder uses directional cycles and connected grids | The decisive issue is correspondence, not merely denoising coordinates. Test cycles, reciprocal edges, row/column aliases and disconnected component reconciliation. A wrong integer ID can survive every smoothness check. |
| Select component | Largest connected network in the description | Largest need not be physically correct when a false dense grid or screen pattern dominates. Our largest-component assumptions need adversarial competing lattices and margins without grid. A small correct region should not be counted as whole-image success. |
| Infer absent nodes | Missing-node regression in Gridder | Filling a hole creates a hypothesis. Preserve a provenance label distinguishing detected, interpolated and extrapolated nodes. Evaluate a fold hidden inside a missing region. Never turn an inferred grid into an image-erasing mask. |
| Choose local mapping | Four-point per-cell transform | Keep planar-projective exactness, local continuity and fold preservation as separate requirements. Independent homographies can agree at corners while disagreeing inside a shared edge. Two affine triangles guarantee a common straight edge but introduce projective bias. See the actual component experiment below. |
| Complete outside grid | Public product behavior outside the inferred template is not established here | Our full-frame projective completion is an additional assumption. Report performance separately inside observed cells, inferred holes and extrapolated margins. Do not claim better preservation than the commercial product without its actual output. |
| Calibrate mm and axes | A known major square defines physical scale | Explicitly separate physical pitch, paper speed, voltage gain and image orientation. A regular grid alone cannot identify all of them. Give metadata to every method in a metadata-assisted track or to none in the automatic track. |
| Handle confidence | Gaussian edge probabilities and product warnings | They are not publicly documented as calibrated millimetre uncertainty. Our ranked scores also are not calibrated uncertainty. The API metadata probability concerns detected format, not coordinate error in mm. Rank methods by empirical error versus returned coverage, including catastrophic tails. |
| Resample pixels | Original source-color inverse sampling is disclosed | Retain original RGB and explicit sampling validity. Add signal-bandwidth and anisotropic-Jacobian checks before reducing pixels/mm. More output pixels cannot recover information lost by an earlier resize. |
| Report failure | API has explicit stage failure and warning codes | Preserve every failure and warning. A returned waveform with skipped normalization is not evidence that image rectification succeeded. Our original-image fallback also must not count as successful rectification. |
| Evaluate | Published overall digitization quality | Compare geometry in mm separately from full signal in mV and ms. No comparison between our geometric percentages and their waveform PCC can rank the rectifiers. The paper aligns signals within 0.5 second and corrects vertical median offset. Report raw calibration and aligned waveform metrics separately. It also explicitly notes unsupported pacing spikes. |

These critiques apply symmetrically. They do not establish that our implementation handles the hard cases better.

## Authentic implementation and API access

An actual official developer showcase exists at [aidemo.pmcardio.com](https://aidemo.pmcardio.com/). It links to [public Swagger documentation](https://eu-west-1.pm-api.com/v0/docs). The page names the OpenAPI document [v0/openapi.json](https://eu-west-1.pm-api.com/v0/openapi.json), which was downloaded by an unauthenticated GET. This is documented access, not endpoint guessing.

The snapshot exposes digitization v3 and v4, requiring `PM-API-Key`. Its description requires a service agreement and appropriate feature quota. No usable key was provided in this task. There is no Dotter mask, Gridder vertex or dense warp export endpoint among the twelve documented paths. V3 returns mV signals, Hz and ms offsets. V4 also identifies the production build and exposes plotting positions in mm. This makes authentic end-to-end signal benchmarking feasible with authorized access, but does not currently enable dense geometry benchmarking.

The schema has concrete warnings such as `dummy_normalization`, `skipped_normalization` and `low_coverage`. Failures include `dotter_small_image` and `gridder_too_many_points`. Therefore successful HTTP status alone is not the outcome. Pin `production_identifier`, `powerfulmedical_system_id`, schema hash and all warnings. Explicitly set paper speed and gain, especially because v4 documents a default gain of 20 mm/mV. Our typical 10 mm/mV inputs must not inherit that default.

No official public source release or model checkpoint for Dotter/Gridder was located in the paper, patent or product links reviewed. That is the outcome of this bounded search, not proof that no such implementation could ever be obtained. The available authentic comparison is currently **blocked by absent API credentials and by absent geometry-stage export**, rather than a nonexistent API.

Important independent evidence: the [CinC 2024 organizer paper](https://cinc.org/archives/2024/pdf/CinC2024-011.pdf) says Powerful Medical granted API access for independent evaluation, and that PMcardio exceeded Challenge algorithms with performance varying across hidden image variants. That is evidence of full digitization capability. It is neither a dense-coordinate result nor a comparison to the separate Kaggle competition or our current M2 method.

## Public external data

[PM-ECG-ID](https://doi.org/10.5281/zenodo.13617672) is a dataset, not the source code or weights. The concept DOI currently resolves to version [13617673](https://zenodo.org/records/13617673). API metadata was successfully read, even though browser retrieval of the landing page failed. It describes 6,000 entries from 100 PTB-XL waveforms, mixing actual photographs/scans and augmentations. The ZIP is 33,863,935,685 bytes, MD5 `e9e21300e7eefa139eacf024d2ebda26`, with GPL-3.0-or-later metadata. It was not downloaded in full.

Metadata links each image to its PTB-XL ECG ID and layout. Before using it for a new external test, check all 100 underlying ECG and patient identities against our training and development groups. Hold out whole source records, not augmented photographs. The published data description does not promise dense physical grid coordinates, so waveform comparison is directly supported whereas millimetre warp scoring would need additional reference annotation.

## Executed component comparison

`component_control.py` is our independent geometry experiment. It is **not commercial Dotter, Gridder, PMcardio or a validated reproduction of their four-point operator**. Its purpose is to distinguish reasonable interpolation decisions using the same exact or noisy four-corner observations.

Ten fixed scenes, each with 12,288 independent interior material queries, compare two affine triangles, a four-point projective homography and bilinear interpolation. Ground truth uses isometric plane, cylindrical and hinged embeddings projected by an ideal finite pinhole camera. Cases include a grid-aligned fold, a fold inside a cell, an oblique fold, three 0.35-pixel noise seeds and a corrupt observed node. A separate boundary query set and shared-edge seam set are retained. The protocol was written before results. Nothing was tuned from the scores.

| Scene | Two triangles RMSE mm | Homography RMSE mm | Bilinear RMSE mm | Returned coverage |
|---|---:|---:|---:|---:|
| Exact plane | 0.20759 | <1e-10 | 0.14261 | 100% |
| Grid-aligned hinge | 0.26065 | <1e-10 | 0.21055 | 100% |
| Axis-aligned cylinder | 0.55451 | 0.38170 | 0.49568 | 95.90% |
| Oblique cylinder | 0.81679 | 0.38026 | 0.57971 | 91.35% |
| Hinge inside a cell | 0.73831 | 0.61638 | 0.68561 | 94.43% |
| Oblique hinge | 0.82981 | 0.39034 | 0.57646 | 94.59% |

RMSE is conditional on returned queries. G, W, C and R retain the full fixed query denominator in the results JSON. Curved physical boundaries need not follow straight edges between observed corners, so some true surface queries fall outside all interpolated cell polygons. That coverage loss is retained. The corrupt-node case rejects three nonconvex cells and still exposes poor accepted coordinates from the remaining cell. Invalid cells are not silently repaired.

There is a critical negative result. Independent cell homographies disagree along shared edges by up to **7.4776 source pixels** for the oblique cylinder and **12.7621 pixels** for the oblique hinge. With noisy planar corners, the maxima range from **0.1745 to 0.5977 pixels**. Both affine triangles and bilinear interpolation remain C0 along the shared edges in these constructions. Thus planar exactness alone is an insufficient acceptance criterion.

Six independent checks pass. They verify material metric preservation away from the hinge, exact planar and aligned-hinge projectivity, seam behavior, affine image-coordinate change invariance including half-pixel translation, and complete accounting of fixed queries and invalid cells. They do not validate a commercial implementation or a clinical image pipeline.

## Concrete next decisions

1. Keep original RGB, provenance and explicit partial/fallback states. Their presence is valuable even when coordinate accuracy fails.
2. Attack ID and pitch ambiguity before more aggressive smoothing. The present primitive study assumes those variables are correct and cannot solve that stage.
3. Add planar-projective exactness and shared-edge continuity as simultaneous mathematical gates. Do not replace raw triangles with unrestricted independent homographies solely because their interior RMSE is lower.
4. A candidate worth a separately frozen experiment is a shared global projective coordinate frame followed by a continuous residual mesh in that frame. It can preserve C0 edges and exact planar projectivity in principle. This study has not tested that candidate.
5. Keep a dense geometry track with fixed queries and a complete waveform track with lead ownership, calibration and all failures. Authentic PMcardio currently fits only the second track through the documented public response.
6. Report unavailable commercial access as unavailable. Publish the primitive ablation under its own name. No simulated result belongs in the PMcardio row.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python component_control.py --out component-results
python -m unittest discover -s . -p test_component_control.py -v
```

`availability.json` records authenticated-access requirements, metadata and schema checksums. `sources.json` maps official URLs to current retrieval references. `component-results` contains the protocol, all thirty method/scene results and all ten arrays, including invalid and failed support. This is sufficient to independently re-score the component study.
