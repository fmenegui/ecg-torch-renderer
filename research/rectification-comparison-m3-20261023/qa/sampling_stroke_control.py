"""Independent analytical fine-stroke control of the frozen native RGB sampler.

No detector, bank image, learned map, grid masking, or source modification.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'ecg-torch-renderer'))
from ecg_gpu.digitization.complete_rectifier import CompleteMap, render_complete


def physical_crop(rgb, origin, density, size=(360, 240)):
    """Nearest display of actual sample centres in the same physical viewport."""
    x = .5+(np.arange(size[0])+.5)*3/size[0]
    y = 9+(np.arange(size[1])+.5)*2/size[1]
    ix = np.floor((x-origin[0])*density+.5).astype(int)
    iy = np.floor((y-origin[1])*density+.5).astype(int)
    return Image.fromarray(rgb[iy[:, None], ix[None, :]])


def contact_sheet(examples, destination):
    canvas = Image.new('RGB', (1220, 1120), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 18)
    title = ImageFont.truetype(font_path, 26)
    small = ImageFont.truetype(font_path, 15)
    draw.text((35, 20), 'Thin strokes can disappear with exact geometry', font=title, fill='#172936')
    draw.text((35, 59), 'Frozen bilinear point sampler, no antialias prefilter and no RGB confidence mask', font=font, fill='#374b58')
    column_titles = ['Original RGB, 10 source px/mm', 'Actual sampler, 4 output px/mm', 'Actual sampler, 8 output px/mm']
    for i, label in enumerate(column_titles):
        draw.text((35+395*i, 100), label, font=font, fill='#172936')
    selected = [(1, 10, '0.1 mm stroke, source column 10, aligned phase'),
                (1, 11, '0.1 mm stroke, source column 11, adverse phase'),
                (10, 11, '1 mm stroke, source columns 11–20, thick control')]
    for row, (width, x, label) in enumerate(selected):
        top = 140+row*300
        draw.text((35, top), label, font=font, fill='#172936')
        for column, density in enumerate((10., 4., 8.)):
            image, origin, peak = examples[width, x, density]
            cropped = physical_crop(image, origin, density)
            left = 35+395*column
            canvas.paste(cropped, (left, top+30))
            draw.rectangle((left-1, top+29, left+360, top+270), outline='#bac5cd')
            suffix = '  ABSENT' if peak == 0 else ''
            draw.text((left, top+275), 'Peak darkness: '+str(peak)+'/255'+suffix,
                      font=small, fill='#923f2b' if peak == 0 else '#374b58')
    draw.text((35, 1050), 'Every panel shows x = 0.5…3.5 mm, y = 9…11 mm using nearest display of actual samples.', font=small, fill='#374b58')
    draw.text((35, 1076), 'Analytical control only. 8 px/mm is not a universal guarantee. Frozen method remains unchanged.', font=small, fill='#374b58')
    canvas.save(destination)


def run():
    source = np.array([[0., 0.], [200., 0.], [200., 200.], [0., 200.]])
    observed = np.array([[50., 50.], [150., 50.], [150., 150.], [50., 150.]])
    ids = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    mapping = CompleteMap(source, source/10, np.empty((0, 2), int),
                          np.array([[0, 1, 2], [0, 2, 3]]),
                          {'method': 'Exact analytical affine map,10source pixels per mm'},
                          observed, observed/10, ids, [[0, 0]])
    records, examples, image_receipts = [], {}, []
    image_dir = Path(__file__).parent/'sampling-stroke-images'
    image_dir.mkdir(exist_ok=True)
    checks = 0
    for width in (1, 10):
        for source_column in range(10, 15):
            rgb = np.full((201, 201, 3), 255, np.uint8)
            rgb[:, source_column:source_column+width] = 0
            point = np.array([[float(source_column), 100.]])
            assert np.isfinite(mapping.forward(point)).all()
            assert not mapping.eligible_geometry(point).any()
            checks += 2
            examples[width, source_column, 10.] = (rgb.copy(), np.zeros(2), 255)
            save_example = (width, source_column) in {(1, 10), (1, 11), (10, 11)}
            if save_example:
                p = image_dir/f'original-width{width}-column{source_column}.png'
                Image.fromarray(rgb).save(p)
                image_receipts.append({'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
            for density in (4., 8.):
                output, valid, evidence, metadata = render_complete(rgb, mapping, pixels_per_mm=density)
                darkness = 255-output[..., 0].astype(int)
                max_darkness = int(darkness[valid].max())
                examples[width, source_column, density] = (output.copy(), np.asarray(metadata['output_origin_mm']), max_darkness)
                if save_example:
                    p = image_dir/f'bilinear-{density:g}px-width{width}-column{source_column}.png'
                    Image.fromarray(output).save(p)
                    image_receipts.append({'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
                records.append({'source_column': source_column, 'source_stroke_width_pixels': width,
                                'source_stroke_width_mm': width/10, 'source_density_pixels_per_mm': 10,
                                'output_density_pixels_per_mm': density,
                                'source_region_trusted': False, 'RGB_resampled': metadata['resampled'],
                                'output_origin_mm': metadata['output_origin_mm'],
                                'output_width': output.shape[1], 'output_height': output.shape[0],
                                'peak_darkness_0_to_255': max_darkness,
                                'nonwhite_output_pixels': int(np.count_nonzero(darkness[valid])),
                                'stroke_completely_absent': max_darkness == 0})
                assert metadata['resampled']
                checks += 1
    at4 = [r for r in records if r['output_density_pixels_per_mm'] == 4 and r['source_stroke_width_pixels'] == 1]
    at8 = [r for r in records if r['output_density_pixels_per_mm'] == 8 and r['source_stroke_width_pixels'] == 1]
    thick = [r for r in records if r['source_stroke_width_pixels'] == 10]
    assert any(r['stroke_completely_absent'] for r in at4)
    assert all(not r['stroke_completely_absent'] for r in at8)
    assert any(not r['stroke_completely_absent'] for r in at4)
    assert all(r['peak_darkness_0_to_255'] == 255 for r in thick)
    checks += 4
    sheet = Path(__file__).parent/'sampling-stroke-contact-sheet.png'
    contact_sheet(examples, sheet)
    source_path = ROOT/'ecg-torch-renderer/ecg_gpu/digitization/complete_rectifier.py'
    report = {'status': 'pass_control_reproduced_sampling_failure', 'checks': checks,
              'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
              'cases': records,
              'method': 'Frozen render_complete using cv2.remap INTER_LINEAR, no prefilter or supersampling. Exact affine source-pixel-centre to material map [x/10,y/10]. White 255, black 0.',
              'thin_stroke_source_columns': [10, 11, 12, 13, 14],
              'thin_stroke_fully_lost_at4_columns': [r['source_column'] for r in at4 if r['stroke_completely_absent']],
              'thick_control': '1 mm = 10 source pixels. Every phase at 4 and 8 px/mm retains full black contrast 255.',
              'contact_sheet': {'file': sheet.name, 'sha256': hashlib.sha256(sheet.read_bytes()).hexdigest(),
                                'display': 'Same physical viewport x0.5..3.5mm,y9..11mm, nearest display only, original and actual sampled PNG values.'},
              'original_and_output_PNGs': image_receipts,
              'finding': 'With exact geometry and no confidence-mask erasure, frozen bilinear point sampling at 4 px/mm completely misses some 0.1 mm strokes depending on phase. The 8 px/mm control retains some contrast at every tested phase.',
              'scope': 'Analytical stress control only, not a measurement of prevalence or clinical signal error. Does not establish 8 px/mm as universally sufficient or an optimal production choice.',
              'decision': 'Do not equate finite full-frame geometry or exact original archiving with resampled stroke fidelity. Investigate density, local area/antialias sampling and known-stroke retention before a preservation claim. No frozen method changed.'}
    (Path(__file__).parent/'sampling-stroke-review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    run()
