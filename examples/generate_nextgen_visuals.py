"""Post-hoc illustrations of frozen evaluation cases, using cached detections.

No neural inference or method selection. Frozen scores rank illustrations.
Gold is used for scoring and parity, never geometry fitting. Use a fresh
--destination directory to preserve previous outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.cell_mesh import CellMap, cell_triangles
from ecg_gpu.digitization.complete_rectifier import CompleteMap
from examples.rectify_denoising_grid import write_geometry_output
from examples.evaluate_rectification_diagnostic import aligned_mm

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def change(case, metric):
    return (case['methods']['soft-1.0']['domains']['grid'][metric]
            -case['methods']['B0']['domains']['grid'][metric])


def metric_deltas(case):
    return {
        domain: {
            metric: (case['methods']['soft-1.0']['domains'][domain][metric]
                     - case['methods']['B0']['domains'][domain][metric])
            for metric in ('g', 'w', 'c')
        }
        for domain in ('grid', 'paper')
    }


def reconstruct(folder, report):
    with np.load(folder/'observed-grid-geometry.npz') as archive:
        observed = archive['xy'].astype(float)
        canonical = archive['canonical_mm'].copy()
        identities = archive['ids'].copy()
        cells = archive['cells'].copy()
    mesh = folder/'full-frame-mesh.npz'
    if mesh.exists():
        with np.load(mesh) as archive:
            mapping = CompleteMap(archive['source'].copy(), archive['target_mm'].copy(),
                np.empty((0,2),int), archive['triangles'].copy(), report['complete_geometry'],
                observed, canonical, identities, cells)
        return mapping, None
    triangles = cell_triangles(observed, canonical, identities, cells)
    return None, CellMap(observed, canonical, triangles)


def parity(folder, report, case, archived_queries, method):
    mapping, partial = reconstruct(folder, report)
    results = {}
    for domain in ('grid','paper'):
        query = archived_queries[domain+'_query']
        expected = archived_queries[method+'_'+domain+'_predicted']
        expected_eligible = archived_queries[method+'_'+domain+'_eligible']
        raw = partial(query) if mapping is None else mapping.forward(query)
        eligible = np.isfinite(raw).all(1) if mapping is None else mapping.eligible_geometry(query, mapped=raw)
        predicted = aligned_mm(raw,case['evaluation_gauge']) if case['evaluation_gauge']['matched'] else np.full_like(raw,np.nan)
        finite = np.isfinite(predicted).all(1)
        same_finite = np.array_equal(finite,np.isfinite(expected).all(1))
        maximum = float(np.max(np.abs(predicted[finite]-expected[finite]))) if finite.any() and same_finite else None
        same_eligible = np.array_equal(eligible,expected_eligible)
        passed = same_finite and same_eligible and (maximum is None or maximum < 1e-8)
        results[domain] = {'finite_mask_equal':same_finite,'eligibility_mask_equal':same_eligible,
            'maximum_absolute_coordinate_difference_mm':maximum,'passed_at_1e_minus_8_mm':passed}
        if not passed:
            raise ValueError('CLI/evaluator parity failed: '+case['key']+' '+method+' '+domain+' '+str(results[domain]))
    return results


def font(size):
    candidates = [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                  Path('/usr/local/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path),size=size)
    return ImageFont.load_default(size=size)


def mosaic(out, case, outputs, selection_reason):
    panel_w, picture_h, gap, top = 760, 530, 18, 108
    width = panel_w*3+gap*4
    canvas = Image.new('RGB',(width,top+picture_h+155),(239,244,248))
    draw = ImageDraw.Draw(canvas)
    split_label = 'VALIDAÇÃO' if case['split']=='validation' else 'TESTE NOVO'
    draw.text((gap,12),f"{split_label} | {case['key']} | {case['backend']} | {case['layout']}",font=font(25),fill=(21,39,58))
    draw.text((gap,47),'Ilustração escolhida depois dos resultados. As métricas usam os arrays nativos.',font=font(18),fill=(57,75,95))
    original = np.load(ROOT/'build/nextgen-bank'/outputs['source_file'])['image']
    panels = [('Original',Image.fromarray(original),'RGB integral, sem alteração',None)]
    for method,label in [('B0','B0'),('soft-1.0','Soft 1,0')]:
        value=outputs['methods'][method]
        image=Image.open(out/value['output_image']).convert('RGB')
        status='proposta retificada' if value['rgb_resampled'] else 'RGB original preservado'
        panels.append((label,image,status,method))
    for index,(title,image,status,method) in enumerate(panels):
        x=gap+index*(panel_w+gap)
        draw.rounded_rectangle((x,top,x+panel_w,top+picture_h+113),radius=9,fill='white')
        picture=ImageOps.contain(image,(panel_w-20,picture_h-16),Image.Resampling.LANCZOS)
        canvas.paste(picture,(x+(panel_w-picture.width)//2,top+8+(picture_h-16-picture.height)//2))
        y=top+picture_h+7
        draw.text((x+14,y),title+' | '+status,font=font(20),fill=(19,65,99))
        if method:
            value=outputs['methods'][method]
            metrics=case['methods'][method]['domains']['grid']
            line=f"G {metrics['g']*100:.2f}%    W {metrics['w']*100:.2f}%    C {metrics['c']*100:.2f}%".replace('.',',')
            draw.text((x+14,y+31),line,font=font(19),fill=(27,43,59))
            detail='Malha completa aceita' if value['accepted_full_map'] else 'Somente coordenadas parciais'
            if value['accepted_full_map'] and not value['rgb_resampled']:
                detail+=' | canvas não reamostrado'
            draw.text((x+14,y+60),detail,font=font(17),fill=(85,96,106))
        else:
            draw.text((x+14,y+31),'Captura sintética, gerada de sinal público PTB-XL.',font=font(17),fill=(85,96,106))
            draw.text((x+14,y+60),'Prévias redimensionadas somente para apresentação.',font=font(17),fill=(85,96,106))
    deltas = metric_deltas(case)
    footer = ('Variação Soft menos B0 (pp): '
              f"grade G {100*deltas['grid']['g']:+.4f}, W {100*deltas['grid']['w']:+.4f} | "
              f"papel G {100*deltas['paper']['g']:+.4f}, W {100*deltas['paper']['w']:+.4f}").replace('.', ',')
    draw.text((gap, top+picture_h+127), footer, font=font(18), fill=(57,75,95))
    canvas.save(out/'comparison.jpg',quality=93,optimize=True)
    return {'file':'comparison.jpg','size':list(canvas.size),'sha256':digest(out/'comparison.jpg'),
            'preview_resize_only':True,'no_content_crop':True,'selection_reason':selection_reason}


def publish_root_artifacts(out, case_folder, record):
    """Expose the report images and a metadata sidecar in one directory."""
    prefix = record['split']+'-'+record['key']+'-comparison'
    image_path = out/(prefix+'.jpg')
    shutil.copyfile(case_folder/'comparison.jpg', image_path)
    record['report_artifacts'] = {
        'mosaic': image_path.name,
        'metadata': prefix+'.json',
        'case_directory': str(case_folder.relative_to(out)),
        'mosaic_sha256': digest(image_path),
    }
    dump(case_folder/'metadata.json', record)
    dump(out/(prefix+'.json'), record)


def main(density=4., destination=None):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    out=ROOT/'build/nextgen-visuals' if destination is None else Path(destination)
    out.mkdir(parents=True,exist_ok=True)
    evaluation=ROOT/'build/nextgen-evaluation'
    summaries=ROOT/'build/nextgen-summary/summary.json'
    reports={s:json.loads((evaluation/(s+'.json')).read_text()) for s in ('validation','test')}
    summary=json.loads(summaries.read_text())
    assert len(reports['validation']['cases'])==48 and len(reports['test']['cases'])==64
    assert all('summary' in r for r in reports.values()) and 'test' in summary['splits']
    assert summary['selected_method']=='soft-1.0'
    manifest={r['key']:r for r in json.loads((ROOT/'build/nextgen-bank/manifest.json').read_text())}
    validation=reports['validation']['cases']
    assessment=reports['test']['cases']
    choices=[('validation',max(validation,key=lambda c:(change(c,'g'),c['key'])),'maior ΔG no domínio grid'),
             ('validation',next(c for c in validation if c['key']=='01025-6'),'regressão de W explicitamente solicitada')]
    choices.append(('test',max(assessment,key=lambda c:(change(c,'g'),c['key'])),'maior ΔG no domínio grid'))
    if max(change(c,'w') for c in assessment)>0:
        worst=max(assessment,key=lambda c:(change(c,'w'),-change(c,'g'),c['key']))
        reason='maior aumento de W no domínio grid'
    else:
        worst=min(assessment,key=lambda c:(change(c,'g'),c['key']))
        reason='menor ΔG, pois nenhum caso test tem aumento de W. Empate por key crescente'
    choices.append(('test',worst,reason))
    selection={'purpose':'post-hoc qualitative illustration only, not method or threshold selection',
               'read_after_complete_test_and_summary':True,'selected_strength_unchanged':1.0,
               'domain_for_case_ranking':'grid','density_px_per_mm':density,
               'new_neural_inference':False,'new_training':False,
               'gold_use':'scoring, ranking illustrations and parity gauge only, never geometry fitting',
               'captures_are_synthetic':True,'external_clinical_photographs':False,
               'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (
                   ROOT/'examples/rectify_denoising_grid.py',ROOT/'ecg_gpu/digitization/robust_refinement.py',
                   ROOT/'ecg_gpu/digitization/complete_rectifier.py')},
               'input_report_sha256':{s:digest(evaluation/(s+'.json')) for s in reports},
               'summary_sha256':digest(summaries),'renderer_sha256':digest(Path(__file__)),
               'presentation_density_note':'4 px/mm supplement follows visual undersampling observed at 2 px/mm, without changing maps or statistical selection' if density==4 else 'initial 2 px/mm preview',
               'cases':[]}
    for split,case,reason in choices:
        selection['cases'].append({'key':case['key'],'split':split,'reason':reason,
                                   'delta_G':change(case,'g'),'delta_W':change(case,'w')})
    dump(out/'selection.json',selection)
    results=[]
    for split,case,reason in choices:
        key=case['key']
        case_folder=out/split/key
        case_folder.mkdir(parents=True,exist_ok=True)
        source=ROOT/'build/nextgen-bank'/manifest[key]['file']
        with np.load(source) as data:
            rgb=data['image'].copy()
        rgb_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
        archive=evaluation/'query-maps'/(key+'.npz')
        with np.load(archive) as data:
            cached={k:data[k].copy() for k in data.files}
        local={'selected_indices':cached['selected'].tolist(),
               'ids':cached['lattice_ids'].tolist(),'cells':cached['cells'].tolist(),
               'partial_status':case['topology']['partial_status']}
        record={'key':key,'split':split,'backend':case['backend'],'layout':case['layout'],
                'selection_reason':reason,'source_file':manifest[key]['file'],
                'post_hoc_illustration':True,'selection_domain':'grid',
                'metric_deltas_soft_minus_B0':metric_deltas(case),
                'source_array_sha256':rgb_hash,'source_archive_sha256':digest(source),
                'cached_query_map_sha256':digest(archive),'density_px_per_mm':density,'methods':{}}
        for strength,method in [(0.,'B0'),(1.,'soft-1.0')]:
            folder=case_folder/method
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError('Avoid stale mixed output in '+str(folder))
            cv2.setRNGSeed(1973)
            report=write_geometry_output(rgb,cached['detected_xy'],cached['detected_scores'],local,
                folder,pixels_per_mm=density,refinement_strength=strength)
            dump(folder/'report.json',report)
            original=np.asarray(Image.open(folder/'original-rgb.png'))
            assert np.array_equal(original,rgb)
            same=parity(folder,report,case,cached,method)
            value={'report':report,'accepted_full_map':(folder/'full-frame-mesh.npz').exists(),
                   'rgb_resampled':report['complete_geometry']['resampled'],
                   'output_status':report['status'],'output_image':str((folder/report['output_image']).relative_to(case_folder)),
                   'original_array_exact':True,'original_array_sha256':hashlib.sha256(original.tobytes()).hexdigest(),
                   'geometry_evaluator_parity':same,
                   'evaluation_grid_metrics':case['methods'][method]['domains']['grid'],
                   'evaluation_paper_metrics':case['methods'][method]['domains']['paper']}
            with Image.open(folder/report['output_image']) as image:
                value['output_size']=list(image.size)
            record['methods'][method]=value
            print(json.dumps({'key':key,'method':method,'resampled':value['rgb_resampled'],
                'full_map':value['accepted_full_map'],'output_size':value['output_size'],
                'parity':all(r['passed_at_1e_minus_8_mm'] for r in same.values())}),flush=True)
        record['mosaic']=mosaic(case_folder,case,record,reason)
        publish_root_artifacts(out, case_folder, record)
        results.append(record)
        dump(out/'metadata.json',{'selection':selection,'cases':results})
    print(json.dumps({'complete':True,'cases':len(results)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density',type=float,choices=(2.,4.),default=4.)
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    main(args.density,args.destination)
