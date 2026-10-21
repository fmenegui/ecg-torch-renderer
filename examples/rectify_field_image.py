"""Experimental RGB-only rectification with a consolidated-study checkpoint.

Outputs preserve missing geometric support. The model assumes the detected
points belong to a 5 mm major grid. Gain, speed, layout and global injectivity
are not inferred or certified by this command.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.field_rectifier import GridFieldNet, chart_lattice
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_point_learning import decode_grid_points
from examples.digitization_grid_rectification import grid_remap, save_rgb


def rectify(checkpoint, image_path, out, *, device='cpu', pixels_per_mm=8.):
    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError('Positive finite display density required')
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['variant'] not in ('raw-point','contrast-point','contrast-global'):
        raise ValueError('Unknown checkpoint variant')
    if state['variant']=='contrast-global':
        raise ValueError('Chart association failed held-out identity checks. Use a point-only checkpoint')
    model=GridFieldNet(contrast=state['variant']!='raw-point').to(device).eval()
    model.load_state_dict(state['state_dict'])
    rgb=np.asarray(Image.open(image_path).convert('RGB'))
    height,width=rgb.shape[:2]
    ratio=min(768/width,512/height)
    w,h=max(1,round(width*ratio)),max(1,round(height*ratio))
    canvas=np.full((512,768,3),255,np.uint8)
    canvas[:h,:w]=np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR))
    tensor=torch.from_numpy(canvas.copy()).permute(2,0,1)[None].float().to(device)/255
    with torch.inference_mode():
        value=model(tensor)
        prediction=decode_grid_points(value['heat'].sigmoid(),value['offsets'],threshold=float(state['threshold']),max_points=512*768)
        keep=prediction['valid'][0]
        xy=prediction['xy'][0,keep]
        scores=prediction['scores'][0,keep]
        chart=F.grid_sample(value['chart'],((xy+.5)/xy.new_tensor([768,512])*2-1)[None,None],align_corners=False)[0,:,0].T
    xy,scores,chart=xy.cpu().numpy(),scores.cpu().numpy(),chart.cpu().numpy()
    inside=((xy>=0)&(xy<=[w-1,h-1])).all(-1)
    xy,scores,chart=xy[inside],scores[inside],chart[inside]
    xy=(xy+.5)/[w/width,h/height]-.5
    topology=chart_lattice(xy,scores,chart) if state['variant']=='contrast-global' else organize_lattice(xy,scores)
    report={'checkpoint_sha256':sha256_file(checkpoint),'input_sha256':sha256_file(image_path),
        'variant':state['variant'],'threshold':state['threshold'],'device':str(device),
        'assumed_major_grid_mm':5.,'display_pixels_per_mm':pixels_per_mm,'topology':topology,
        'gain_inferred':False,'paper_speed_inferred':False,'status':'rejected'}
    np.savez_compressed(out/'points.npz',xy=xy,scores=scores,chart=chart)
    if topology['status']=='accepted' and len(topology['cells'])>=16:
        ids=np.asarray(topology['ids'])
        canonical=ids[:,::-1]*(5*pixels_per_mm)
        size=np.maximum(512,(canonical.max(0)+1).astype(int))[::-1]
        if max(size)>4096:
            report['error']='Inferred extent exceeds the 4096px diagnostic bound'
        else:
            warped,support=grid_remap(rgb,xy[topology['selected_indices']],canonical,ids,
                tuple(size),allowed_cells=topology['cells'])
            report.update(status='rectified',output_sha256=save_rgb(out/'rectified.png',warped),
                          supported_pixel_fraction=float(support.mean()))
            np.savez_compressed(out/'support.npz',valid=support)
    _write_json(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','image','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--pixels-per-mm',type=float,default=8.)
    a=p.parse_args()
    torch.set_num_threads(4)
    result=rectify(a.checkpoint,a.image,a.out,device=a.device,pixels_per_mm=a.pixels_per_mm)
    print(json.dumps({key:result[key] for key in ('status','variant','threshold')}))
