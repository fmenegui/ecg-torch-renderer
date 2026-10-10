"""Export controlled scene variations and every standard layout for inspection.

PNG encoding, JSON and zip writing happen after rendering, outside training.
The bundled waveform is entirely artificial. No patient record is downloaded.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from PIL import Image, ImageDraw

from .demo import demo_signals
from .gpu_renderer import ECGRenderer, RenderConfig


def png(value):
    array=(value.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    return Image.fromarray(array)


def plain(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,dict):
        return {key:plain(item) for key,item in value.items()}
    return value


def export_gallery(out,device,height=768,width=1024,image_format='jpeg'):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    values,fs,_=demo_signals(fs=500,duration=10)
    signals=torch.as_tensor(values,dtype=torch.float32,device=device)[None]
    renderer=ECGRenderer(RenderConfig(height=height,width=width,profile='photo',severity=.8,
                                      antialias_samples=4,surface_resolution=128,shadow_samples=12)).to(device)
    fixed={'layout':'3x4+r1','speed_index':1,'gain_index':1,'sequential':True,
           'quarter':0,'angle':.04,'zoom':.83,'translate_x':-.02,'translate_y':.01,
           'perspective_x':.12,'perspective_y':-.04,'radial':.006,'curl_x':0.,'curl_y':0.,
           'crease_warp':0.,'crease_strength':0.,'surface_mode':'flat','surface_strength':.85,
           'relief_view':[.52,-.30],'paper_palette':0,'paper_tint':0.,'paper_age':0.,
           'paper_noise':.005,'paper_color':[1.,1.,1.],'palette':0,'grid_strength':.30,
           'minor_strength':.27,'grid_width':.28,'grid_present':1.,'grid_style':0,'line_width':1.25,
           'pulse_present':1.,'coffee_strength':0.,'bleed_strength':0.,
           'ink_color':[.065,.060,.058],'ink_density':.96,'thermal_fade':.04,'stain_strength':0.,
           'band_strength':0.,'font_size':width*.012,'header_present':1.,'label_present':1.,
           'lead_present':1.,'label_below':False,'exposure':.98,'light_x':.035,'light_y':-.025,
           'shadow_strength':0.,'vignette':.03,'ambient_light':.61,
           'light_direction':[[.7,-.8,1.3],[-.8,.4,1.7],[.2,.5,1.5]],
           'light_energy':[.38,.13,.07],'light_color':[[1.,.97,.94],[.96,.98,1.],[1.,1.,1.]],
           'flash_strength':.018,'specular_strength':.025,'ambient_occlusion':.22,
           'white_balance':[1.,.985,.975],'blur':.45,'motion':.45,'motion_angle':.35,
           'read_noise':.0015,'shot_noise':.0015,'gamma':1.,'jpeg_quality':88.,'jpeg_active':1.,
           'background_type':'wood','background_angle':.12,'object_shadow':'none',
           'object_shadow_strength':.48,'cast_shadow_strength':.42,'cast_shadow_softness':.018,
           'tear_active':0.,'occlusion_active':0.,'pen_active':0.,'artifact_present':0.}
    cases=[
        ('wood-flat','Papel plano sobre madeira',{}),
        ('wood-crumpled','Papel amassado sobre madeira',{'surface_mode':'crumpled','layout':'3x4+r3','surface_strength':1.}),
        ('marble-fold','Dobra sobre marmore',{'surface_mode':'fold','background_type':'marble','paper_palette':1,'surface_axis':.45}),
        ('laminate-curl','Curvatura sobre laminado',{'surface_mode':'curl','background_type':'laminate','layout':'6x2+r2','paper_palette':2}),
        ('fabric-accordion','Sanfona sobre tecido',{'surface_mode':'accordion','background_type':'fabric','layout':'3x4+r2','surface_frequency':3.8}),
        ('metal-wrinkles','Rugas finas sobre metal',{'surface_mode':'wrinkled','background_type':'metal','paper_palette':4,'palette':1}),
        ('clinical-corner','Canto levantado sobre bancada',{'surface_mode':'corner','background_type':'clinical','paper_palette':5,'corner_position':[1.,1.],'corner_size':.45}),
        ('leather-roll','Papel enrolado sobre couro',{'surface_mode':'rolled','background_type':'leather','surface_axis':-.30,'paper_palette':3}),
        ('hand-shadow','Sombra de mao',{'object_shadow':'hand','object_shadow_position':[-.35,.20],'object_shadow_size':[.95,.90],'object_shadow_angle':.20,'object_shadow_softness':.02}),
        ('phone-shadow','Sombra de celular',{'object_shadow':'phone','object_shadow_position':[.60,-.15],'object_shadow_size':[.60,.85],'object_shadow_angle':.25,'object_shadow_softness':.035,'paper_palette':6}),
        ('blinds-shadow','Sombra de persiana',{'object_shadow':'blinds','object_shadow_angle':.32,'object_shadow_size':[.60,.65],'object_shadow_softness':.023,'paper_palette':2}),
        ('window-shadow','Sombra de janela',{'object_shadow':'window','object_shadow_angle':.1,'object_shadow_size':[1.1,.95],'object_shadow_softness':.04,'background_type':'plain'}),
        ('cabrera-aged','Cabrera e papel envelhecido',{'layout':'3x4-cabrera+r3','surface_mode':'wrinkled','paper_palette':7,'palette':3,'paper_age':.08,'thermal_fade':.30}),
        ('tear-annotation','Rasgo e anotacao',{'surface_mode':'curl','tear_active':1.,'tear_side':0,'tear_position':.2,'tear_size':.15,
                                            'artifact_present':[1.,0.,0.,0.],'artifact_x':.65,'artifact_y':.44,'artifact_angle':-.25,'artifact_text':0}),
        ('two-long','Duas derivacoes longas',{'layout':'2x1-long','surface_mode':'fold','paper_palette':1,'background_type':'laminate'}),
        ('three-long','Tres derivacoes longas',{'layout':'3x1-long','surface_mode':'crumpled','paper_palette':0,'background_type':'wood'})]
    thumb_w=384
    thumb_h=round(height*thumb_w/width)
    mosaic=Image.new('RGB',(thumb_w*4,(thumb_h+32)*4),'white')
    draw=ImageDraw.Draw(mosaic)
    entries=[]
    for index,(name,title,parameters) in enumerate(cases):
        seed=62000+index*97
        batch=renderer(signals,fs,parameters={**fixed,**parameters},stage='all',return_field=True,
                       return_instances=True,return_scene=True,generator=torch.Generator(device=device).manual_seed(seed))
        image=png(batch.image[0])
        extension='.jpg' if image_format=='jpeg' else '.png'
        if image_format=='jpeg':
            image.save(out/(name+extension),quality=95,subsampling=0)
        else:
            image.save(out/(name+extension))
        metadata={'seed':seed,'source':'entirely artificial ECG','metadata':plain(batch.metadata)}
        (out/(name+'.json')).write_text(json.dumps(metadata,indent=2))
        entries.append({'name':name,'title':title,'seed':seed,'image':name+extension})
        x,y=(index%4)*thumb_w,(index//4)*(thumb_h+32)
        draw.text((x+8,y+8),title,fill='black')
        mosaic.paste(image.resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(x,y+32))
        if name=='wood-crumpled':
            image.save(out/'wood-crumpled.png')
            png((batch.scene['normals'][0]+1)/2).save(out/'crumpled-normals.png')
            height_map=batch.scene['height'][0]
            png((height_map/height_map.max().clamp_min(1e-6)).expand(3,-1,-1)).save(out/'crumpled-height.png')
            targets=batch.targets['traces'][0].amax(0)*batch.targets['valid'][0,0]
            overlay=torch.lerp(batch.image[0],torch.tensor([.05,.95,.30],device=device)[:,None,None].expand_as(batch.image[0]),targets[None]*.65)
            png(overlay).save(out/'crumpled-overlay.png')
    mosaic.save(out/'scene-preview.png')
    layout_dir=out/'layouts'
    layout_dir.mkdir(exist_ok=True)
    clean=ECGRenderer(RenderConfig(height=480,width=640,profile='clean',antialias_samples=3)).to(device)
    layout_mosaic=Image.new('RGB',(640*3,(480+32)*10),'white')
    ld=ImageDraw.Draw(layout_mosaic)
    for index,spec in enumerate(clean.layout_specs):
        params={'layout':spec.name,'speed_index':1,'gain_index':1,'lead_present':1.,'label_present':1.,'label_below':False,
                'paper_palette':0,'paper_tint':0.,'palette':0,'grid_strength':.30,'grid_style':0,
                'pulse_present':1.,'artifact_present':0.,'line_width':1.1}
        batch=clean(signals,fs,parameters=params,photo=False,stage='metadata',generator=torch.Generator(device=device).manual_seed(191+index))
        image=png(batch.image[0])
        image.save(layout_dir/(spec.name+'.png'))
        x,y=index%3*640,index//3*512
        ld.text((x+10,y+8),spec.name,fill='black')
        layout_mosaic.paste(image,(x,y+32))
    if image_format=='jpeg':
        layout_mosaic.save(out/'layout-preview.jpg',quality=95,subsampling=0)
    else:
        layout_mosaic.save(out/'layout-preview.png')
    report={'schema':'ecg-replot-scene-gallery-v2','device':str(device),'torch_version':str(torch.__version__),
            'real_patient_records':0,'clinical_validation':False,'realism_comparison_executed':False,
            'configuration':renderer.specification(),'scenes':entries,'layouts':[asdict(spec) for spec in clean.layout_specs],
            'procedural_geometry':'2.5D height fields with inverse relief projection and normal-based shading',
            'external_3d_assets_used':False,
            'preview_encoding':image_format+' outside training, JPEG quality 95 when selected',
            'lossless_audit_example':'wood-crumpled.png'}
    (out/'gallery.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--height',type=int,default=768)
    parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--format',choices=['jpeg','png'],default='jpeg')
    parser.add_argument('--zip',type=Path)
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. CPU output verifies software only')
    torch.set_num_threads(args.threads)
    report=export_gallery(args.out,torch.device(args.device),args.height,args.width,args.format)
    if args.zip:
        args.zip.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in sorted(args.out.rglob('*')):
                if path.is_file():
                    archive.write(path,'ecg-generator-examples-v2/'+str(path.relative_to(args.out)))
        with zipfile.ZipFile(args.zip) as archive:
            if archive.testzip():
                raise AssertionError('Gallery archive checksum failed')
    print(json.dumps({'scenes':len(report['scenes']),'layouts':len(report['layouts']),'device':report['device'],
                      'out':str(args.out.resolve()),'zip':str(args.zip) if args.zip else None}))


if __name__=='__main__':
    main()
