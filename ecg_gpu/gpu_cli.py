"""Convert NPY/NPZ or optional WFDB 12-lead signals into rendered images."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .gpu_renderer import RenderConfig, DocumentAssets
from .gpu_module import ECGImageLayer
from .gpu_supervision import ECGSupervisionLayer, SupervisionConfig
from .gpu_parameters import parameters_from_json
from .schema_v05 import LEADS, SPEEDS, GAINS


def load_signals(path,fs=None,units=None):
    path=Path(path)
    names=None
    lengths=None
    declared_units=None
    if path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as source:
            signals=np.asarray(source['signals'],dtype=np.float32)
            if fs is None and 'fs' in source:
                fs=np.asarray(source['fs'],dtype=np.float32)
            names=[name.decode() if isinstance(name,bytes) else str(name) for name in source['lead_names']] if 'lead_names' in source else None
            lengths=np.asarray(source['lengths']) if 'lengths' in source else None
            if 'units' in source:
                tag=source['units'].item()
                declared_units=tag.decode() if isinstance(tag,bytes) else str(tag)
    elif path.suffix.lower()=='.npy':
        signals=np.asarray(np.load(path,allow_pickle=False),dtype=np.float32)
    elif path.suffix.lower()=='.hea':
        if units is not None:
            raise ValueError('WFDB units are read from each channel. --units applies only to NPY/NPZ')
        try:
            import wfdb
        except ImportError as error:
            raise ValueError('WFDB input needs the optional wfdb dependency') from error
        record=wfdb.rdrecord(str(path.with_suffix('')))
        signals=np.asarray(record.p_signal.T,dtype=np.float32)
        names=list(record.sig_name)
        fs=record.fs if fs is None else fs
        unit_scale={'mv':1.,'uv':.001,'µv':.001,'v':1000.}
        for index,unit in enumerate(record.units):
            if str(unit).lower() not in unit_scale:
                raise ValueError('Unsupported WFDB voltage unit')
            signals[index]*=unit_scale[str(unit).lower()]
        declared_units='mV'
    else:
        raise ValueError('Input must be .npz, .npy or optional WFDB .hea')
    if signals.ndim==2:
        signals=signals[None]
    if signals.ndim!=3 or signals.shape[0]<1 or signals.shape[1]!=12 or signals.shape[2]<2:
        raise ValueError('Signal shape must be [12,T] or [B,12,T]')
    if names is not None:
        normalized=[name.upper() for name in names]
        if len(normalized)!=12 or len(set(normalized))!=12 or set(normalized)!=set(name.upper() for name in LEADS):
            raise ValueError('Exactly twelve standard lead names required')
        signals=signals[:,[normalized.index(name.upper()) for name in LEADS]]
    unit='mV' if units is None and declared_units is None else units or declared_units
    scale={'mv':1.,'uv':.001,'µv':.001,'v':1000.}.get(str(unit).lower())
    if scale is None:
        raise ValueError('Supported input units are mV, uV or V')
    signals=signals*scale
    if fs is None:
        raise ValueError('Sampling rate required. Use --fs or NPZ fs')
    rates=np.asarray(fs,dtype=np.float32)
    if rates.ndim>1 or (rates.ndim==1 and len(rates) not in {1,len(signals)}):
        raise ValueError('fs must be scalar or one rate per signal')
    if not np.isfinite(signals).all() or not np.isfinite(rates).all() or (rates<=0).any():
        raise ValueError('Finite signals and positive sample rates required')
    if lengths is not None:
        if (lengths.shape!=(len(signals),)
                or not np.issubdtype(lengths.dtype,np.number)
                or np.iscomplexobj(lengths)
                or not np.isfinite(lengths).all()
                or (lengths != np.floor(lengths)).any()
                or (lengths < 2).any()
                or (lengths > signals.shape[-1]).any()):
            raise ValueError('Invalid padded source lengths: finite integer sample counts required')
        lengths=lengths.astype(np.int64)
    return signals.copy(),rates,lengths


def plain(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,dict):
        return {key:plain(item) for key,item in value.items()}
    return value


def normalized_input_fingerprint(signals,rates,lengths):
    """Hash the actual decoded input, including WFDB waveform content.

    Canonical arrays use C order and explicit little-endian float32/int64.
    The combined digest prefixes a compact schema/shape header and includes
    sampling rates and valid lengths, so equal voltage bytes at a different
    sampling rate do not have the same physical-input identity.
    """
    count,_,samples=signals.shape
    arrays={
        'signals':np.ascontiguousarray(signals,dtype='<f4'),
        'rates':np.ascontiguousarray(np.broadcast_to(rates,(count,)),dtype='<f4'),
        'lengths':np.ascontiguousarray(np.full(count,samples) if lengths is None else lengths,dtype='<i8'),
    }
    header={'schema':'ecg-normalized-input-v1','order':'C',
            'arrays':{name:{'shape':list(value.shape),'dtype':value.dtype.str}
                      for name,value in arrays.items()}}
    digest=hashlib.sha256(json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\0')
    for value in arrays.values():
        digest.update(value.tobytes(order='C'))
    return {**header,'sha256':digest.hexdigest(),
            'array_sha256':{name:hashlib.sha256(value.tobytes(order='C')).hexdigest()
                            for name,value in arrays.items()}}


def render_file(input_path,output_path,*,fs=None,units=None,device='auto',config=None,
                parameters=None,seed=1973,metadata_path=None,document_assets=None,fonts=None,compile_raster=False,
                targets_dir=None,supervision=None,photo=True,as_texture=False,stream_id=0):
    if as_texture and (photo or targets_dir is None):
        raise ValueError('Texture export requires photo=False and targets_dir to retain paper_support')
    signals,rates,lengths=load_signals(input_path,fs,units)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Choose cpu or install a suitable CUDA build/hardware')
    layer_type = ECGSupervisionLayer if targets_dir is not None else ECGImageLayer
    layer_options = {'supervision': supervision or SupervisionConfig()} if targets_dir is not None else {}
    layer=layer_type(config=config or RenderConfig(),parameters=parameters,seed=seed,randomize=False,
                        document_assets=document_assets,fonts=fonts,compile_raster=compile_raster,
                        photo=photo,stream_id=stream_id,**layer_options).to(device)
    renderer=layer.renderer
    value=torch.from_numpy(signals)
    if device.type=='cuda':
        value=value.pin_memory()
    value=value.to(device,non_blocking=True)
    needs_instances = targets_dir is not None and bool({'instance_masks','instance_boxes','instance_visible_fraction'} & set(layer.supervision.outputs))
    batch=layer.render_batch(value,torch.as_tensor(rates,device=device),lengths=lengths,
                            stage='trace' if targets_dir is not None else 'image',
                            return_baselines=False,return_instances=needs_instances,return_instance_boxes=False,
                            as_texture=as_texture)
    output=Path(output_path)
    if output.suffix.lower() in {'.png','.jpg','.jpeg'}:
        if len(signals)!=1:
            raise ValueError('A single image filename requires one input. Use an output directory for batches')
        paths=[output]
    else:
        output.mkdir(parents=True,exist_ok=True)
        paths=[output/f'ecg-{index:05d}.png' for index in range(len(signals))]
    for index,path in enumerate(paths):
        path.parent.mkdir(parents=True,exist_ok=True)
        array=(batch.image[index].cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
        Image.fromarray(array).save(path,**({'quality':95,'subsampling':0} if path.suffix.lower() in {'.jpg','.jpeg'} else {}))
    report={'schema':'ecg-torch-renderer-cli-v1','device':str(device),'torch_version':str(torch.__version__),
            'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            'normalized_input':normalized_input_fingerprint(signals,rates,lengths),
            'seed':seed,'stream_id':stream_id,'outputs':[str(path.resolve()) for path in paths],
            'image_shape':list(batch.image.shape),'signal_units':'mV','configuration':renderer.specification(),
            'compile_raster':compile_raster,'metadata':plain(batch.metadata)}
    if targets_dir is not None:
        target_output=Path(targets_dir)
        target_output.mkdir(parents=True,exist_ok=True)
        paired=layer.supervise(batch)
        encoded={key:tensor.detach().cpu().numpy() for key,tensor in paired.items() if key!='image'}
        if as_texture:
            encoded['paper_support']=batch.targets['paper_support'].detach().cpu().numpy()
        np.savez_compressed(target_output/'targets.npz',**encoded)
        boxes={key:plain(tensor) for key,tensor in paired.items() if 'box' in key or key.endswith('_ids') or key.startswith('instance_') and 'masks' not in key}
        (target_output/'boxes.json').write_text(json.dumps({'coordinate_format':'xyxy_pixels_exclusive',
            'empty_box':-1,'lead_names':list(layer.supervision.leads),'batch':boxes},indent=2)+'\n')
        for name in ['bw','wb','trace_mask']:
            if name in paired:
                arrays=(paired[name].detach().cpu().numpy()[:,0]*255).round().clip(0,255).astype(np.uint8)
                for index,array in enumerate(arrays):
                    Image.fromarray(array).save(target_output/f'{name}-{index:05d}.png')
        report['targets_directory']=str(target_output.resolve())
        report['target_shapes']={key:list(tensor.shape) for key,tensor in paired.items() if key!='image'}
    if metadata_path:
        path=Path(metadata_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fs',type=float)
    parser.add_argument('--units',choices=['mV','uV','V'],help='NPY/no unit tag defaults to mV')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--height',type=int,default=768)
    parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--profile',choices=['clean','scan','photo','mixed'],default='mixed')
    parser.add_argument('--severity',type=float,default=.6)
    parser.add_argument('--surface',choices=['mixed','off','flat','curl','fold','accordion','crumpled','wrinkled','rolled','corner'],default='mixed')
    parser.add_argument('--surface-resolution',type=int,default=96)
    parser.add_argument('--shadow-samples',type=int,default=8)
    parser.add_argument('--antialias',type=int,default=4)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=1973)
    parser.add_argument('--stream-id',type=int,default=0,help='Independent deterministic worker/rank stream')
    parser.add_argument('--rectified',action='store_true',help='Render a final rectified document without photographic geometry')
    parser.add_argument('--as-texture',action='store_true',help='Prepare intact albedo and separate damage support for 3D. Requires --rectified')
    parser.add_argument('--camera-mode',choices=['full','direct','sensor'],default='full')
    parser.add_argument('--no-jpeg',action='store_true',help='Omit JPEG computation through a static camera specialization')
    parser.add_argument('--max-font-size',type=float)
    parser.add_argument('--layout',default=None)
    parser.add_argument('--speed',type=float,choices=SPEEDS[1:])
    parser.add_argument('--gain',type=float,choices=GAINS[1:])
    parser.add_argument('--parameters',type=json.loads,default={},help='JSON fixed values or uniform/choice/bernoulli distributions')
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--document-assets',type=Path)
    parser.add_argument('--font',action='append')
    parser.add_argument('--compile-raster',action='store_true')
    parser.add_argument('--targets-dir',type=Path,help='Optional aligned targets.npz, boxes.json and BW/WB PNG exports')
    parser.add_argument('--targets',default='lead_masks,bw,wb,lead_boxes,instance_boxes',help='Comma-separated advanced supervision outputs')
    parser.add_argument('--target-leads',default=','.join(LEADS),help='Comma-separated canonical lead names in output channel order')
    parser.add_argument('--binary-targets',action='store_true')
    parser.add_argument('--target-threshold',type=float,default=.05)
    parser.add_argument('--projected-targets',action='store_true',help='Retain projected masks behind explicit opaque occluders')
    args=parser.parse_args()
    if args.threads<1 or args.seed<0 or not isinstance(args.parameters,dict):
        parser.error('Positive threads, nonnegative seed and parameter object required')
    torch.set_num_threads(args.threads)
    overrides=dict(args.parameters)
    if args.layout:
        overrides['layout']=args.layout
    if args.speed is not None:
        overrides['speed_index']=SPEEDS[1:].index(args.speed)
    if args.gain is not None:
        overrides['gain_index']=GAINS[1:].index(args.gain)
    try:
        config=RenderConfig(height=args.height,width=args.width,profile=args.profile,severity=args.severity,
                            surface_mode=args.surface,surface_resolution=args.surface_resolution,
                            shadow_samples=args.shadow_samples,antialias_samples=args.antialias,
                            camera_mode=args.camera_mode,jpeg_enabled=not args.no_jpeg,max_font_size=args.max_font_size)
        assets=DocumentAssets.from_npz(args.document_assets) if args.document_assets else None
        report=render_file(args.input,args.output,fs=args.fs,units=args.units,device=args.device,config=config,
                           photo=not args.rectified,as_texture=args.as_texture,stream_id=args.stream_id,
                           parameters=parameters_from_json(overrides),seed=args.seed,metadata_path=args.metadata,document_assets=assets,fonts=args.font,compile_raster=args.compile_raster,
                           targets_dir=args.targets_dir,supervision=SupervisionConfig(
                               outputs=tuple(item.strip() for item in args.targets.split(',') if item.strip()),
                               leads=tuple(item.strip() for item in args.target_leads.split(',') if item.strip()),
                               binary=args.binary_targets,threshold=args.target_threshold,visible_only=not args.projected_targets)
                           if args.targets_dir is not None else None)
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
        parser.error(str(error))
    print(json.dumps({key:report[key] for key in ['device','outputs','image_shape','signal_units','seed']}))


if __name__=='__main__':
    main()


