"""Deterministic raster blocks, independently compilable without the RNG.

The blocks preserve the bounded-segment and original-sample envelope model.
Compilation is opt-in. No crop, row band, amplitude cap or host tensor read is
introduced. Geometry and camera random sampling stay outside these kernels.
"""
import torch


def curve_coverage(ya,yb,px,yy,thickness,within,low,high,preserve_extrema,reduce_leads):
    """[B,C,S,W] segment endpoints -> coverage, optionally union over C."""
    samples=ya.shape[2]
    result=None
    for sample in range(samples):
        a=ya[:,:,sample,None,:]
        dy=(yb[:,:,sample]-ya[:,:,sample])[:,:,None,:]
        dx=1/samples
        x=px[sample]
        py=yy-a
        t=((x*dx+py*dy)/(dx*dx+dy.square())).clamp(0,1)
        distance=((x-t*dx).square()+(py-t*dy).square()).sqrt()
        coverage=(thickness-distance).clamp(0,1)
        result=coverage if result is None else torch.maximum(result,coverage)
    if preserve_extrema:
        exists=torch.isfinite(low)[:,:,None,:]
        distance=torch.maximum(low[:,:,None,:]-yy,yy-high[:,:,None,:]).clamp_min(0)
        envelope=(thickness-distance).clamp(0,1)
        result=torch.maximum(result,torch.where(exists,envelope,0.))
    result=result*within
    return result.amax(1,keepdim=True) if reduce_leads else result


def calibration_coverage(base,scale,left,ppm,speed,thickness,present,active,xx,yy):
    """Union of calibration marks, without retaining per-lead page tensors."""
    right=left[:,None]-ppm[:,None]*2
    start=right-speed[:,None]*ppm[:,None]*.2
    top=base-scale
    vertical=((thickness-(xx-start[:,:,None,None]).abs()).clamp(0,1)
              +(thickness-(xx-right[:,:,None,None]).abs()).clamp(0,1))
    vertical=vertical*((yy>=top[:,:,None,None])&(yy<=base[:,:,None,None]))
    horizontal=(thickness-(yy-top[:,:,None,None]).abs()).clamp(0,1)
    horizontal=horizontal*((xx>=start[:,:,None,None])&(xx<=right[:,:,None,None]))
    foot=(thickness-(yy-base[:,:,None,None]).abs()).clamp(0,1)
    foot=foot*((xx>=start[:,:,None,None]-ppm[:,None,None,None])&(xx<start[:,:,None,None]))
    pulse=(vertical+horizontal+foot).clamp(0,1)
    pulse=pulse*active[:,:,None,None]
    pulse=pulse*present[:,:,None,None]
    return pulse.amax(1,keepdim=True)


def raster_kernels(compiled=False):
    """Lazy compilation, shape-adaptive, with no silent eager fallback."""
    if not isinstance(compiled,bool):
        raise ValueError('compile_raster must be a boolean')
    if not compiled:
        return curve_coverage,calibration_coverage
    # fullgraph rejects unsupported blocks instead of silently graph-breaking.
    # RNG/counters stay eager. Different antialias counts can specialize again.
    return (torch.compile(curve_coverage,fullgraph=True,dynamic=True),
            torch.compile(calibration_coverage,fullgraph=True,dynamic=True))
