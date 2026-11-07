"""Coherent, reproducible capture recipes built from existing torch controls.

These are engineering stress scenarios, not fitted camera/device models.
They change the document and capture only, never the source ECG physiology.
"""
import math

from .gpu_parameters import Choice, Uniform
from .gpu_renderer import RenderConfig
from .convenience import did_you_mean


CAPTURE_SCENARIOS = (
    'clean_print', 'flatbed_scan', 'aged_scan', 'phone_photo',
    'low_light_phone', 'folded_photo', 'damaged_archive',
    'crumpled_photo', 'folded_archive', 'water_damage', 'coffee_spill',
    'torn_edges', 'cut_sheet', 'missing_fragments', 'oblique_phone',
    'thermal_damage',
)


def capture_scenario(name, *, height=768, width=1024, severity=.6):
    """Return a fresh ``(RenderConfig, parameters)`` pair for ECGImageLayer.

    ``severity`` scales adverse effects within a scenario. It is not a
    calibrated probability of failure and zero does not switch capture type.
    Layout, speed and gain can be overridden in the returned dictionary.
    Explicit missing-lead, label and pulse dropout is disabled. Damage and
    shadows can still hide printed content, tracked by the usual valid mask.
    """
    if name not in CAPTURE_SCENARIOS:
        raise ValueError(f'Unknown capture scenario {name!r}. Choose from {CAPTURE_SCENARIOS}.'
                         + did_you_mean(name, CAPTURE_SCENARIOS))
    if isinstance(severity, bool) or not isinstance(severity, (int, float)) or not math.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError('Scenario severity must be finite and in [0,1]')
    s = float(severity)
    photo = name not in {'clean_print', 'flatbed_scan', 'aged_scan'}
    profile = 'clean' if name == 'clean_print' else 'photo' if photo else 'scan'
    surface = {'folded_photo':'fold', 'folded_archive':'accordion',
               'damaged_archive':'wrinkled', 'crumpled_photo':'crumpled',
               'water_damage':'wrinkled', 'torn_edges':'curl',
               'missing_fragments':'wrinkled'}.get(name, 'flat')
    config = RenderConfig(height=height, width=width, profile=profile,
                          severity=s, surface_mode=surface,
                          camera_mode='direct' if name == 'clean_print' else 'sensor',
                          jpeg_enabled=photo)
    p = {
        'layout': Choice(['3x4+r1', '6x2', '12x1']),
        'speed_index': 1, 'gain_index': 1, 'sequential': True,
        'quarter': 0, 'lead_present': 1., 'label_present': 1.,
        'pulse_present': 1., 'header_present': 1., 'label_below': False,
        'grid_present': 1., 'grid_style': 0, 'paper_palette': 0,
        'paper_tint': 0., 'paper_color': [1., 1., 1.],
        'ink_color': [.035, .035, .035], 'ink_density': .97,
        'grid_strength': .30, 'minor_strength': .30,
        'line_width': 1.1, 'grid_width': .28,
        'artifact_present': 0., 'pen_active': 0.,
        'occlusion_active': 0., 'tear_active': 0., 'asset_active': 0.,
        'coffee_strength': 0., 'stain_strength': 0., 'bleed_strength': 0.,
        'paper_age': 0., 'thermal_fade': 0., 'band_strength': 0.,
        'fiber_strength': .12*s, 'foxing_strength': 0., 'water_strength': 0.,
        'thermal_darkening_strength': 0., 'edge_wear_strength': 0.,
        'hole_active': 0., 'cut_active': 0., 'missing_corner_active': 0.,
        'object_shadow': 'none', 'shadow_strength': 0.,
        'crease_strength': 0., 'curl_x': 0., 'curl_y': 0., 'crease_warp': 0.,
        'photo': float(photo),
    }
    if not photo:
        # Flatbed capture has neither tabletop perspective nor directional
        # camera shake. Scanner banding and print ageing remain separate.
        p.update(angle=0., zoom=1., translate_x=0., translate_y=0.,
                 perspective_x=0., perspective_y=0., radial=0.,
                 surface_strength=0., relief_view=[0., 0.],
                 ambient_light=1., light_energy=0., flash_strength=0.,
                 specular_strength=0., ambient_occlusion=0., cast_shadow_strength=0.,
                 motion=0., vignette=0., light_x=0., light_y=0.,
                 exposure=1., white_balance=[1., 1., 1.], gamma=1.,
                 blur=0. if name == 'clean_print' else Uniform(.15, .15 + .55*s),
                 sensor_active=float(name != 'clean_print'),
                 read_noise=0. if name == 'clean_print' else Uniform(0., .004*s),
                 shot_noise=0., jpeg_active=0., paper_noise=0. if name == 'clean_print' else Uniform(0., .006*s))
    else:
        p.update(angle=Uniform(-.12*s, .12*s), zoom=Uniform(.78, .91),
                 translate_x=Uniform(-.04*s, .04*s), translate_y=Uniform(-.03*s, .03*s),
                 perspective_x=Uniform(-.22*s, .22*s), perspective_y=Uniform(-.14*s, .14*s),
                 radial=Uniform(-.025*s, .025*s),
                 relief_view=Uniform([-.4*s, -.3*s], [.4*s, .3*s]),
                 surface_strength=0., background_type=Choice(['wood', 'clinical', 'laminate']),
                 ambient_light=Uniform(.65, .82), light_energy=[.19, .10, .06],
                 flash_strength=0., specular_strength=.025*s,
                 exposure=Uniform(.90, 1.05),
                 white_balance=Choice([[1.015, 1., .985], [1., 1., 1.], [.99, 1., 1.01]]),
                 light_color=[[1., .99, .97], [.97, .99, 1.], [1., 1., 1.]],
                 background_tint=[1., 1., 1.],
                 blur=Uniform(.2, .2+.65*s), motion=Uniform(0., 1.2*s),
                 sensor_active=1., read_noise=Uniform(.0005*s, .003*s),
                 shot_noise=Uniform(.0005*s, .003*s),
                 jpeg_active=1., jpeg_quality=Uniform(92.-22.*s, 96.-8.*s),
                 object_shadow=Choice(['none', 'phone', 'hand']),
                 object_shadow_strength=Uniform(0., .3*s))
    if name == 'aged_scan':
        p.update(paper_age=Uniform(.12*s, .32*s), foxing_strength=Uniform(.12*s, .4*s),
                 thermal_fade=Uniform(.1*s, .5*s),
                 band_strength=Uniform(.003*s, .025*s),
                 stain_strength=Uniform(0., .15*s),
                 bleed_strength=Uniform(0., .04*s))
    if name == 'low_light_phone':
        p.update(ambient_light=Uniform(.24, .40), light_energy=[.07, .03, .02],
                 exposure=Uniform(.70-.25*s, .85-.15*s),
                 motion=Uniform(.5*s, 3.2*s),
                 read_noise=Uniform(.006*s, .018*s),
                 shot_noise=Uniform(.004*s, .012*s))
    if name == 'folded_photo':
        p.update(surface_strength=Uniform(.35*s, .85*s),
                 ambient_occlusion=Uniform(.15*s, .35*s),
                 cast_shadow_strength=Uniform(.25*s, .55*s))
    if name == 'damaged_archive':
        p.update(surface_strength=Uniform(.2*s, .65*s),
                 paper_age=Uniform(.12*s, .30*s), thermal_fade=Uniform(.15*s, .55*s),
                 coffee_strength=Uniform(.1*s, .4*s), foxing_strength=.35*s,
                 edge_wear_strength=.55*s, bleed_strength=Uniform(0., .07*s),
                 tear_active=1., tear_size=Uniform(.015+.02*s, .015+.10*s),
                 artifact_present=[1., 0., 0., 0.])
    if name == 'crumpled_photo':
        p.update(surface_strength=Uniform(.65*s, .92*s), crease_count=4,
                 fold_amplitude=[.032]*8, fold_span=[.30]*8,
                 fold_width=[.04]*8, fold_length=[1.1]*8, surface_axis=.35,
                 relief_view=[.52, -.32], ambient_occlusion=0.,
                 cast_shadow_strength=.45*s, object_shadow='none')
    if name == 'folded_archive':
        p.update(surface_strength=Uniform(.5*s, .9*s), surface_frequency=2.6,
                 paper_age=.22*s, foxing_strength=.3*s, thermal_fade=.2*s,
                 edge_wear_strength=.5*s, cast_shadow_strength=.5*s,
                 ambient_occlusion=.20*s, object_shadow='none')
    if name in {'crumpled_photo','folded_archive'}:
        # Reveal actual normals with raking light instead of increasing
        # stretching or painting dark stripes onto a flat document.
        p.update(ambient_light=.26,
                 light_direction=[[-1.,.3,.28],[.2,-.1,1.],[0.,0.,1.]],
                 light_energy=[.85,.04,.025], exposure=Uniform(.82,.92),
                 specular_strength=.01)
    if name == 'water_damage':
        p.update(surface_strength=.45*s, water_strength=.8*s, stain_strength=.25*s,
                 stain_x=.62, stain_y=.52, paper_age=.08*s,
                 thermal_fade=.55*s, fade_position=.64, fade_width=.18,
                 foxing_strength=.15*s, object_shadow='none')
    if name == 'coffee_spill':
        p.update(coffee_strength=.75*s, coffee_radius=.19, stain_strength=.18*s,
                 stain_x=.64, stain_y=.56, water_strength=.25*s,
                 object_shadow='none', paper_noise=0.)
    if name == 'torn_edges':
        p.update(surface_strength=.32*s, tear_active=1., tear_count=3,
                 tear_size=.13*s+.01, tear_raggedness=.95,
                 edge_wear_strength=.85*s, edge_wear_depth=.012,
                 tear_position=-.18, object_shadow='none')
    if name == 'cut_sheet':
        p.update(cut_active=[1.,1.,0.], cut_center=[[-.1,-.2],[.5,.3],[0.,0.]],
                 cut_length=[.7,.28,.1], cut_width=[.008,.004,.004],
                 cut_angle=[.48,-.75,0.], tear_raggedness=.15,
                 zoom=.86, object_shadow='none')
    if name == 'missing_fragments':
        p.update(surface_strength=.3*s, missing_corner_active=1.,
                 missing_corner_index=2, missing_corner_size=[.35,.3],
                 hole_active=[1.,1.,0.], hole_center=[[-.35,.1],[.4,.4],[0.,0.]],
                 hole_radius=[[.09,.065],[.045,.035],[.03,.03]],
                 tear_raggedness=.85, object_shadow='none')
    if name == 'oblique_phone':
        p.update(perspective_x=.37*s, perspective_y=-.24*s,
                 angle=-.17*s, zoom=.72, translate_x=-.05,
                 radial=.035*s, object_shadow='none', blur=.35)
    if name == 'thermal_damage':
        p.update(paper_age=.12*s, thermal_darkening_strength=.65*s,
                 thermal_fade=.6*s, fade_position=.70, fade_width=.13,
                 foxing_strength=.18*s, object_shadow='none')
    return config, p

