"""Regressions for the independently reproduced October 2026 audit cases."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from PIL import ImageFont

from ecg_gpu import (ECGImageLayer, ECGRenderer, ECGSupervisionLayer,
                     RenderConfig, SupervisionConfig, detection_targets,
                     render_seed, samples_from_batch, export_yolo_dataset)
from ecg_gpu.gpu_appearance import CameraAppearance
from ecg_gpu.gpu_renderer import material_pixel_area


class AuditFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.signal = torch.zeros(1, 12, 1001)

    def config(self, **options):
        return RenderConfig(**dict(dict(height=128, width=192, profile='photo',
            surface_mode='off', surface_resolution=32, shadow_samples=0,
            antialias_samples=1), **options))

    def fixed(self, **options):
        return dict(dict(layout='3x4+r1', quarter=0, angle=0., zoom=1.,
            translate_x=0., translate_y=0., perspective_x=0., perspective_y=0.,
            radial=0., curl_x=0., curl_y=0., crease_warp=0.,
            surface_strength=0., asset_active=0., lead_present=1.,
            header_present=1., pen_active=0., artifact_present=0.,
            occlusion_active=0., tear_active=0., sensor_active=0., jpeg_active=0.), **options)

    def render(self, renderer, p, **options):
        return renderer(self.signal, 100, parameters=p,
                        generator=torch.Generator().manual_seed(42), **options)

    def test_quarter_override_recomposes_the_rotation_and_image_basis(self):
        r = ECGRenderer(self.config())
        sampled = r.sample_parameters(1, generator=torch.Generator().manual_seed(42))
        jitter = sampled['angle'][0] + int(sampled['quarter'][0]) * math.pi / 2
        for quarter in range(4):
            b = self.render(r, {'quarter': quarter}, return_field=True)
            angle = float(b.metadata['parameters']['angle'][0])
            self.assertAlmostEqual(angle, -quarter * math.pi / 2 + float(jitter), places=6)
            self.assertEqual(int(b.targets['orientation'][0]), quarter)
        # Isolate rotation and check its actual image-coordinate basis.
        b = self.render(r, self.fixed(quarter=1, angle=-math.pi/2), return_field=True)
        field = b.inverse_field[0]
        self.assertLess(abs(float(field[64, 97, 0]-field[64, 96, 0])), 1e-6)
        self.assertGreater(float(field[64, 97, 1]-field[64, 96, 1]), 0.)

    def test_absolute_angle_derives_orientation_and_inconsistent_pair_is_rejected(self):
        r = ECGRenderer(self.config())
        for angle, expected in [(0., 0), (-math.pi/2, 1), (math.pi, 2), (math.pi/2, 3)]:
            b = self.render(r, {'angle': angle})
            self.assertEqual(int(b.targets['orientation'][0]), expected)
        with self.assertRaisesRegex(RuntimeError, 'contradicts'):
            self.render(r, {'angle': math.pi, 'quarter': 0})

    def test_photo_parameter_gates_geometry_and_image_layer_has_static_photo_option(self):
        r = ECGRenderer(self.config())
        a = self.render(r, {'photo': 0}, return_field=True)
        b = self.render(r, {'photo': 1}, return_field=True)
        y, x = torch.meshgrid((torch.arange(128)+.5)*2/128-1,
                              (torch.arange(192)+.5)*2/192-1, indexing='ij')
        torch.testing.assert_close(a.inverse_field[0], torch.stack((x,y),-1), rtol=0, atol=2e-7)
        self.assertFalse(torch.equal(a.image, b.image))
        layer = ECGImageLayer(fs=100, config=self.config(), photo=False, seed=42).eval()
        self.assertTrue(torch.equal(layer(self.signal), layer.render_batch(self.signal, photo=False, stage='image').image))

    def test_perspective_preserves_fully_visible_unit_labels_with_analytic_containment(self):
        r = ECGRenderer(self.config(width=640, height=384, profile='clean'))
        p = self.fixed(font_size=12., speed_index=1, gain_index=1)
        canonical = self.render(r, p, photo=False)
        _, _, units = r._text(canonical.metadata, canonical.metadata['parameters'], False)
        ys, xs = (units[0,0] > 0).nonzero(as_tuple=True)
        u, v = (xs-319.5)/320, (ys-191.5)/320
        for perspective in (.22, .35):
            # Direct forward projective equation, independent of renderer helpers.
            x, y = u/(1-perspective*u), v/(1-perspective*u)
            self.assertTrue(((x.abs()<1)&(y.abs()<384/640)).all())
            observed = self.render(r, dict(p,perspective_x=perspective))
            self.assertTrue(observed.metadata['unit_text_visible'][0])
            self.assertEqual(int(observed.targets['speed'][0]), 2)
            self.assertEqual(int(observed.targets['gain'][0]), 2)
        hidden = self.render(r, dict(p,occlusion_active=1.,occlusion_x=0.,occlusion_y=0.,
                                    occlusion_w=1.,occlusion_h=.2))
        self.assertFalse(hidden.metadata['unit_text_visible'][0])

    def test_inverse_area_matches_an_analytic_projective_jacobian(self):
        h,w,k = 80,120,.22
        y,x=torch.meshgrid((torch.arange(h)+.5-h/2)/(w/2),
                           (torch.arange(w)+.5-w/2)/(w/2),indexing='ij')
        denominator=1+k*x
        field=torch.stack((x/denominator,y/denominator*w/h),-1)[None]
        observed=material_pixel_area(field,h,w)[0,0,2:-2,2:-2]
        expected=denominator[2:-2,2:-2].pow(-3)
        torch.testing.assert_close(observed,expected,rtol=3e-5,atol=1e-5)

    def test_rectified_document_removes_paper_and_texture_defers_removal(self):
        r = ECGRenderer(self.config(profile='clean'))
        p = self.fixed(hole_active=[1.,0.,0.],hole_center=[[0.,0.]]*3,
            hole_radius=[[.2,.2],[0.,0.],[0.,0.]],tear_raggedness=0.)
        final = self.render(r,p,photo=False,return_instances=True)
        texture = self.render(r,p,photo=False,as_texture=True,return_instances=True)
        hole=final.targets['paper_support']==0
        self.assertGreater(int(hole.sum()),500)
        self.assertTrue((final.targets['valid'][hole]==0).all())
        self.assertTrue((texture.targets['valid'][hole]==1).all())
        self.assertTrue((final.image[hole.expand_as(final.image)]==1).all())
        layer=ECGSupervisionLayer(config=r.config)
        supervised=layer.supervise(final)
        self.assertTrue((supervised['lead_masks']*hole==0).all())
        self.assertTrue(torch.equal(final.reference,self.signal))
        with self.assertRaisesRegex(ValueError,'requires photo=False'):
            self.render(r,p,as_texture=True)

    def test_static_glyph_bounds_cover_large_fonts_and_long_atlas_tokens(self):
        long='M'*80
        r=ECGRenderer(self.config(width=1024,height=256,max_font_size=64),artifact_texts=(long,))
        font=ImageFont.truetype('DejaVuSans.ttf',24)
        self.assertGreaterEqual(r.atlas.shape[-1],math.ceil(font.getlength(long)))
        args=(torch.tensor([[3]]),torch.tensor([[10.]]),torch.tensor([[10.]]),
              torch.tensor([[55.]]),torch.tensor([[1.]]))
        observed=r._stamp(*args)
        reference=r._stamp(*args,patch=(128,384))
        self.assertTrue(torch.equal(observed,reference))
        self.assertGreater(float(observed[...,42:,:].sum()),0.)
        with self.assertRaisesRegex(RuntimeError,'max_font_size'):
            self.render(r,self.fixed(font_size=100.))

    def test_instance_visibility_is_supplied_and_export_filter_is_end_to_end(self):
        layer=ECGSupervisionLayer(fs=100,config=self.config(profile='clean'),photo=False,
            parameters=self.fixed(),randomize=False,
            supervision=SupervisionConfig(outputs=('instance_boxes',)))
        a=layer(self.signal)
        b=layer(self.signal,parameters={'occlusion_active':1.,'occlusion_x':0.,'occlusion_y':0.,
                                      'occlusion_w':1.,'occlusion_h':1.})
        self.assertTrue((a['instance_visible_fraction'][a['instance_active']]>.99).all())
        self.assertTrue((b['instance_visible_fraction']==0).all())
        self.assertGreater(len(detection_targets(a,min_visible_fraction=.5)['cls']),0)
        self.assertEqual(len(detection_targets(b,min_visible_fraction=.5)['cls']),0)
        records=samples_from_batch(a,source_ids=['source'],group_ids=['group'],sample_ids=['sample'])
        with tempfile.TemporaryDirectory() as tmp:
            manifest=export_yolo_dataset(records,Path(tmp)/'out',val_fraction=0.,min_visible_fraction=.5)
            self.assertEqual(manifest['min_visible_fraction'],.5)
            self.assertGreater(manifest['samples'][0]['num_objects'],0)

    def test_streams_are_independent_replayable_and_checkpointed(self):
        a=ECGImageLayer(fs=100,config=self.config(),stream_id=1)
        b=ECGImageLayer(fs=100,config=self.config(),stream_id=2)
        replay=ECGImageLayer(fs=100,config=self.config(),stream_id=1)
        first=a(self.signal)
        self.assertTrue(torch.equal(first,replay(self.signal)))
        self.assertFalse(torch.equal(first,b(self.signal)))
        state=copy.deepcopy(a.state_dict())
        expected=a(self.signal)
        restored=ECGImageLayer(fs=100,config=self.config(),stream_id=0)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(expected,restored(self.signal)))
        self.assertEqual(render_seed(42,'source',view_id=3),render_seed(42,'source',view_id=3))
        self.assertNotEqual(render_seed(42,'source',view_id=3),render_seed(42,'source',view_id=4))

    def test_static_direct_camera_skips_disabled_work_with_exact_rng_parity(self):
        from test_capture_physics import parameters
        image=torch.linspace(0,1,3*16*24).reshape(1,3,16,24)
        p=parameters(sensor_active=0.,jpeg_active=0.,read_noise=.01)
        full=CameraAppearance()
        fast=CameraAppearance(mode='direct',jpeg_enabled=False)
        ga,gb=(torch.Generator().manual_seed(23) for _ in range(2))
        expected=full(image,p,ga)
        with mock.patch.object(fast,'optical_blur',side_effect=AssertionError('unexpected blur')), \
             mock.patch.object(fast,'jpeg',side_effect=AssertionError('unexpected JPEG')):
            observed=fast(image,p,gb)
        self.assertTrue(torch.equal(expected,observed))
        self.assertTrue(torch.equal(ga.get_state(),gb.get_state()))
        sensor=CameraAppearance(mode='sensor',jpeg_enabled=False)
        p=parameters(sensor_active=1.,jpeg_active=0.,blur=.7,shot_noise=.01)
        expected=full(image,p,torch.Generator().manual_seed(23))
        observed=sensor(image,p,torch.Generator().manual_seed(23))
        self.assertTrue(torch.equal(expected,observed))

    def test_invalid_integer_controls_and_fractional_presence_are_rejected_early(self):
        for key in ('height','width','antialias_samples','lead_chunk_size',
                    'surface_resolution','inverse_iterations','shadow_samples'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'integer'):
                RenderConfig(**{key:1.5})
        r=ECGRenderer(self.config())
        for key in ('lead_present','photo'):
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'binary'):
                self.render(r,{key:.001})

    def test_demo_limb_identities_and_morphology_are_not_twelve_scalar_copies(self):
        import numpy as np
        from ecg_gpu.demo import demo_signals
        signals,_,_=demo_signals()
        i,ii,iii,avr,avl,avf=signals[:6]
        for actual,expected in [(iii,ii-i),(avr,-(i+ii)/2),(avl,i-ii/2),(avf,ii-i/2)]:
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
        self.assertGreater(np.linalg.matrix_rank(signals[6:]),1)

    def test_cli_rectified_targets_and_stream_provenance(self):
        import numpy as np
        from ecg_gpu.gpu_cli import render_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.savez(root/'source.npz',signals=self.signal.numpy(),fs=100.)
            report=render_file(root/'source.npz',root/'image.png',device='cpu',
                config=self.config(profile='clean'),photo=False,stream_id=3,
                parameters=self.fixed(),targets_dir=root/'targets',
                supervision=SupervisionConfig(outputs=('instance_visible_fraction',)))
            self.assertEqual(report['stream_id'],3)
            with np.load(root/'targets'/'targets.npz') as data:
                self.assertEqual(data['instance_visible_fraction'].shape,(1,15))


if __name__ == '__main__':
    unittest.main()
