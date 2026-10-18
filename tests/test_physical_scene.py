"""Independent references for tabletop projection and radiometric boundaries."""
import unittest

import torch

from ecg_gpu import ECGRenderer, RenderConfig
from ecg_gpu.gpu_document import DocumentScene


class PhysicalSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shadow_translation_matches_ray_plane_intersection(self):
        # Ray P-tL intersects z=0 at (Px-z*Lx/Lz, Py-z*Ly/Lz).
        # A point caster lets us measure its centroid independently of splatting.
        mask = torch.zeros(1,1,39,51)
        mask[0,0,19,25] = 1
        height = torch.full_like(mask,.13)
        light = torch.tensor([.6,-.3,.8])
        pixels_per_unit = 25.
        offsets = (light[:2]/light[2]*pixels_per_unit)[None,None]
        shadow = DocumentScene._project_shadow(mask,height,offsets)[0,0]
        yy,xx = torch.meshgrid(torch.arange(39),torch.arange(51),indexing='ij')
        self.assertAlmostEqual(float(shadow.sum()),1.,places=6)
        self.assertAlmostEqual(float((shadow*xx).sum()),25-.13*.6/.8*25,places=5)
        self.assertAlmostEqual(float((shadow*yy).sum()),19+.13*.3/.8*25,places=5)

    def test_shadow_uses_each_caster_height_not_receiving_background_height(self):
        mask = torch.zeros(1,1,25,61)
        mask[0,0,12,20],mask[0,0,12,45] = 1,1
        height = torch.zeros_like(mask)
        height[0,0,12,20],height[0,0,12,45] = .1,.3
        shadow = DocumentScene._project_shadow(mask,height,torch.tensor([[[20.,0.]]]))
        self.assertEqual(float(shadow[0,0,12,18]),1.)
        self.assertEqual(float(shadow[0,0,12,39]),1.)
        self.assertEqual(float(shadow[0,0,12,20]),0.)
        self.assertEqual(float(shadow.sum()),2.)

    def test_shadow_does_not_wrap_at_sensor_boundary(self):
        mask = torch.zeros(1,1,25,31)
        mask[0,0,12,2] = 1
        shadow = DocumentScene._project_shadow(mask,torch.ones_like(mask),torch.tensor([[[8.,0.]]]))
        self.assertEqual(float(shadow.sum()),0.)

    def test_elevated_sheet_has_no_false_attached_contact_halo(self):
        scene = DocumentScene(80,120,resolution=48,shadow_samples=0)
        p = scene.sample_parameters(1,torch.ones(1),torch.ones(1),generator=torch.Generator().manual_seed(9))
        p.update(angle=torch.zeros(1),zoom=torch.ones(1))
        p['light_energy'].zero_()
        p['cast_shadow_strength'].fill_(1.)
        mask = torch.zeros(1,1,80,120)
        mask[:,:,20:60,30:90] = 1
        attached = scene.cast_shadow(mask,{'height':torch.zeros_like(mask)},p)
        lifted = scene.cast_shadow(mask,{'height':torch.ones_like(mask)*.1},p)
        self.assertGreater(float(attached.sum()),1.)
        self.assertLess(float(lifted.sum()),float(attached.sum())*1e-10)

    def test_flat_lambertian_sheet_has_uniform_normalized_irradiance(self):
        scene = DocumentScene(80,120,resolution=48,shadow_samples=4)
        p = scene.sample_parameters(1,torch.ones(1),torch.ones(1),generator=torch.Generator().manual_seed(22))
        p.update(crease_strength=torch.zeros(1))
        p['flash_strength'].zero_()
        height = torch.zeros_like(scene.x)
        normal = torch.cat((height,height,torch.ones_like(height)),1)
        p['specular_strength'].zero_()
        irradiance = scene.illumination(height,normal,p)
        torch.testing.assert_close(irradiance,torch.ones_like(irradiance),rtol=0,atol=2e-7)

    def test_linear_illumination_and_shadow_have_display_encoded_boundary(self):
        # Independent sRGB reference: halving a white surface's radiance gives
        # 1.055*(.5**(1/2.4))-.055, about .735, rather than display code .5.
        from ecg_gpu.gpu_appearance import linear_to_srgb,srgb_to_linear
        expected = 1.055*(.5**(1/2.4))-.055
        actual = linear_to_srgb(srgb_to_linear(torch.ones(1))*.5)
        self.assertAlmostEqual(float(actual),expected,places=6)

    def test_pen_and_stamp_validity_follows_the_same_material_inverse_warp(self):
        from torch.nn import functional as F
        renderer = ECGRenderer(RenderConfig(height=96,width=144,profile='photo',surface_mode='fold',
                                             antialias_samples=1,surface_resolution=48,shadow_samples=0))
        signal = torch.zeros(1,12,300)
        fixed = {'quarter':0,'angle':.18,'zoom':.76,'surface_strength':.7,
                 'perspective_x':.15,'perspective_y':-.1,'occlusion_active':0.,
                 'pen_active':1.,'pen_x':.3,'pen_y':.4,'pen_phase':.5,
                 'artifact_present':[1.,0.,0.,0.],'artifact_x':.23,'artifact_y':.25,
                 'artifact_font':0,'artifact_text':0,'artifact_size':8.,'artifact_angle':.2}
        canonical = renderer(signal,100,parameters=fixed,photo=False,stage='all',
                             generator=torch.Generator().manual_seed(41))
        capture = renderer(signal,100,parameters=fixed,stage='all',return_scene=True,return_field=True,
                           generator=torch.Generator().manual_seed(41))
        expected = F.grid_sample(canonical.targets['valid'],capture.inverse_field,align_corners=False,padding_mode='border')
        expected *= capture.scene['paper_mask']
        torch.testing.assert_close(capture.targets['valid'],expected,rtol=0,atol=1e-7)
        self.assertGreater(float(((capture.scene['paper_mask']-capture.targets['valid'])>.1).sum()),10.)

    def test_absent_paper_marks_do_not_square_fractional_silhouette_coverage(self):
        renderer = ECGRenderer(RenderConfig(height=80,width=120,profile='photo',surface_mode='flat',
                                             antialias_samples=1,surface_resolution=48,shadow_samples=0))
        result = renderer(torch.zeros(1,12,300),100,return_scene=True,stage='all',
                          parameters={'quarter':0,'angle':.13,'zoom':.73,'pen_active':0.,
                                      'artifact_present':0.,'occlusion_active':0.,'tear_active':0.},
                          generator=torch.Generator().manual_seed(44))
        torch.testing.assert_close(result.targets['valid'],result.scene['paper_mask'],rtol=0,atol=1e-7)
        fractional = (result.targets['valid']>0)&(result.targets['valid']<1)
        self.assertTrue(fractional.any())

    def test_optical_overrides_reject_unsupported_psf_and_nonphysical_values(self):
        renderer = ECGRenderer(RenderConfig(height=64,width=96,profile='clean',antialias_samples=1))
        for key,value in [('blur',1.2),('blur',-.1),('motion',3.3),('motion',-.1),
                          ('read_noise',-.01),('shot_noise',-.01),('exposure',-.1),
                          ('gamma',0.),('white_balance',-.1)]:
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):
                renderer(torch.zeros(1,12,100),100,parameters={key:value},photo=False)

    def test_scene_changes_do_not_move_lead_targets_or_source_signals(self):
        r = ECGRenderer(RenderConfig(height=80,width=120,profile='photo',surface_mode='fold',
                                     antialias_samples=1,surface_resolution=48,shadow_samples=0))
        signal = torch.sin(torch.arange(600)[None,None]/9).expand(1,12,-1)*.1
        fixed = {'quarter':0,'angle':.03,'zoom':.8,'surface_strength':.6,
                 'artifact_present':0.,'pen_active':0.,'occlusion_active':0.,
                 'flash_strength':0.,'sensor_active':0.,'jpeg_active':0.}
        a = r(signal,100,stage='all',return_field=True,return_instances=True,
              parameters={**fixed,'light_energy':[.5,.1,.02]},generator=torch.Generator().manual_seed(77))
        b = r(signal,100,stage='all',return_field=True,return_instances=True,
              parameters={**fixed,'light_energy':[.1,.7,.04]},generator=torch.Generator().manual_seed(77))
        torch.testing.assert_close(a.inverse_field,b.inverse_field,rtol=0,atol=0)
        self.assertTrue(torch.equal(a.reference,signal))
        for key in a.targets:
            torch.testing.assert_close(a.targets[key],b.targets[key],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
