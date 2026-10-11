import unittest
from unittest import mock
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ecg_gpu import ECGRenderer, RenderConfig, build_model, task_loss, LayoutSpec, DocumentAssets
from ecg_gpu.gpu_appearance import CameraAppearance
from ecg_gpu.gpu_document import SURFACES, BACKGROUNDS, SHADOWS
from ecg_gpu.schema_v05 import LEADS
TORCH=True

class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.renderer = ECGRenderer(RenderConfig(height=128, width=192, profile='clean', antialias_samples=2))
        cls.fixed = {'layout': 0, 'speed_index': 1, 'gain_index': 1, 'sequential': True,
                     'quarter': 0, 'angle': 0., 'zoom': 1., 'header_present': 1., 'label_present': 1.}

    def render(self, signals, **kwargs):
        return self.renderer(signals, 100, photo=False, generator=torch.Generator().manual_seed(713),
                             parameters=self.fixed, **kwargs)

    def test_physical_gain_time_windows_and_reference_are_preserved(self):
        t = torch.arange(1000) / 100
        signals = (t * .12)[None, None].expand(1, 12, -1).clone()
        original = signals.clone()
        batch = self.render(signals, stage='all')
        self.assertTrue(torch.equal(original, batch.reference))
        meta = batch.metadata
        self.assertAlmostEqual(float(meta['gain'][0]), 10.)
        self.assertAlmostEqual(float(meta['speed'][0]), 25.)
        for lead in [0, 3, 6, 9]:
            x = int((meta['x0'][0, lead] * .6 + meta['x1'][0, lead] * .4).round())
            time = meta['start_s'][0, lead] + (x - meta['x0'][0, lead]) / (meta['speed'][0] * meta['ppm'][0])
            expected = meta['baseline'][0, lead] - time * .12 * meta['gain'][0] * meta['ppm'][0]
            actual = batch.targets['traces'][0, lead, :, x].argmax()
            self.assertLessEqual(abs(float(actual - expected)), 1.5)
        self.assertGreater(float(meta['start_s'][0, 3]), 0.)

    def test_raster_matches_frozen_v021_all_30_layouts(self):
        import json
        from ecg_gpu.compare import load_legacy
        # Compare the immutable implementation bit for bit in this runtime.
        # Stored float32 results can differ across PyTorch/platform releases.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        legacy = load_legacy(Path(__file__).resolve().parents[1]/'references/ecg-gpu-generator-v021.zip', directory.name)
        with np.load(Path(__file__).parent/'fixtures/raster-v021.npz',allow_pickle=False) as fixture:
            cases=json.loads(str(fixture['metadata_json']))['cases']
            for case in cases:
                renderer=ECGRenderer(RenderConfig(**case['configuration']))
                prefix=f"case{case['case']}/"
                signals,rates,lengths=[torch.from_numpy(fixture[prefix+name]) for name in ['signals','rates','lengths']]
                p={key[len(prefix+'p/'):]:torch.from_numpy(fixture[key]) for key in fixture.files if key.startswith(prefix+'p/')}
                instances=renderer._instances(p,((lengths-1)/rates).clamp(max=renderer.config.max_duration_s))
                with torch.no_grad():
                    ink,pulse,targets=renderer._curves(signals,rates,lengths,instances,p,'all',True)
                    original = legacy.ECGRenderer(legacy.RenderConfig(**case['configuration']))
                    old_ink,old_pulse,old_targets=original._curves(signals,rates,lengths,instances,p,'all',True)
                old_values={'ink':old_ink,'pulse':old_pulse,**old_targets}
                for name,value in {'ink':ink,'pulse':pulse,**targets}.items():
                    context=f"Layout group {case['layouts']}, {name}: "
                    torch.testing.assert_close(value,old_values[name],rtol=0,atol=0,
                                                msg=lambda detail:context+detail)
                    torch.testing.assert_close(value,torch.from_numpy(fixture[prefix+name]),rtol=0,atol=2e-6,
                                                msg=lambda detail:context+detail)

    def test_compiled_raster_preserves_frozen_masks_and_image_union(self):
        import json
        with np.load(Path(__file__).parent/'fixtures/raster-v021.npz',allow_pickle=False) as fixture:
            cases=json.loads(str(fixture['metadata_json']))['cases']
            for case in [cases[0],cases[3]]:
                renderer=ECGRenderer(RenderConfig(**case['configuration']),compile_raster=True)
                prefix=f"case{case['case']}/"
                signals,rates,lengths=[torch.from_numpy(fixture[prefix+name]) for name in ['signals','rates','lengths']]
                p={key[len(prefix+'p/'):]:torch.from_numpy(fixture[key]) for key in fixture.files if key.startswith(prefix+'p/')}
                instances=renderer._instances(p,((lengths-1)/rates).clamp(max=renderer.config.max_duration_s))
                with torch.no_grad():
                    ink,pulse,targets=renderer._curves(signals,rates,lengths,instances,p,'all',True)
                    union,union_pulse,_=renderer._curves(signals,rates,lengths,instances,p,'image')
                for name,value in {'ink':ink,'pulse':pulse,**targets}.items():
                    torch.testing.assert_close(value,torch.from_numpy(fixture[prefix+name]),rtol=0,atol=2e-6)
                torch.testing.assert_close(union,ink,rtol=0,atol=2e-6)
                torch.testing.assert_close(union_pulse,pulse,rtol=0,atol=2e-6)

    def test_original_subpixel_extremum_survives_rasterization(self):
        signals = torch.zeros(1, 12, 5000)
        signals[0, 0, 1234] = 1.5
        parameters = {**self.fixed, 'layout': 2}
        batch = self.renderer(signals, 500, photo=False, parameters=parameters,
                              generator=torch.Generator().manual_seed(713), stage='trace')
        meta = batch.metadata
        x = int((meta['x0'][0, 0] + 1234 / 500 * meta['speed'][0] * meta['ppm'][0]).round())
        y = int((meta['baseline'][0, 0] - 1.5 * meta['gain'][0] * meta['ppm'][0]).round())
        self.assertGreater(float(batch.targets['traces'][0, 0, max(0,y-1):y+2, x].max()), .3)
        self.assertEqual(float(batch.reference[0, 0, 1234]), 1.5)

    def test_crossings_keep_both_lead_channels(self):
        signals = torch.zeros(1, 12, 1000)
        parameters = {**self.fixed, 'layout': 2}
        initial = self.renderer(signals, 100, photo=False, parameters=parameters,
                                generator=torch.Generator().manual_seed(713), stage='trace')
        meta = initial.metadata
        signals[0, 0] = -(meta['baseline'][0, 1] - meta['baseline'][0, 0]) / (meta['gain'][0] * meta['ppm'][0])
        crossed = self.renderer(signals, 100, photo=False, parameters=parameters,
                                generator=torch.Generator().manual_seed(713), stage='trace')
        self.assertTrue(((crossed.targets['traces'][0, 0] > .4) & (crossed.targets['traces'][0, 1] > .4)).any())

    def test_photo_targets_use_identical_inverse_field(self):
        signals = torch.sin(torch.arange(1000)[None, None] / 35).expand(1, 12, -1) * .25
        canonical = self.render(signals, stage='all')
        parameters = {**self.fixed, 'quarter': 1, 'angle': -torch.pi/2,
                      'translate_x': .07, 'perspective_x': .04, 'curl_y': .012}
        photo = self.renderer(signals, 100, parameters=parameters, stage='all', return_field=True,
                              generator=torch.Generator().manual_seed(713))
        for name in ['traces', 'baselines', 'labels', 'grid', 'vertices']:
            expected = F.grid_sample(canonical.targets[name], photo.inverse_field, align_corners=False)
            torch.testing.assert_close(photo.targets[name], expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(int(photo.targets['orientation'][0]), 1)

    def test_missing_units_and_sticker_visibility_are_explicit(self):
        parameters = {**self.fixed, 'header_present': 0., 'occlusion_active': 1.,
                      'occlusion_x': .2, 'occlusion_y': .3, 'occlusion_w': .1, 'occlusion_h': .1}
        batch = self.renderer(torch.zeros(1, 12, 1000), 100, parameters=parameters, photo=False,
                              generator=torch.Generator().manual_seed(4), stage='all')
        self.assertEqual(int(batch.targets['speed'][0]), 0)
        self.assertEqual(int(batch.targets['gain'][0]), 0)
        self.assertEqual(float(batch.targets['valid'][0, 0, 43, 48]), 0)
        self.assertAlmostEqual(float(batch.metadata['speed'][0]), 25.)

    def test_layouts_rhythm_and_simultaneous_windows(self):
        for layout in range(5):
            batch = self.renderer(torch.zeros(2, 12, 1000), torch.tensor([100., 125.]), photo=False,
                                  parameters={**self.fixed, 'layout': layout, 'sequential': False},
                                  lengths=torch.tensor([1000, 800]), stage='metadata',
                                  generator=torch.Generator().manual_seed(8))
            self.assertTrue((batch.metadata['start_s'] == 0).all())
            self.assertTrue((batch.metadata['active'].sum(1) == (13 if layout == 4 else 12)).all())
            self.assertTrue((batch.targets['layout'] == layout + 1).all())

    def test_hot_path_avoids_cpu_image_operations_and_tensor_copies(self):
        with mock.patch('PIL.Image.open', side_effect=AssertionError('Image decode in hot path')), \
             mock.patch('PIL.ImageDraw.ImageDraw.text', side_effect=AssertionError('Text rasterization in hot path')), \
             mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Device to host transfer')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy in hot path')):
            batch = self.render(torch.zeros(1, 12, 1000), stage='geometry')
        self.assertEqual(tuple(batch.image.shape), (1, 3, 128, 192))

    def test_invalid_signals_are_rejected(self):
        signals = torch.zeros(1, 12, 100)
        signals[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            self.render(signals)
        with self.assertRaisesRegex(RuntimeError, 'Positive'):
            self.renderer(torch.zeros(1, 12, 100), 0)
        with self.assertRaisesRegex(RuntimeError, 'Invalid layout'):
            self.renderer(torch.zeros(1, 12, 100), 100, parameters={'zoom':0.})

    def test_unprinted_leads_and_absent_grid_have_no_supervision_ink(self):
        batch = self.renderer(torch.zeros(1, 12, 500), 100, photo=False, stage='all',
                              parameters={**self.fixed, 'lead_present':0., 'grid_present':0.},
                              generator=torch.Generator().manual_seed(9))
        for name in ['traces', 'baselines', 'labels', 'grid', 'vertices']:
            self.assertEqual(float(batch.targets[name].sum()), 0.)
        self.assertFalse(batch.metadata['active'].any())

    def test_renderer_reproducibility_and_training_backward(self):
        renderer = ECGRenderer(RenderConfig(height=96, width=128, profile='photo', severity=.8))
        signals = torch.randn(2, 12, 400, generator=torch.Generator().manual_seed(39)) * .12
        for stage in ['geometry', 'trace', 'metadata']:
            a = renderer(signals, 100, stage=stage, photo=stage != 'trace', generator=torch.Generator().manual_seed(17))
            b = renderer(signals, 100, stage=stage, photo=stage != 'trace', generator=torch.Generator().manual_seed(17))
            self.assertTrue(torch.equal(a.image, b.image))
            model = build_model(stage, width=4, depth=2)
            loss = task_loss(stage, model(F.pad(a.image, (0,0,0,32), value=1)),
                             {k:F.pad(v,(0,0,0,32)) if v.ndim == 4 else v for k,v in a.targets.items()})
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_texture_banks_are_sampled_without_changing_reference(self):
        renderer = ECGRenderer(RenderConfig(height=64, width=96, profile='photo'),
                               paper_textures=torch.full((2, 3, 16, 24), .98),
                               backgrounds=torch.full((3, 3, 24, 32), .35))
        signals = torch.zeros(2, 12, 400)
        batch = renderer(signals, 100, stage='geometry', generator=torch.Generator().manual_seed(21))
        self.assertTrue((batch.metadata['paper_texture_id'] >= 0).all())
        self.assertTrue((batch.metadata['background_id'] >= 0).all())
        self.assertTrue(torch.equal(batch.reference, signals))

    @unittest.skipUnless(TORCH and torch.cuda.is_available(), 'CUDA unavailable in this environment')
    def test_cuda_entire_batch_remains_on_device(self):
        renderer = ECGRenderer(RenderConfig(height=96, width=128),compile_raster=True).cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            batch = renderer(torch.zeros(2, 12, 1000, device='cuda'), torch.full((2,),100.,device='cuda'),
                             generator=torch.Generator(device='cuda').manual_seed(19), stage='all', return_field=True)
        self.assertEqual(batch.image.dtype,torch.float32)
        self.assertTrue(batch.image.is_cuda and batch.reference.is_cuda and batch.inverse_field.is_cuda)
        self.assertTrue(all(value.is_cuda for value in batch.targets.values()))

class CameraTests(unittest.TestCase):
    def test_dct_roundtrip_and_quality_including_odd_sizes(self):
        camera = CameraAppearance()
        constant = torch.full((1, 3, 17, 19), .5)
        decoded = camera.jpeg(constant, torch.tensor([100.]))
        self.assertEqual(decoded.shape, constant.shape)
        self.assertLess(float((decoded - constant).abs().max()), .006)
        texture = torch.rand(2, 3, 33, 35, generator=torch.Generator().manual_seed(12))
        low = camera.jpeg(texture, torch.full((2,), 15.))
        high = camera.jpeg(texture, torch.full((2,), 95.))
        self.assertGreater(float((low - texture).square().mean()), float((high - texture).square().mean()))

class DocumentSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.renderer = ECGRenderer(RenderConfig(height=128,width=192,profile='photo',
                                                severity=.7,antialias_samples=2,shadow_samples=4))
        cls.fixed = {'layout':'3x4+r3','quarter':0,'angle':.045,'zoom':.85,
                     'perspective_x':.12,'perspective_y':-.06,'radial':0.,
                     'curl_x':0.,'curl_y':0.,'crease_warp':0.,
                     'header_present':1.,'label_present':1.,'lead_present':1.,
                     'grid_present':1.,'occlusion_active':0.,'pen_active':0.,
                     'artifact_present':0.,'tear_active':0.}

    def render(self, parameters=None, **kwargs):
        t=torch.arange(1000)/100
        signals=(.25*torch.sin(t*6))[None,None].expand(1,12,-1).clone()
        return self.renderer(signals,100,parameters={**self.fixed,**(parameters or {})},
                             generator=torch.Generator().manual_seed(751),**kwargs)

    def test_all_layouts_keep_their_lead_rows_time_scales_and_class(self):
        renderer=ECGRenderer(RenderConfig(height=96,width=128,profile='clean',antialias_samples=1))
        signals=torch.zeros(1,12,1000)
        for index,spec in enumerate(renderer.layout_specs):
            b=renderer(signals,100,photo=False,stage='metadata',
                       parameters={'layout':spec.name,'lead_present':1.,'artifact_present':0.,'sequential':True},
                       generator=torch.Generator().manual_seed(11))
            self.assertEqual(int(b.metadata['active'].sum()),sum(map(len,spec.rows))+len(spec.rhythm))
            self.assertEqual(int(b.targets['layout'][0]),index+1)
            for row,cells in enumerate(spec.rows):
                for column,token in enumerate(cells):
                    lead=LEADS.index(token.removeprefix('-'))
                    self.assertEqual(int(b.metadata['row'][0,lead]),row)
                    self.assertEqual(int(b.metadata['amplitude_sign'][0,lead]),-1 if token.startswith('-') else 1)
                    duration=float(b.metadata['duration_s'][0])
                    self.assertAlmostEqual(float(b.metadata['start_s'][0,lead]),column*duration/len(cells),places=5)
                    physical_span=(b.metadata['end_s'][0,lead]-b.metadata['start_s'][0,lead])*b.metadata['speed'][0]*b.metadata['ppm'][0]
                    self.assertAlmostEqual(float(b.metadata['x1'][0,lead]-b.metadata['x0'][0,lead]),float(physical_span),places=4)

    def test_custom_uneven_rows_are_unknown_class_with_exact_windows(self):
        spec=LayoutSpec('custom',(('I','II','III'),('V1',),('aVR','V5')),('V6','II'))
        renderer=ECGRenderer(RenderConfig(height=96,width=128,profile='clean',antialias_samples=1),layouts=[spec])
        b=renderer(torch.zeros(1,12,1000),100,photo=False,stage='metadata',
                   parameters={'lead_present':1.,'sequential':True},generator=torch.Generator().manual_seed(1))
        self.assertEqual(int(b.targets['layout'][0]),0)
        self.assertEqual(int(b.metadata['active'].sum()),8)
        self.assertAlmostEqual(float(b.metadata['end_s'][0,6]),9.99,places=4)
        self.assertAlmostEqual(float(b.metadata['end_s'][0,0]),9.99/3,places=4)
        with self.assertRaises(ValueError):
            LayoutSpec('bad',(('II','II'),))

    def test_cabrera_inverts_printed_avr_without_altering_source(self):
        renderer=ECGRenderer(RenderConfig(height=256,width=384,profile='clean',antialias_samples=2))
        signals=torch.zeros(1,12,1000)
        signals[:,3]=.45
        b=renderer(signals,100,photo=False,stage='trace',return_instances=True,
                   parameters={'layout':'3x4-cabrera','gain_index':1,'speed_index':1,
                               'lead_present':1.,'artifact_present':0.,'pen_active':0.,'occlusion_active':0.},
                   generator=torch.Generator().manual_seed(2))
        x=round(float((b.metadata['x0'][0,3]+b.metadata['x1'][0,3])/2))
        y=int(b.targets['instances'][0,3,:,x].argmax())
        expected=float(b.metadata['baseline'][0,3]+.45*b.metadata['gain'][0]*b.metadata['ppm'][0])
        self.assertAlmostEqual(y,expected,delta=1.)
        self.assertTrue(torch.equal(signals,b.reference))

    def test_rhythm_selection_and_instance_boxes_preserve_identity(self):
        b=self.render({'rhythm_leads':['III','V4','aVR']},stage='all',return_instances=True)
        self.assertEqual(b.metadata['lead_id'][0,12:].tolist(),[2,9,3])
        for index in range(15):
            lead=int(b.metadata['lead_id'][0,index])
            self.assertTrue((b.targets['traces'][0,lead]+1e-6>=b.targets['instances'][0,index]).all())
            box=b.metadata['instance_boxes_xyxy'][0,index]
            visible=b.targets['instances'][0,index]*b.targets['valid'][0,0]>.05
            if bool(b.metadata['instance_visible'][0,index]):
                ys,xs=torch.where(visible)
                self.assertGreater(len(xs),0)
                self.assertTrue(((xs>=box[0])&(xs<box[2])&(ys>=box[1])&(ys<box[3])).all())

    def test_every_surface_has_aligned_masks_finite_normals_and_solver_evidence(self):
        for mode in SURFACES:
            canonical=self.render({'surface_mode':mode,'surface_strength':.8},photo=False,stage='all')
            b=self.render({'surface_mode':mode,'surface_strength':.8},stage='all',return_field=True,return_scene=True)
            for name in ['traces','labels','baselines','grid','vertices']:
                expected=F.grid_sample(canonical.targets[name],b.inverse_field,align_corners=False)
                torch.testing.assert_close(b.targets[name],expected,atol=3e-6,rtol=3e-6)
            for value in b.scene.values():
                self.assertTrue(torch.isfinite(value).all())
            valid=b.scene['solver_valid'].bool()
            self.assertLess(float(b.scene['projection_residual'][valid].max()),2/192+1e-6)
            norms=b.scene['normals'].square().sum(1).sqrt()
            torch.testing.assert_close(norms,torch.ones_like(norms),atol=2e-6,rtol=2e-6)
            if mode!='flat':
                self.assertGreater(float(b.scene['height'].max()),0.)

    def test_asset_visibility_and_uv_are_used_for_images_and_supervision(self):
        h,w=128,192
        yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
        field=torch.stack([2*(xx+.5)/w-1,2*(yy+.5)/h-1],-1)[None]
        # A bank can represent an arbitrary visible sheet arrangement, including
        # mirrored UV and hidden regions, without relying on a height solver.
        field[...,0]*=-1
        mask=torch.ones(1,1,h,w)
        mask[...,50:70,70:120]=0
        assets=DocumentAssets(field,mask)
        r=ECGRenderer(RenderConfig(height=h,width=w,profile='clean',surface_mode='off'),document_assets=assets)
        signal=torch.zeros(1,12,1000)
        common={'layout':0,'quarter':0,'angle':0.,'zoom':1.,'asset_active':1.,'asset_index':0,
                'tear_active':0.,'pen_active':0.,'artifact_present':0.,'occlusion_active':0.,'lead_present':1.}
        canonical=r(signal,100,photo=False,parameters=common,stage='all',generator=torch.Generator().manual_seed(1))
        b=r(signal,100,parameters=common,stage='all',return_scene=True,return_field=True,
            generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(b.inverse_field,field,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(b.scene['paper_mask'],mask,atol=3e-6,rtol=3e-6)
        self.assertEqual(int(b.metadata['document_asset_id'][0]),0)
        self.assertEqual(float(b.targets['valid'][0,0,60,90]),0.)
        expected=F.grid_sample(canonical.targets['traces'],field,align_corners=False)
        torch.testing.assert_close(b.targets['traces'],expected,atol=3e-6,rtol=3e-6)
        self.assertEqual(len(r.document.asset_hashes),5)

    def test_asset_import_schema_and_invalid_geometry_are_rejected(self):
        h,w=16,16
        field=torch.zeros(1,h,w,2)
        mask=torch.ones(1,1,h,w)
        with tempfile.TemporaryDirectory() as location:
            path=Path(location)/'assets.npz'
            np.savez(path,schema='ecg-document-assets-v1',inverse_field=field.numpy(),mask=mask.numpy())
            loaded=DocumentAssets.from_npz(path)
            self.assertEqual(tuple(loaded.inverse_field.shape),(1,h,w,2))
            np.savez(path,schema='wrong',inverse_field=field.numpy(),mask=mask.numpy())
            with self.assertRaisesRegex(ValueError,'schema'):
                DocumentAssets.from_npz(path)
        with self.assertRaisesRegex(ValueError,'normals'):
            ECGRenderer(RenderConfig(height=64,width=64),document_assets=DocumentAssets(field,mask,normals=torch.zeros(1,3,h,w)))
        with self.assertRaises(ValueError):
            ECGRenderer(RenderConfig(height=64,width=64),document_assets=DocumentAssets(field,mask*2))

    def test_contact_and_cast_shadows_only_darken_the_background(self):
        b=self.render({'cast_shadow_strength':.8},return_scene=True,stage='geometry')
        paper=b.scene['paper_mask']
        cast=b.scene['cast_shadow']
        # Fractional sheet-edge coverage blends a background fraction. Cast
        # shadows may affect that fraction but never fully covered paper.
        self.assertLessEqual(float(cast[paper>=1-1e-6].max()),1e-6)
        self.assertTrue((cast<=.8*(1-paper)+1e-6).all())
        self.assertGreater(float(cast.max()),.05)

    def test_table_materials_and_object_shadow_families_are_distinct(self):
        p=self.renderer.sample_parameters(1,generator=torch.Generator().manual_seed(1))
        images=[]
        for kind in range(len(BACKGROUNDS)):
            p['background_type']=torch.tensor([kind])
            image=self.renderer.document.background(p,torch.Generator().manual_seed(2))
            self.assertEqual(tuple(image.shape),(1,3,128,192))
            self.assertTrue((image>=0).all() and (image<=1).all())
            images.append(image)
        for a,b in zip(images,images[1:]):
            self.assertGreater(float((a-b).abs().mean()),.01)
        for kind in range(len(SHADOWS)):
            p['object_shadow']=torch.tensor([kind])
            p['object_shadow_strength']=torch.tensor([.6])
            shadow=self.renderer.document.object_shadow(p)
            self.assertTrue(torch.isfinite(shadow).all())
            self.assertEqual(tuple(shadow.shape),(1,1,128,192))
            self.assertGreater(float(shadow.max()),.1) if kind else self.assertEqual(float(shadow.sum()),0.)

    def test_ragged_tear_removes_paper_and_marks_pixels_invalid(self):
        r=ECGRenderer(RenderConfig(height=128,width=192,profile='clean',surface_mode='off'))
        b=r(torch.zeros(1,12,1000),100,stage='geometry',return_scene=True,
            parameters={'quarter':0,'angle':0.,'zoom':1.,'tear_active':1.,'tear_side':0,
                        'tear_position':0.,'tear_size':.14,'artifact_present':0.,'pen_active':0.,'occlusion_active':0.},
            generator=torch.Generator().manual_seed(5))
        self.assertEqual(float(b.scene['paper_mask'][0,0,64,191]),0.)
        self.assertEqual(float(b.targets['valid'][0,0,64,191]),0.)
        self.assertEqual(float(b.scene['paper_mask'][0,0,64,120]),1.)

    def test_annotations_invalidate_covered_ink_without_changing_signal(self):
        b=self.render({'artifact_present':1.,'artifact_x':.2,'artifact_y':.3,'artifact_angle':.25},stage='geometry')
        a=self.render({'artifact_present':0.},stage='geometry')
        self.assertGreater(float((a.image-b.image).abs().sum()),0.)
        self.assertGreater(float((a.targets['valid']-b.targets['valid']).clamp_min(0).sum()),0.)
        self.assertTrue(torch.equal(a.reference,b.reference))

    def test_photo_scene_hot_path_stays_in_torch_and_reproduces_seed(self):
        a=self.render({'surface_mode':'crumpled','object_shadow':'hand'},return_scene=True)
        with mock.patch.object(torch.Tensor,'cpu',side_effect=AssertionError('CPU transfer')), \
             mock.patch.object(torch.Tensor,'numpy',side_effect=AssertionError('NumPy')), \
             mock.patch.object(torch.Tensor,'item',side_effect=AssertionError('Scalar device sync')), \
             mock.patch('PIL.Image.open',side_effect=AssertionError('Decode')):
            b=self.render({'surface_mode':'crumpled','object_shadow':'hand'},return_scene=True)
        self.assertTrue(torch.equal(a.image,b.image))
        self.assertTrue(torch.equal(a.scene['normals'],b.scene['normals']))

    def test_grid_styles_pulses_and_back_print_do_not_change_signal_targets(self):
        b=self.render({'grid_style':0,'pulse_present':1.,'bleed_strength':0.,'coffee_strength':0.},photo=False,stage='all')
        c=self.render({'grid_style':1,'pulse_present':0.,'bleed_strength':.05,'coffee_strength':.18},photo=False,stage='all')
        torch.testing.assert_close(b.targets['traces'],c.targets['traces'])
        torch.testing.assert_close(b.targets['baselines'],c.targets['baselines'])
        self.assertTrue(torch.equal(b.reference,c.reference))
        self.assertEqual(float(c.targets['vertices'][:,1].sum()),0.)
        self.assertGreater(float((b.image-c.image).abs().sum()),0.)
        for style in [2,3]:
            d=self.render({'grid_style':style},photo=False,stage='geometry')
            self.assertTrue(torch.isfinite(d.image).all())
            self.assertGreater(float(d.targets['grid'].sum()),0.)
