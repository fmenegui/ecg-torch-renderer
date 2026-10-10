"""Public module/CLI contracts, independent-network gradients and replay."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu.gpu_module import ECGImageLayer
    from ecg_gpu.gpu_renderer import ECGRenderer, RenderConfig
    from ecg_gpu.gpu_parameters import Uniform, Choice, Bernoulli, parameters_from_json
    from ecg_gpu.gpu_cli import load_signals, render_file
    from ecg_gpu.schema_v05 import LEADS


@unittest.skipUnless(TORCH, 'Training extra required')
class ImageLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = RenderConfig(height=64, width=96, profile='photo', surface_resolution=32,
                                  shadow_samples=0, antialias_samples=1)
        cls.signals = torch.sin(torch.arange(500)/13)[None,None].expand(2,12,-1)*.12

    def test_nn_sequential_trains_image_network(self):
        layer = ECGImageLayer(fs=100, config=self.config)
        cnn = torch.nn.Sequential(torch.nn.Conv2d(3,4,3,stride=4), torch.nn.ReLU(),
                                  torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4,2))
        network = torch.nn.Sequential(layer, cnn)
        source = self.signals.clone().requires_grad_()
        output = network(source)
        self.assertEqual(tuple(output.shape), (2,2))
        output.square().mean().backward()
        self.assertIsNone(source.grad)
        self.assertFalse(list(layer.parameters()))
        self.assertTrue(any(value.grad is not None and value.grad.abs().sum()>0 for value in cnn.parameters()))
        self.assertTrue(all(value.grad is None or torch.isfinite(value.grad).all() for value in cnn.parameters()))

    def test_image_only_matches_paired_mode_without_retaining_targets(self):
        renderer = ECGRenderer(self.config)
        a = renderer(self.signals,100,stage='image',generator=torch.Generator().manual_seed(81))
        b = renderer(self.signals,100,stage='all',generator=torch.Generator().manual_seed(81))
        self.assertTrue(torch.equal(a.image,b.image))
        self.assertEqual(a.targets,{})
        self.assertIsNone(a.metadata['unit_text_visible'])
        layer = ECGImageLayer(fs=100,renderer=renderer)
        batch = layer.render_batch(self.signals,stage='all',return_field=True,return_instances=True)
        self.assertIs(batch.reference,self.signals)
        self.assertIn('instances',batch.targets)
        self.assertEqual(tuple(batch.inverse_field.shape),(2,64,96,2))

    def test_train_changes_eval_replays_and_fixed_mode_replays(self):
        layer = ECGImageLayer(fs=100,config=self.config,seed=54)
        self.assertFalse(torch.equal(layer(self.signals),layer(self.signals)))
        layer.eval()
        self.assertTrue(torch.equal(layer(self.signals),layer(self.signals)))
        fixed = ECGImageLayer(fs=100,config=self.config,randomize=False,seed=54)
        self.assertTrue(torch.equal(fixed(self.signals),fixed(self.signals)))
        explicit = torch.Generator().manual_seed(40)
        a = layer(self.signals,generator=explicit)
        b = layer(self.signals,generator=torch.Generator().manual_seed(40))
        self.assertTrue(torch.equal(a,b))

    def test_checkpoint_replays_next_batch_including_distributions(self):
        parameters = {'layout':Choice(['3x4+r2','6x2']), 'angle':Uniform(-.12,.12),
                      'header_present':Bernoulli(.5), 'rhythm_leads':['II','V1','V5']}
        layer = ECGImageLayer(fs=100,config=self.config,parameters=parameters,seed=27)
        layer(self.signals)
        stream = io.BytesIO()
        torch.save(layer.state_dict(),stream)
        expected = layer(self.signals)
        replacement = ECGImageLayer(fs=100,config=self.config,parameters=parameters,seed=999)
        stream.seek(0)
        replacement.load_state_dict(torch.load(stream,weights_only=True))
        self.assertTrue(torch.equal(expected,replacement(self.signals)))
        changed = ECGImageLayer(fs=125,config=self.config,parameters=parameters)
        with self.assertRaisesRegex(ValueError,'configuration/assets changed'):
            changed.load_state_dict(layer.state_dict())

    def test_fixed_and_random_parameters_are_batched_buffers(self):
        layer = ECGImageLayer(fs=100,config=self.config,parameters={
            'layout':Choice(['12x1','3x4+r3']), 'angle':Uniform(-.2,.2),
            'lead_present':Bernoulli(1.), 'gain_index':1, 'background_type':'wood'})
        batch = layer.render_batch(self.signals,stage='image')
        p = batch.metadata['parameters']
        self.assertTrue(((p['layout']==2)|(p['layout']==6)).all())
        self.assertTrue((p['angle']>=-.2).all() and (p['angle']<=.2).all())
        self.assertTrue((p['lead_present']==1).all())
        self.assertTrue((p['gain_index']==1).all())
        self.assertIn('fixed_gain_index',dict(layer.named_buffers()))
        overridden = layer.render_batch(self.signals,stage='image',parameters={'layout':'6x2'})
        self.assertTrue((overridden.metadata['parameters']['layout']==1).all())
        with self.assertRaisesRegex(ValueError,'continuous'):
            ECGImageLayer(config=self.config,parameters={'layout':Uniform(0,3)})
        with self.assertRaisesRegex(ValueError,'probability'):
            ECGImageLayer(config=self.config,parameters={'header_present':Bernoulli(1.2)})
        with self.assertRaisesRegex(ValueError,'Unknown rendering parameter'):
            ECGImageLayer(config=self.config,parameters={'typo':1})

    def test_module_hot_path_stays_in_torch(self):
        layer = ECGImageLayer(fs=100,config=self.config,parameters={'angle':Uniform(-.1,.1)})
        with mock.patch.object(torch.Tensor,'cpu',side_effect=AssertionError('Host copy')), \
             mock.patch.object(torch.Tensor,'numpy',side_effect=AssertionError('NumPy')), \
             mock.patch.object(torch.Tensor,'item',side_effect=AssertionError('Sync')), \
             mock.patch('PIL.ImageDraw.ImageDraw.text',side_effect=AssertionError('CPU text')):
            image = layer(self.signals)
        self.assertEqual(tuple(image.shape),(2,3,64,96))

    def test_autocast_keeps_render_pixels_float32_and_trains_cnn(self):
        layer=ECGImageLayer(fs=100,config=self.config,randomize=False,seed=712)
        expected=layer(self.signals)
        cnn=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,stride=4),torch.nn.AdaptiveAvgPool2d(1),
                                torch.nn.Flatten(),torch.nn.Linear(4,2))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            observed=layer(self.signals)
            output=cnn(observed)
            loss=torch.nn.functional.cross_entropy(output,torch.tensor([0,1]))
        loss.backward()
        self.assertEqual(observed.dtype,torch.float32)
        self.assertTrue(torch.equal(expected,observed))
        self.assertEqual(output.dtype,torch.bfloat16)
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all() for value in cnn.parameters()))
        with self.assertRaisesRegex(ValueError,'buffers float32'):
            layer.half()(self.signals)

    def test_cached_rng_and_rate_preserve_explicit_replay(self):
        layer=ECGImageLayer(fs=100,config=self.config,seed=333)
        first=layer.render_batch(self.signals,stage='image')
        cached=layer._rng_cache[layer.renderer.device]
        second=layer.render_batch(self.signals,stage='image')
        self.assertIs(cached,layer._rng_cache[layer.renderer.device])
        self.assertEqual(first.metadata['sample_rate_hz'].data_ptr(),layer._sampling_rate.data_ptr())
        self.assertEqual(second.metadata['sample_rate_hz'].data_ptr(),layer._sampling_rate.data_ptr())
        expected=layer.renderer(self.signals,100,stage='image',generator=torch.Generator().manual_seed(333+1000003))
        self.assertTrue(torch.equal(second.image,expected.image))
        self.assertNotIn('_sampling_rate',layer.state_dict())

    def test_compilation_setting_is_guarded_and_legacy_eager_state_loads(self):
        eager=ECGImageLayer(fs=100,config=self.config)
        state=eager.state_dict()
        compiled=ECGImageLayer(fs=100,config=self.config,compile_raster=True)
        with self.assertRaisesRegex(ValueError,'Rasterizer backend changed'):
            compiled.load_state_dict(state)
        state['_extra_state'].pop('compile_raster')
        replacement=ECGImageLayer(fs=100,config=self.config)
        replacement.load_state_dict(state)
        self.assertTrue(torch.equal(eager(self.signals),replacement(self.signals)))


@unittest.skipUnless(TORCH, 'Training extra required')
class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_input_reorders_leads_converts_voltage_and_keeps_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'signal.npz'
            names = LEADS[::-1]
            source = np.stack([np.full(200,LEADS.index(name)+1,dtype=np.float32) for name in names])
            np.savez(path,signals=source*1000,fs=100.,lead_names=np.asarray(names,dtype='S'),
                     units=np.asarray(b'uV'),lengths=np.array([190]))
            signals,fs,lengths = load_signals(path)
            np.testing.assert_allclose(signals[0,:,0],np.arange(1,13),rtol=1e-6)
            self.assertEqual(float(fs),100.)
            self.assertEqual(lengths.tolist(),[190])
            np.savez(path,signals=source,fs=100.,lead_names=np.asarray(['I']*12))
            with self.assertRaisesRegex(ValueError,'lead names'):
                load_signals(path)
            np.save(path.with_suffix('.npy'),source)
            with self.assertRaisesRegex(ValueError,'Sampling rate'):
                load_signals(path.with_suffix('.npy'))

    def test_wfdb_channel_units_are_not_scaled_twice(self):
        record = types.SimpleNamespace(p_signal=np.full((20,12),1000.,dtype=np.float32),
                                      sig_name=LEADS,fs=500.,units=['uV']*12)
        with mock.patch.dict(sys.modules,{'wfdb':types.SimpleNamespace(rdrecord=lambda _:record)}):
            signals,_,_ = load_signals('synthetic.hea')
            np.testing.assert_allclose(signals,1.)
            with self.assertRaisesRegex(ValueError,'WFDB units'):
                load_signals('synthetic.hea',units='uV')

    def test_batch_export_metadata_and_seed_reproduction(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder/'signals.npz'
            np.savez(path,signals=np.zeros((2,12,200),np.float32),fs=np.array([100.,125.]))
            config = RenderConfig(height=64,width=96,profile='clean',surface_mode='off',
                                  surface_resolution=32,shadow_samples=0,antialias_samples=1)
            a = render_file(path,folder/'first',config=config,device='cpu',metadata_path=folder/'metadata.json')
            b = render_file(path,folder/'second',config=config,device='cpu')
            self.assertEqual(a['image_shape'],[2,3,64,96])
            self.assertEqual(json.loads((folder/'metadata.json').read_text())['signal_units'],'mV')
            for first,second in zip(a['outputs'],b['outputs']):
                self.assertEqual(Path(first).read_bytes(),Path(second).read_bytes())
                with Image.open(first) as image:
                    self.assertEqual(image.size,(96,64))
            with self.assertRaisesRegex(ValueError,'one input'):
                render_file(path,folder/'single.png',config=config,device='cpu')

    def test_cli_subprocess_with_json_distribution(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder/'signal.npy'
            np.save(path,np.zeros((12,200),np.float32))
            module = render_file.__module__
            completed = subprocess.run([sys.executable,'-m',module,'--input',str(path),'--fs','100',
                '--output',str(folder/'ecg.png'),'--device','cpu','--height','64','--width','96',
                '--surface','off','--surface-resolution','32','--shadow-samples','0','--antialias','1',
                '--parameters',json.dumps({'layout':{'choice':['6x2']},'angle':{'uniform':[0,0]}})],
                check=True,capture_output=True,text=True,timeout=30)
            report = json.loads(completed.stdout)
            self.assertEqual(report['image_shape'],[1,3,64,96])
            self.assertTrue((folder/'ecg.png').is_file())
            with self.assertRaisesRegex(ValueError,'distribution'):
                parameters_from_json({'angle':{'wrong':[0,1]}})
