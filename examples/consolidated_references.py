"""Matched frozen reference checks and exploratory physical-image transfer."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.field_rectifier import GridFieldNet
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_benchmark import BenchmarkConfig, load_external
from ecg_gpu.grid_evaluation import match_grid_points, summarize_grid_matches
from examples.consolidated_grid_study import predict, progress
from examples.digitization_npj_profiles import run
from examples.run_paper_experiment import run_kaggle


def main(root, source, assets, external):
    results=[]
    for i,patient in enumerate(json.loads((root/'signal-inputs.json').read_text())):
        inputs=root/patient['original']
        out=root/'reference-results'/patient['source_id']
        enabled=run(inputs,out/'npj-dewarping-enabled',source,device='cuda',arms=('configured',),dewarping=True)
        kaggle=run_kaggle(inputs,out/'kaggle',assets,device='cuda')
        results.append({'source_id':patient['source_id'],'group_id':patient['group_id'],
                        'npj_dewarping_enabled':enabled['cases'],'kaggle':kaggle['cases']})
        _write_json(root/'reference-results.json',results)
        progress(root,'frozen-reference-comparison',completed=i+1,total=8)
    if external is not None:
        samples=load_external(external,BenchmarkConfig(width=768,height=512))
        records=[]
        for trial in json.loads((root/'trials.json').read_text()):
            state=torch.load(root/trial['checkpoint'],map_location='cpu',weights_only=True)
            model=GridFieldNet(contrast=trial['variant']!='raw-point').cuda().eval()
            model.load_state_dict(state['state_dict'])
            for sample in samples:
                xy, _scores, _, _=predict(model,sample,threshold=trial['validation']['threshold'])
                record={'name':trial['name'],'id':sample['key'],'source_id':sample['source_id'],
                        'group_id':sample['group_id'],'image_sha256':sample['metadata']['sha256'],
                        'predictions':len(xy),'metrics':None}
                if sample['xy'].shape[1]:
                    region=(np.asarray(sample['metadata']['annotation_region']).reshape(2,2)+.5)*sample['scale']-.5
                    inside=((xy>=region[0])&(xy<=region[1])).all(-1)
                    record.update(metrics=summarize_grid_matches([match_grid_points(xy[inside],sample['xy'][0].numpy(),radius=1.5)]),
                                  annotation_method=sample['metadata']['annotation_method'])
                records.append(record)
            del model
            torch.cuda.empty_cache()
        _write_json(root/'external-transfer.json',{'cases':records,'limits':[
            'Exploratory transfer, previously examined physical captures of one synthetic printed ECG',
            'One complete ROI has automatic silver annotations, not expert gold standard',
            'Other two images are unannotated, prediction counts are not accuracy',
            'Matching radius is 1.5 pixels after aspect-preserving resize to 768x512',
            'No adapted source images or overlays are redistributed']})
    progress(root,'all-experiments-completed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','assets'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--external',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    main(a.root,a.source,a.assets,a.external)
