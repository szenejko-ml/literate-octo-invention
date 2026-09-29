"""Fresh-interpreter roundtrip check. Inputs must be trusted local files."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from doctoral_models import predict_bundle

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact',required=True); parser.add_argument('--sample',required=True)
    parser.add_argument('--report',required=True); parser.add_argument('--allow-cloud',action='store_true')
    a=parser.parse_args()
    bundle=joblib.load(a.artifact); sample=joblib.load(a.sample)
    result=predict_bundle(bundle,sample['X'],allow_cloud=a.allow_cloud)
    diff=float(np.max(np.abs(result.prob_euploid.to_numpy()-sample['prob'])))
    same=bool(np.array_equal(result.zone.to_numpy(),sample['zone']))
    report={'status':'passed' if diff<=1e-12 and same else 'failed',
        'n_checked':len(result),'maximum_absolute_probability_difference':diff,
        'identical_zone_assignments':same,'fresh_python_process':True,
        'requires_cloud':bool(bundle['metadata']['cloud_model'])}
    Path(a.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['status']!='passed': raise RuntimeError(report)
    print('Artifact readback passed')

if __name__=='__main__': main()