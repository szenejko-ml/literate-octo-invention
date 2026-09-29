#!/usr/bin/env python
"""All-year nested reanalysis with mandatory joblib model persistence.

No legacy checkpoints are accepted. No full-cohort fit is used as validation.
Run with --audit-only first; see README_PL.md for the Windows commands.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from doctoral_models import (VERSION, Protocol, ALL_MODELS, LOCAL_MODELS, RAW_PREDICTORS,
    BINARY, CONTROLS, CST, CSTPLUS, NT_EQUATION, derive_nt, prepare_features, override_mask,
    feature_names, model_spec, positive_probability, fit_calibrator, calibrate,
    select_thresholds, assign_zones, predict_bundle, versions, RemoteTabPFN)

HERE=Path(__file__).resolve().parent
COHORTS=('all_years_shared','all_years_predictor_complete','2019_shared')


def json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,Path): return str(value)
    raise TypeError(f'Not JSON-serializable: {type(value)}')


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=json_default,allow_nan=False).encode()).hexdigest()


def utc(): return datetime.now(timezone.utc).isoformat()


def atomic(path, writer):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(fd)
    try:
        writer(Path(tmp))
        if not Path(tmp).stat().st_size: raise IOError(f'Empty output: {path}')
        with open(tmp,'rb+') as f: os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_json(obj,path):
    atomic(path,lambda p:p.write_text(json.dumps(obj,indent=2,default=json_default,
                                               ensure_ascii=False,allow_nan=False),encoding='utf-8'))


def write_joblib(obj,path):
    atomic(path,lambda p:joblib.dump(obj,p,compress=3,protocol=5))
    # Integrity sidecar: detect incomplete or corrupted stage saves before load.
    checksum=sha_file(path)
    atomic(str(path)+'.sha256',lambda p:p.write_text(checksum,encoding='ascii'))


def write_csv(df,path):
    # Explicit compression because the temporary path ends in .tmp.
    compression='gzip' if str(path).endswith('.gz') else None
    atomic(path,lambda p:df.to_csv(p,index=False,compression=compression))


def cohort_data(input_path,sheet,group_column=None):
    original=pd.read_excel(input_path,sheet_name=sheet,engine='openpyxl')
    required=['lp','rok','euploidia']+RAW_PREDICTORS+CONTROLS+CST+CSTPLUS
    missing=sorted(set(required)-set(original.columns))
    if missing: raise ValueError(f'Source workbook missing required columns: {missing}')
    numeric=list(dict.fromkeys(required))
    data=original[numeric].apply(pd.to_numeric,errors='coerce')
    nonnumeric={c:int((original[c].notna()&data[c].isna()).sum()) for c in numeric}
    for c in ['lp','rok']:
        if data[c].isna().any() or not np.equal(data[c],np.floor(data[c])).all():
            raise ValueError(f'{c} must be complete integer-valued identifiers')
        data[c]=data[c].astype('int64')
    if data.duplicated(['rok','lp']).any():
        raise ValueError('Duplicate rok+lp. Resolve at pregnancy level before CV.')
    data['record_id']=data['rok'].astype(str)+'_'+data['lp'].astype(str)
    if group_column:
        if group_column not in original or original[group_column].isna().any():
            raise ValueError('Patient grouping column must exist and be complete')
        data['_group']=original[group_column].astype(str).map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    else:
        data['_group']=data['record_id']
    for c,allowed in [('euploidia',[0,1]),('DV',[0,1,2])]+[(v,[0,1]) for v in BINARY+CONTROLS]:
        invalid=data[c].notna()&~data[c].isin(allowed)
        if invalid.any(): raise ValueError(f'{c}: {int(invalid.sum())} invalid coded values')
    # Non-positive denominators are not silently imputed or converted to low risk.
    for c in CST+CSTPLUS:
        if (data[c].dropna()<=0).any(): raise ValueError(f'{c}: invalid non-positive denominator')
    for c in ['NuchalT','PAPPA','fbHCG','wiek','FHR']:
        if (data[c].dropna()<=0).any(): raise ValueError(f'{c}: invalid non-positive value')
    data['target']=data['euploidia']
    crl_ok=data['CRL'].between(45,84)
    needed=RAW_PREDICTORS+CONTROLS+['target']
    complete=np.isfinite(data[needed].to_numpy(float)).all(axis=1)
    comparator_complete=np.isfinite(data[CST+CSTPLUS].to_numpy(float)).all(axis=1)
    broad=crl_ok&complete
    masks={'all_years_shared':broad&comparator_complete,
           'all_years_predictor_complete':broad,
           '2019_shared':broad&comparator_complete&data['rok'].eq(2019)}
    # Do not export names, free text, dates, or the original workbook.
    # Source outcome flags are for error-description only, never for training.
    flags=[c for c in ['T21','T18','T13','45xo','other'] if c in original]
    for c in flags: data[c]=pd.to_numeric(original[c],errors='coerce')
    data=derive_nt(data)
    data['CST_min_denom']=data[CST].min(axis=1,skipna=False)
    data['CSTplus_min_denom']=data[CSTPLUS].min(axis=1,skipna=False)
    reasons=pd.DataFrame({'record_id':data['record_id'],'rok':data['rok'],
        'CRL_in_range':crl_ok,'predictors_outcome_controls_complete':complete,
        'CST_and_CSTplus_complete':comparator_complete})
    counts=[]
    for name,mask in masks.items():
        frame=data.loc[mask].copy().reset_index(drop=True)
        for year,group in frame.groupby('rok'):
            counts.append({'cohort':name,'year':int(year),'n':len(group),
                'n_euploid':int(group['target'].eq(1).sum()),
                'n_non_euploid':int(group['target'].eq(0).sum())})
    audit={'source_file':Path(input_path).name,'source_sha256':sha_file(input_path),
        'source_n':len(data),'CRL_eligible_n':int(crl_ok.sum()),
        'model_eligible_n':int(broad.sum()),'paired_comparator_complete_n':int(masks['all_years_shared'].sum()),
        'excluded_for_comparator_only':int((broad&~comparator_complete).sum()),
        'excluded_non_euploid_for_comparator_only':int((broad&~comparator_complete&data['target'].eq(0)).sum()),
        'missing_after_CRL':{c:int(data.loc[crl_ok,c].isna().sum()) for c in needed+CST+CSTPLUS},
        'nonnumeric_coercions':{k:v for k,v in nonnumeric.items() if v},
        'cohort_counts_by_year':counts,
        'unit_of_analysis':'pregnancy; record_id=rok+lp',
        'patient_group_column_used':group_column,
        'repeated_patient_limitation':None if group_column else 'Unique pregnancy IDs do not exclude multiple pregnancies from one woman.',
        'outcome_verification_limitation':'Binary labels cannot establish verification modality or its timing.',
        'control_timing_limitation':'Verify anomaly/CHD were available at the index first-trimester examination.',
        'zero_filled_missing_comparators':False,'regression_reconstruction_of_CSTplus_used':False}
    frames={name:data.loc[m].copy().reset_index(drop=True) for name,m in masks.items()}
    return frames,audit,reasons


def cv_splits(frame,n,seed,grouped=False):
    y=frame['target'].to_numpy(int)
    if min(np.bincount(y,minlength=2))<n:
        raise ValueError('Insufficient minority-class observations for the fixed CV design')
    if grouped:
        splitter=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
        pairs=list(splitter.split(frame,y,groups=frame['_group']))
    else:
        splitter=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
        pairs=list(splitter.split(frame,y))
    seen=np.zeros(len(frame),int)
    for tr,te in pairs:
        if set(frame.iloc[tr]['record_id']) & set(frame.iloc[te]['record_id']):
            raise AssertionError('Pregnancy overlap across fit/holdout')
        if grouped and set(frame.iloc[tr]['_group']) & set(frame.iloc[te]['_group']):
            raise AssertionError('Patient overlap across fit/holdout')
        if len(np.unique(y[tr]))!=2 or len(np.unique(y[te]))!=2:
            raise ValueError('A CV split lacks one outcome class; revise grouping design')
        seen[te]+=1
    assert np.all(seen==1)
    return pairs


def source_hashes():
    files=['doctoral_models.py','run_doctoral_reanalysis.py','verify_doctoral_artifact.py',
           'summarize_doctoral_reanalysis.py']
    return {name:sha_file(HERE/name) for name in files}


def check_cached(path,signature):
    if not path.exists(): return None
    if path.stat().st_size==0: return None
    sidecar=Path(str(path)+'.sha256')
    if not sidecar.is_file() or sidecar.read_text(encoding='ascii').strip()!=sha_file(path):
        print(f'Unverified stage file; recomputing it: {path}',flush=True)
        return None
    obj=joblib.load(path)  # only files produced by this run, in a trusted directory
    if obj.get('signature')!=signature:
        raise RuntimeError(f'Incompatible cache: {path}. Use a new output directory; never overwrite it.')
    return obj


def tune(spec, frame, X, p, seed, target, signature):
    cached=check_cached(target/'tuning.joblib',signature)
    if cached is not None:
        est=cached['estimator']
        if isinstance(est,RemoteTabPFN): est.allow_remote=True
        return est,cached['best_params'],cached['best_cv_auc']
    folds=cv_splits(frame,p.inner_splits,seed, bool(p.group_column))
    y=frame['target'].to_numpy(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if spec.distributions:
            search=RandomizedSearchCV(spec.estimator,spec.distributions,n_iter=spec.iterations,
                scoring='roc_auc',cv=folds,random_state=p.seed,n_jobs=1,refit=True,
                error_score='raise',verbose=1,pre_dispatch=1)
            search.fit(X,y); est=search.best_estimator_; best=search.best_params_
            auc=float(search.best_score_)
            write_csv(pd.DataFrame(search.cv_results_),target/'tuning_cv_results.csv')
        else:
            est=spec.estimator.fit(X,y); best={}; auc=None
    print('  estimator fitted; saving tuning.joblib',flush=True)
    notices=sorted(set(f'{w.category.__name__}: {w.message}' for w in caught))
    write_json(notices,target/'fit_warnings.json')
    # Save the actual fitted estimator immediately, before calibration work.
    write_joblib({'signature':signature,'estimator':est,'best_params':best,'best_cv_auc':auc},
                 target/'tuning.joblib')
    split_rows=[]
    for k,(_,te) in enumerate(folds,1):
        split_rows.extend({'record_id':str(frame.iloc[i]['record_id']),'inner_fold':k} for i in te)
    write_csv(pd.DataFrame(split_rows),target/'CONTROLLED_inner_fold_assignments.csv.gz')
    return est,best,auc


def fresh_process_verify(bundle, frame, target, allow_cloud):
    expected=predict_bundle(bundle,frame,allow_cloud=allow_cloud)
    # Sensitive numeric data only; no names, record IDs, free text or outcomes.
    payload={'X':frame[RAW_PREDICTORS+CONTROLS].copy(),'prob':expected['prob_euploid'].to_numpy(),
             'zone':expected['zone'].to_numpy()}
    payload_path=target/'CONTROLLED_readback_input.joblib'
    write_joblib(payload,payload_path)
    command=[sys.executable,str(HERE/'verify_doctoral_artifact.py'),
        '--artifact',str(target/'bundle.joblib'),'--sample',str(payload_path),
        '--report',str(target/'artifact_readback_verification.json')]
    if allow_cloud: command.append('--allow-cloud')
    completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if completed.returncode:
        raise RuntimeError('Fresh-process artifact verification failed:\n'+completed.stdout+'\n'+completed.stderr)
    payload_path.unlink(missing_ok=True)
    report=json.loads((target/'artifact_readback_verification.json').read_text())
    if report['status']!='passed': raise RuntimeError('Artifact verification did not pass')
    return report


def fit_one(name, train, test, cohort, fold_name, p, args, root_signature):
    target=Path(args.out)/cohort/'models'/fold_name/name
    target.mkdir(parents=True,exist_ok=True)
    env=versions(name)
    signature=digest({'run':root_signature,'model':name,'environment':env,
        'cohort':cohort,'fold':fold_name,'train_ids':train['record_id'].tolist(),
        'test_ids':test['record_id'].tolist() if test is not None else [],
        'tabpfn_path':args.tabpfn_model_path if name=='tabpfn' else None,'threads':args.jobs})
    done=target/'COMPLETE.json'
    if done.exists():
        info=json.loads(done.read_text())
        if info['signature']!=signature:
            raise RuntimeError(f'Checkpoint signature changed: {target}. Use a new run directory.')
        intact=all((target/f).is_file() and (target/f).stat().st_size>0 and sha_file(target/f)==h
                   for f,h in info['files'].items())
        if intact:
            print(f'RESUME verified: {cohort}/{fold_name}/{name}',flush=True)
            return pd.read_csv(target/'CONTROLLED_outer_predictions.csv.gz') if test is not None else None
        print(f'Incomplete/corrupt outputs; rebuild from compatible stages: {target}',flush=True)
        done.unlink()
    print(f'\nFIT {cohort} / {fold_name} / {name} (n={len(train)})',flush=True)
    spec=model_spec(name,p,args.jobs,args.allow_cloud,args.tabpfn_model_path)
    X=prepare_features(train,p.nt_representation,spec.encoding)
    y=train['target'].to_numpy(int)
    fold_no=int(fold_name.rsplit('_',1)[1]) if fold_name.startswith('outer_fold_') else 91
    seed=p.seed+fold_no*100
    est,best,inner_auc=tune(spec,train,X,p,seed+1,target,signature)
    subfolds=cv_splits(train,p.crossfit_splits,seed+2,bool(p.group_column))
    raw=np.full(len(train),np.nan)
    split_rows=[]
    for k,(fit_ix,hold_ix) in enumerate(subfolds,1):
        split_rows.extend({'record_id':str(train.iloc[i]['record_id']),'calibration_subfold':k} for i in hold_ix)
        cache_path=target/f'crossfit_{k:02d}.joblib'
        cache=check_cached(cache_path,signature)
        if cache is not None and np.array_equal(cache['holdout_positions'],hold_ix):
            raw[hold_ix]=cache['raw_probability']
        else:
            print(f'  development cross-fit {k}/{len(subfolds)}',flush=True)
            sub=clone(est)
            if isinstance(sub,RemoteTabPFN): sub.allow_remote=True
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                sub.fit(X.iloc[fit_ix],y[fit_ix])
                pred=positive_probability(sub,X.iloc[hold_ix])
            write_json(sorted(set(f'{w.category.__name__}: {w.message}' for w in caught)),
                       target/f'crossfit_{k:02d}_warnings.json')
            content={'signature':signature,'holdout_positions':hold_ix,'raw_probability':pred}
            if args.save_crossfit_models: content['estimator']=sub
            write_joblib(content,cache_path); raw[hold_ix]=pred
    assert np.isfinite(raw).all()
    write_csv(pd.DataFrame(split_rows),target/'CONTROLLED_calibration_fold_assignments.csv.gz')
    cal=fit_calibrator(raw,y,p.seed)
    p_dev=calibrate(cal,raw); force=override_mask(train)
    high,low,threshold_info=select_thresholds(y,p_dev,force,p)
    metadata={'version':VERSION,'model':name,'cohort':cohort,'fold':fold_name,
        'artifact_role':'outer-fold estimator' if test is not None else 'full-cohort final inference model',
        'n_training':len(train),'n_euploid':int((y==1).sum()),'n_non_euploid':int((y==0).sum()),
        'training_year_counts':{str(k):int(v) for k,v in train['rok'].value_counts().items()},
        'environment':env,'created_utc':utc(),'signature':signature,'best_params':best,
        'inner_cv_selection_auc_not_validation_result':inner_auc,'threshold_derivation':threshold_info,
        'calibration_intercept':float(cal.intercept_[0]),'calibration_slope':float(cal.coef_[0,0]),
        'performance_source':'pooled outer-test predictions only; never training/full-cohort predictions',
        'crossfit_note':'Hyperparameters selected within outer-development; calibration subfolds reuse selected hyperparameters. Development probabilities are not independent validation.',
        'cloud_model':name=='tabpfn',
        'cloud_artifact_limitation':'Replayable numeric training context and configuration, NOT server neural-network weights; API access required.' if name=='tabpfn' else None,
        'explicit_cloud_backend_identifier_requested':name=='tabpfn' and args.tabpfn_model_path!='auto',
        'cloud_backend_weights_archived':False if name=='tabpfn' else None,
        'historical_backend_equivalence_established':False if name=='tabpfn' else None,
        'research_only':True}
    bundle={'artifact_format':VERSION,'estimator':est,'calibrator':cal,'high_threshold':high,
        'low_threshold':low,'schema':{'raw_predictors':RAW_PREDICTORS,'features':feature_names(p.nt_representation),
        'nt_representation':p.nt_representation,'nt_reference_equation':NT_EQUATION,'encoding':spec.encoding,
        'binary_markers':{c:'0=absent/normal; 1=present/abnormal (source coding)' for c in BINARY},
        'DV':'0=positive; 1=absent; 2=reversed A-wave','target':'1=euploid; 0=non-euploid',
        'CST_and_CSTplus_are_predictors':False,'controls':CONTROLS,
        'override_rule':'anomaly == 1 OR CHD == 1 forces high',
        'first_trimester_control_timing_confirmed':p.first_trimester_controls_confirmed},
        'metadata':metadata}
    write_joblib(bundle,target/'bundle.joblib')
    loaded=joblib.load(target/'bundle.joblib')
    check_frame=test if test is not None else train.sample(n=min(64,len(train)),random_state=p.seed)
    before=predict_bundle(bundle,check_frame,allow_cloud=args.allow_cloud)
    after=predict_bundle(loaded,check_frame,allow_cloud=args.allow_cloud)
    difference=float(np.max(np.abs(before['prob_euploid']-after['prob_euploid'])))
    if difference>1e-12 or not np.array_equal(before['zone'],after['zone']):
        raise RuntimeError(f'joblib roundtrip changed predictions or zones for {name}: {difference}')
    sample=check_frame.sample(n=min(32,len(check_frame)),random_state=p.seed)
    verification=fresh_process_verify(bundle,sample,target,args.allow_cloud)
    metadata['artifact_sha256']=sha_file(target/'bundle.joblib')
    metadata['full_holdout_readback_max_abs_difference']=difference
    metadata['fresh_process_verification']=verification
    write_json(metadata,target/'model_manifest.json')
    dev=pd.DataFrame({'record_id':train['record_id'],'target':y,'raw_crossfit':raw,
                      'calibrated_crossfit':p_dev,'override_high':force})
    write_csv(dev,target/'CONTROLLED_threshold_development.csv.gz')
    files=['bundle.joblib','model_manifest.json','artifact_readback_verification.json']
    if test is not None:
        pred=pd.DataFrame({'record_id':test['record_id'].to_numpy(),'target':test['target'].to_numpy(int),
            'prob_euploid':before['prob_euploid'].to_numpy(), 'zone':before['zone'].to_numpy(),
            'high_threshold':high,'low_threshold':low})
        pred['zone_model_only']=assign_zones(pred['prob_euploid'],high,low,np.zeros(len(test),bool))
        write_csv(pred,target/'CONTROLLED_outer_predictions.csv.gz')
        files.append('CONTROLLED_outer_predictions.csv.gz')
    else: pred=None
    write_json({'signature':signature,'status':'complete','created_utc':utc(),
        'files':{f:sha_file(target/f) for f in files}},done)
    print(f'SAVED + VERIFIED: {target / "bundle.joblib"}',flush=True)
    return pred


def comparator_predictions(frame,train,test,p):
    output=pd.DataFrame({'record_id':test['record_id']})
    for prefix in ('CST','CSTplus'):
        denom_col=prefix+'_min_denom'
        dtest=test[denom_col].to_numpy(float); dtrain=train[denom_col].to_numpy(float)
        ok_test=np.isfinite(dtest)&(dtest>0); ok_train=np.isfinite(dtrain)&(dtrain>0)
        for policy in ('fixed','nested_denom'):
            method=prefix+'_'+policy
            zone=np.full(len(test),None,object); model_only=np.full(len(test),None,object)
            high=low=np.nan
            if policy=='fixed':
                high=p.fixed_high_denom; low=p.fixed_low_denom
                z=np.full(ok_test.sum(),'intermediate',object)
                z[dtest[ok_test]<=high]='high'; z[dtest[ok_test]>low]='low'
            elif len(np.unique(train.loc[ok_train,'target']))==2:
                high,low,_=select_thresholds(train.loc[ok_train,'target'],dtrain[ok_train],
                                             override_mask(train.loc[ok_train]),p)
                z=assign_zones(dtest[ok_test],high,low,np.zeros(ok_test.sum(),bool))
            else: continue
            model_only[ok_test]=z
            z=z.copy(); z[override_mask(test.loc[ok_test])]='high'; zone[ok_test]=z
            output['zone_'+method]=zone; output['zone_model_only_'+method]=model_only
            output['score_'+method]=dtest
            output['high_threshold_'+method]=high; output['low_threshold_'+method]=low
    return output


def get_args():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',required=True)
    a.add_argument('--sheet',default='Dane_Wyczyszczone')
    a.add_argument('--out',default='doctoral_ALLYEARS_NTmom_joblib_v2')
    a.add_argument('--models',nargs='+',choices=ALL_MODELS,default=list(LOCAL_MODELS))
    a.add_argument('--cohorts',nargs='+',choices=COHORTS,default=list(COHORTS[:2]))
    a.add_argument('--nt-representation',choices=['mom','raw','delta'],default='mom')
    a.add_argument('--jobs',type=int,default=4)
    a.add_argument('--bootstrap-reps',type=int,default=2000)
    a.add_argument('--group-column',default=None)
    a.add_argument('--confirm-first-trimester-controls',action='store_true')
    a.add_argument('--save-crossfit-models',action='store_true')
    a.add_argument('--audit-only',action='store_true')
    a.add_argument('--smoke',action='store_true',help='Reduced TEST-ONLY run; cannot be combined with production outputs')
    a.add_argument('--allow-cloud',action='store_true')
    a.add_argument('--tabpfn-model-path',default='auto')
    a.add_argument('--no-final-fit',action='store_true',help='Only outer-fold artifacts; default also fits final artifacts')
    return a.parse_args()


def run(args):
    args.input=str(Path(args.input).resolve()); args.out=str(Path(args.out).resolve())
    if args.smoke: args.out=str(Path(args.out)/'SMOKE_TEST_ONLY')
    if args.jobs<1: raise ValueError('--jobs must be a positive CPU thread count')
    p=Protocol(nt_representation=args.nt_representation,bootstrap_reps=args.bootstrap_reps,
        group_column=args.group_column,first_trimester_controls_confirmed=args.confirm_first_trimester_controls)
    if args.smoke: p=replace(p,smoke=True,outer_splits=2,inner_splits=2,crossfit_splits=2,bootstrap_reps=0)
    p.validate()
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    lock=root/'RUNNING.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Run lock exists: {lock}. Do not launch twice. After a crash, remove it only after confirming no run is active.')
    with os.fdopen(fd,'w') as f: f.write(f'pid={os.getpid()}\nstarted={utc()}\n')
    try:
        frames,audit,exclusions=cohort_data(args.input,args.sheet,args.group_column)
        src=source_hashes()
        protocol={'pipeline_version':VERSION,'protocol':asdict(p),'input_sha256':audit['source_sha256'],
            'source_code_sha256':src,'threshold_policy_source':'Uploaded 2019 code; empirical 3%/5% development objectives retained.',
            'comparator_change':'Nested comparator uses observed minimum denominator, not the legacy sum of inverse denominators.',
            'fixed_boundary_note':'The most recent uploaded code already uses <=300; older NT3 revisions used <300.',
            'tabpfn_input_note':'DV remains categorical code 0/1/2; numeric array plus categorical index replaces string-valued DataFrame input.',
            'outer_fold_design':'Outcome-stratified master folds; identical common-record assignments across cohorts.',
            'preprocessing':'Within every fit; CRL included alongside selected NT representation.',
            'fixed_clinical_bands':'high denominator <=300; intermediate (300,1000]; low >1000',
            'cohort_order':'all_years_shared primary; all_years_predictor_complete selection-sensitivity',
            'registration':'New analysis specification, written after prior exploratory analyses; not retrospectively preregistered.',
            'purpose':'Additional unpublished doctoral analysis; not alteration of constituent publications.'}
        root_sig=digest(protocol)
        if (root/'protocol.json').exists():
            previous=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
            if digest(previous)!=root_sig: raise RuntimeError('Protocol/input/code changed. Use a NEW output directory.')
        else:
            write_json(protocol,root/'protocol.json')
            for f in src:
                (root/'code_snapshot').mkdir(exist_ok=True)
                shutil.copy2(HERE/f,root/'code_snapshot'/f)
        write_json(audit,root/'cohort_audit.json')
        write_csv(exclusions,root/'private'/'eligibility_audit.csv.gz')
        write_csv(pd.DataFrame(audit['cohort_counts_by_year']),root/'cohort_counts.csv')
        print(json.dumps({k:audit[k] for k in ['source_n','model_eligible_n','paired_comparator_complete_n',
            'excluded_for_comparator_only','excluded_non_euploid_for_comparator_only']},indent=2),flush=True)
        if not p.first_trimester_controls_confirmed:
            print('CAUTION: anomaly/CHD timing is an unverified clinical assumption; do not label the study submission-ready.',flush=True)
        master=frames['all_years_predictor_complete'].copy()
        if p.smoke:
            rng=np.random.default_rng(p.seed)
            keep=[]
            for label,count in [(0,60),(1,240)]:
                ids=master.loc[master.target.eq(label),'record_id'].to_numpy()
                keep.extend(rng.choice(ids,min(count,len(ids)),replace=False))
            master=master.loc[master.record_id.isin(keep)].reset_index(drop=True)
            frames={k:d.loc[d.record_id.isin(keep)].reset_index(drop=True) for k,d in frames.items()}
        folds=cv_splits(master,p.outer_splits,p.seed,bool(p.group_column))
        master['outer_fold']=0
        for k,(_,te) in enumerate(folds,1): master.loc[te,'outer_fold']=k
        mapping=master.set_index('record_id')['outer_fold']
        write_csv(master[['record_id','rok','target','outer_fold']],root/'private'/'master_outer_folds.csv.gz')
        for c in args.cohorts:
            frames[c]['outer_fold']=frames[c]['record_id'].map(mapping).astype(int)
            write_csv(frames[c],root/c/'private'/'cohort.csv.gz')
        if args.audit_only:
            write_json({'status':'audit_only_no_training','cohorts':args.cohorts,'created_utc':utc()},root/'last_execution.json')
            print('AUDIT COMPLETE. No models trained.',flush=True); return 0
        if 'tabpfn' in args.models and not args.allow_cloud:
            raise PermissionError('TabPFN requested without --allow-cloud; no clinical data were transmitted.')
        # Check the entire requested dependency set before expensive training.
        for name in args.models: model_spec(name,p,args.jobs,args.allow_cloud,args.tabpfn_model_path)
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
        (root/'requirements_execution.txt').write_text(freeze.stdout,encoding='utf-8')
        write_json({'status':'running','created_utc':utc(),'cohorts':args.cohorts,
            'requested_models':args.models,'protocol_signature':root_sig},root/'last_execution.json')
        failures=[]
        for c in args.cohorts:
            frame=frames[c]
            write_json({'cohort':c,'n':len(frame),'requested_models':args.models,'run_signature':root_sig,
                        'years':sorted(frame['rok'].unique().tolist()),'smoke_test':p.smoke},root/c/'cohort_manifest.json')
            all_predictions={m:[] for m in args.models}; comparator_chunks=[]
            for fold in range(1,p.outer_splits+1):
                train=frame.loc[frame.outer_fold.ne(fold)].reset_index(drop=True)
                test=frame.loc[frame.outer_fold.eq(fold)].reset_index(drop=True)
                assert not set(train.record_id)&set(test.record_id)
                if p.group_column: assert not set(train._group)&set(test._group)
                comparator_chunks.append(comparator_predictions(frame,train,test,p))
                for name in args.models:
                    try:
                        with threadpool_limits(limits=args.jobs):
                            pred=fit_one(name,train,test,c,f'outer_fold_{fold:02d}',p,args,root_sig)
                        all_predictions[name].append(pred)
                    except Exception as exc:
                        failures.append({'cohort':c,'stage':f'outer_fold_{fold:02d}','model':name,
                                         'error':str(exc),'traceback':traceback.format_exc()})
                        write_json(failures,root/'failed_jobs.json')
                        if name!='tabpfn': raise
                        print(f'TabPFN incomplete: {exc}. Local models will continue.',flush=True)
            oof=frame.drop(columns=['_group']).copy()
            comps=pd.concat(comparator_chunks,ignore_index=True)
            assert not comps.record_id.duplicated().any()
            oof=oof.merge(comps,on='record_id',validate='one_to_one',how='left')
            complete=[]
            for name,parts in all_predictions.items():
                if len(parts)!=p.outer_splits: continue
                pred=pd.concat(parts,ignore_index=True)
                if pred.record_id.duplicated().any() or set(pred.record_id)!=set(oof.record_id):
                    raise AssertionError('Incomplete/duplicate OOF participants')
                check=oof[['record_id','target']].merge(pred[['record_id','target']],on='record_id',validate='one_to_one')
                assert np.array_equal(check.target_x,check.target_y)
                pred=pred.drop(columns=['target']).rename(columns={
                    'prob_euploid':'prob_'+name,'zone':'zone_'+name,'zone_model_only':'zone_model_only_'+name,
                    'high_threshold':'high_threshold_'+name,'low_threshold':'low_threshold_'+name})
                oof=oof.merge(pred,on='record_id',validate='one_to_one',how='left'); complete.append(name)
            write_csv(oof,root/c/'private'/'oof_predictions.csv.gz')
            write_json({'requested_models':args.models,'complete_oof_models':complete,
                        'n':len(oof),'n_outer_folds':p.outer_splits,'all_requested_complete':complete==args.models},
                       root/c/'oof_status.json')
            if not args.no_final_fit:
                for name in complete:
                    try:
                        with threadpool_limits(limits=args.jobs):
                            fit_one(name,frame,None,c,'final',p,args,root_sig)
                    except Exception as exc:
                        failures.append({'cohort':c,'stage':'final','model':name,'error':str(exc),
                                         'traceback':traceback.format_exc()})
                        write_json(failures,root/'failed_jobs.json')
                        if name!='tabpfn': raise
            from summarize_doctoral_reanalysis import summarize_cohort
            print(f'AGGREGATE held-out results and exploratory error profiles: {c}',flush=True)
            summarize_cohort(root/c,p)
        from summarize_doctoral_reanalysis import sensitivity_comparison
        sensitivity_comparison(root,p)
        final_status='incomplete' if failures else ('smoke_test_only_complete' if p.smoke else 'complete')
        write_json({'status':final_status,'cohorts':args.cohorts,'requested_models':args.models,
            'created_utc':utc(),'protocol_signature':root_sig,'final_models_requested':not args.no_final_fit,
            'failed_jobs':failures},root/'last_execution.json')
        if failures:
            print('INCOMPLETE: review failed_jobs.json. Do not report missing models as completed.',flush=True)
            return 2
        print(f'COMPLETE: {root}. Performance = outer OOF only. Final joblib files are separate.',flush=True)
        return 0
    except BaseException as exc:
        write_json({'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            'created_utc':utc(),'cohorts':args.cohorts,'requested_models':args.models,
            'error_type':type(exc).__name__,'error':str(exc),
            'note':'Previously verified stage artifacts are preserved; this execution is not complete.'},
            root/'last_execution.json')
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    try:
        sys.exit(run(get_args()))
    except KeyboardInterrupt:
        print('Interrupted. Verified checkpoints are preserved. Repeat the same command to resume.',flush=True)
        sys.exit(130)