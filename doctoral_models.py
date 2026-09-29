"""Importable model definitions and inference. No file I/O or API calls on import.

Keep this module beside scripts when loading joblib artifacts.
Only load artifacts you trust: joblib/pickle can execute code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple
import importlib.metadata
import inspect
import platform
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

VERSION = "2026.09.09.2"
LOCAL_MODELS = ('logistic_regression', 'cart', 'random_forest', 'extra_trees',
                'xgboost', 'catboost', 'lightgbm')
ALL_MODELS = LOCAL_MODELS + ('tabpfn',)
BINARY = ['NB', 'TR', 'SUA']
RAW_PREDICTORS = ['wiek', 'CRL', 'NuchalT', 'FHR', 'PAPPA', 'fbHCG', 'NB', 'TR', 'SUA', 'DV']
CONTROLS = ['anomaly', 'CHD']
CST = ['T21CST', 'T18CST', 'T13CST']
CSTPLUS = ['T21CST+', 'T18CST+', 'T13CST+']
NT_EQUATION = 'log10(expected NT)=-0.657704+0.0204600*CRL-0.000114893*CRL^2'

@dataclass(frozen=True)
class Protocol:
    seed: int = 42
    nt_representation: str = 'mom'
    outer_splits: int = 5
    inner_splits: int = 5
    crossfit_splits: int = 3
    tuning_iterations: int = 15
    catboost_iterations: int = 6
    low_error_cap: float = 0.03
    high_fpr_cap: float = 0.05
    minimum_low_coverage: float = 0.20
    fixed_high_denom: float = 300.0
    fixed_low_denom: float = 1000.0
    bootstrap_reps: int = 2000
    phenotype_permutations: int = 9999
    group_column: Optional[str] = None
    first_trimester_controls_confirmed: bool = False
    smoke: bool = False

    def validate(self):
        if self.nt_representation not in ('raw', 'delta', 'mom'):
            raise ValueError('Unsupported NT representation')
        if min(self.outer_splits, self.inner_splits, self.crossfit_splits) < 2:
            raise ValueError('Every CV level requires at least 2 folds')
        if not (0 <= self.low_error_cap < 1 and 0 <= self.high_fpr_cap < 1):
            raise ValueError('Invalid error constraints')
        if self.bootstrap_reps < 0 or self.phenotype_permutations < 1:
            raise ValueError('Resampling counts must be non-negative (permutations positive)')
        if self.low_error_cap + self.high_fpr_cap >= 1:
            raise ValueError('Error constraints do not support separated zones')
        if not 0 < self.fixed_high_denom < self.fixed_low_denom:
            raise ValueError('Fixed risk denominators must be ordered')
        if not 0 <= self.minimum_low_coverage <= 1:
            raise ValueError('Invalid coverage safeguard')
        return self


def versions(model_name=None):
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'openpyxl']
    if model_name in ('xgboost', 'catboost', 'lightgbm'):
        names.append(model_name)
    if model_name == 'tabpfn':
        names.append('tabpfn-client')
    out = {'python': platform.python_version()}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def feature_names(rep):
    return ['wiek', 'CRL', {'raw':'NT_raw','delta':'NT_delta','mom':'NT_MoM'}[rep],
            'FHR','PAPPA','fbHCG','NB','TR','SUA','DV']


def derive_nt(frame):
    out = frame.copy()
    crl = pd.to_numeric(out['CRL'], errors='raise').astype(float)
    nt = pd.to_numeric(out['NuchalT'], errors='raise').astype(float)
    expected = 10.0 ** (-0.657704 + 0.0204600*crl - 0.000114893*crl**2)
    out['NT_raw'] = nt
    out['NT_expected'] = expected
    out['NT_delta'] = nt - expected
    out['NT_MoM'] = nt / expected
    return out


def prepare_features(frame, rep, encoding):
    required = RAW_PREDICTORS
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing raw predictors: {missing}')
    clean = frame[required].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(clean.to_numpy(float)).all():
        raise ValueError('Inference requires complete finite raw predictors; no silent imputation')
    if not clean['CRL'].between(45,84).all():
        raise ValueError('CRL outside the validated 45-84 mm input range')
    if (clean[['NuchalT', 'PAPPA', 'fbHCG', 'wiek', 'FHR']] <= 0).any().any():
        raise ValueError('NT, biochemical MoMs, maternal age and FHR must be positive')
    for c in BINARY:
        if not clean[c].isin([0,1]).all():
            raise ValueError(f'{c}: expected unchanged 0/1 coding')
    if not clean['DV'].isin([0,1,2]).all():
        raise ValueError('DV: expected 0=positive, 1=absent, 2=reversed A-wave')
    X = derive_nt(clean)[feature_names(rep)].astype(np.float32)
    if encoding == 'native_categorical':
        X['DV'] = X['DV'].astype(int).astype(str)
    elif encoding not in ('one_hot', 'tabpfn_categorical'):
        raise ValueError(f'Unknown encoding {encoding}')
    return X


def override_mask(frame):
    if not set(CONTROLS).issubset(frame):
        raise ValueError('Both anomaly and CHD are required for policy inference')
    c = frame[CONTROLS].apply(pd.to_numeric, errors='coerce')
    if not c.isin([0,1]).all().all():
        raise ValueError('Controls must be explicitly coded 0/1; missing is not normal')
    return (c.eq(1).any(axis=1)).to_numpy(bool)


def positive_probability(estimator, X):
    p = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_)
    where = np.flatnonzero(classes == 1)
    if where.size != 1 or p.ndim != 2:
        raise ValueError('Estimator does not expose class 1 (euploid) probabilities')
    out = p[:, where[0]]
    if not np.isfinite(out).all() or ((out < 0)|(out > 1)).any():
        raise ValueError('Invalid probabilities')
    return out


def fit_calibrator(raw, y, seed):
    p = np.clip(np.asarray(raw,float),1e-6,1-1e-6)
    x = np.log(p/(1-p)).reshape(-1,1)
    model = LogisticRegression(C=1e6,solver='lbfgs',max_iter=5000,random_state=seed)
    model.fit(x,np.asarray(y,int))
    if float(model.coef_[0,0]) <= 0:
        raise RuntimeError('Non-positive sigmoid slope: stop rather than silently reverse ranking')
    return model


def calibrate(model, raw):
    p = np.clip(np.asarray(raw,float),1e-6,1-1e-6)
    return model.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]


def assign_zones(score, high, low, override):
    score = np.asarray(score,float)
    force = np.asarray(override, bool)
    if score.ndim != 1 or force.shape != score.shape:
        raise ValueError('Scores and override must be aligned one-dimensional arrays')
    if not np.isfinite(score).all() or not np.isfinite([high, low]).all() or high >= low:
        raise ValueError('Invalid scores or overlapping thresholds')
    out = np.full(len(score),'intermediate',object)
    out[score <= high] = 'high'
    out[score >= low] = 'low'
    out[force] = 'high'
    return out


def select_thresholds(y, score, override, protocol):
    """Exact tied-score threshold optimization in O(n log n), not rounded grids.

    Same objectives and tie breaks as the supplied legacy pipeline. Every tied
    group is included/excluded as a unit. Overrides consume the high-risk budget.
    Only DEVELOPMENT data may enter this function.
    """
    y = np.asarray(y,int); score = np.asarray(score,float); force = np.asarray(override,bool)
    if len(y) != len(score) or len(force) != len(y) or set(np.unique(y)) != {0,1}:
        raise ValueError('Threshold selection requires aligned data and both classes')
    if not np.isfinite(score).all():
        raise ValueError('Threshold scores are not finite')
    values, inverse = np.unique(score,return_inverse=True)
    eu = np.bincount(inverse, weights=((y==1)&~force).astype(int), minlength=len(values)).astype(int)
    non = np.bincount(inverse, weights=((y==0)&~force).astype(int), minlength=len(values)).astype(int)
    n_non = int((y==0).sum()); n_eu = int((y==1).sum())
    cap_low = int(np.floor(protocol.low_error_cap*n_non+1e-12))
    cap_high = int(np.floor(protocol.high_fpr_cap*n_eu+1e-12))
    low_non = np.cumsum(non[::-1])[::-1]
    low_eu = np.cumsum(eu[::-1])[::-1]
    possible_low = np.flatnonzero(low_non <= cap_low)
    if len(possible_low):
        i = max(possible_low, key=lambda j:(int(low_non[j]+low_eu[j]),int(low_eu[j]),-float(values[j])))
        low = float(values[i])
    else:
        low = float(np.nextafter(values[-1],np.inf))
    forced_eu = int(((y==1)&force).sum()); forced_non = int(((y==0)&force).sum())
    high_values = np.r_[np.nextafter(values[0],-np.inf), values]
    high_eu = np.r_[0,np.cumsum(eu)]+forced_eu
    high_non = np.r_[0,np.cumsum(non)]+forced_non
    possible_high = np.flatnonzero(high_eu <= cap_high)
    feasible = forced_eu <= cap_high
    if len(possible_high):
        def key(j):
            n = int(high_eu[j]+high_non[j]); ppv = high_non[j]/n if n else 0.0
            return int(high_non[j]), float(ppv), -n, float(high_values[j])
        j = max(possible_high,key=key)
        high = float(high_values[j])
    else:
        high = float(high_values[0])
    if high >= low:
        raise RuntimeError('Independent thresholds overlap; no post-hoc repair is allowed')
    zone = assign_zones(score,high,low,force)
    info = {
        'development_n':len(y),'development_non_euploid':n_non,'development_euploid':n_eu,
        'maximum_development_false_reassurance':cap_low,
        'maximum_development_euploid_high':cap_high,
        'override_euploid':forced_eu,'override_non_euploid':forced_non,
        'high_constraint_feasible_after_override':bool(feasible),
        'development_low_n':int((zone=='low').sum()),
        'development_false_reassurance':int(((zone=='low')&(y==0)).sum()),
        'development_euploid_high':int(((zone=='high')&(y==1)).sum()),
        'development_non_euploid_high':int(((zone=='high')&(y==0)).sum()),
        'high_threshold':high,'low_threshold':low,
        'threshold_selection':'development only; exact tied values; no pooled-OOF tuning'
    }
    assert info['development_false_reassurance'] <= cap_low
    if feasible:
        assert info['development_euploid_high'] <= cap_high
    return high,low,info


class CloneSafeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self,iterations=500,depth=6,learning_rate=.03,l2_leaf_reg=3.0,
                 random_strength=1.0,border_count=128,auto_class_weights=None,
                 random_seed=42,thread_count=4,cat_features=('DV',)):
        self.iterations=iterations; self.depth=depth; self.learning_rate=learning_rate
        self.l2_leaf_reg=l2_leaf_reg; self.random_strength=random_strength
        self.border_count=border_count; self.auto_class_weights=auto_class_weights
        self.random_seed=random_seed; self.thread_count=thread_count; self.cat_features=cat_features
    def fit(self,X,y):
        from catboost import CatBoostClassifier
        kwargs=self.get_params(deep=False).copy(); cats=kwargs.pop('cat_features')
        self.model_=CatBoostClassifier(**kwargs,loss_function='Logloss',eval_metric='AUC',
                                      verbose=False,allow_writing_files=False)
        self.model_.fit(X,y,cat_features=list(cats))
        self.classes_=np.asarray(self.model_.classes_); self.n_features_in_=X.shape[1]
        self.feature_names_in_=np.asarray(X.columns,object)
        return self
    def predict_proba(self,X):
        return self.model_.predict_proba(X)
    def predict(self,X):
        return np.asarray(self.model_.predict(X)).ravel().astype(int)
    def __sklearn_is_fitted__(self):
        return hasattr(self,'model_')


class RemoteTabPFN(ClassifierMixin, BaseEstimator):
    """A replayable cloud inference context, NOT local neural-network weights.

    Serialized state is an allow-list: numeric training predictors, labels and
    configuration only. Tokens and the live SDK client are never serialized.
    Predicting after loading requires explicit allow_remote=True and API access.
    The training context is sensitive research data, even without identifiers.
    """
    def __init__(self,model_path='auto',random_state=42,categorical_features_indices=(9,),
                 allow_remote=False,thinking_mode=False):
        self.model_path=model_path; self.random_state=random_state
        self.categorical_features_indices=categorical_features_indices
        self.allow_remote=allow_remote; self.thinking_mode=thinking_mode
    def fit(self,X,y):
        if not self.allow_remote:
            raise PermissionError('TabPFN requires explicit cloud permission')
        self.X_train_=np.asarray(X,dtype=np.float32).copy()
        self.y_train_=np.asarray(y,dtype=int).copy()
        if self.X_train_.ndim != 2 or len(self.X_train_) != len(self.y_train_):
            raise ValueError('TabPFN training arrays are not aligned')
        if not np.isfinite(self.X_train_).all() or set(np.unique(self.y_train_)) != {0,1}:
            raise ValueError('TabPFN requires finite inputs and both outcome classes')
        self.classes_=np.array([0,1]); self.n_features_in_=self.X_train_.shape[1]
        self._client=None
        return self
    def _make_client(self):
        if not self.allow_remote:
            raise PermissionError('Loading a cloud context does not authorize transmission')
        from tabpfn_client import TabPFNClassifier
        params=dict(model_path=self.model_path,random_state=self.random_state,
                    categorical_features_indices=list(self.categorical_features_indices),
                    thinking_mode=self.thinking_mode)
        model=TabPFNClassifier(**params)
        model.fit(self.X_train_,self.y_train_)
        return model
    def predict_proba(self,X):
        if not self.allow_remote:
            raise PermissionError('Explicit cloud permission is required at inference')
        for attempt in range(6):
            try:
                if self._client is None:
                    self._client=self._make_client()
                values=np.asarray(self._client.predict_proba(np.asarray(X,dtype=np.float32)), dtype=float)
                remote_classes=np.asarray(getattr(self._client, 'classes_', [0,1]))
                if set(remote_classes.tolist()) != {0,1} or values.shape != (len(X),2):
                    raise ValueError('Unexpected TabPFN probability schema')
                # Expose columns in the declared local classes_ order, even if
                # the SDK returns the two outcome classes in the reverse order.
                order=[int(np.flatnonzero(remote_classes == k)[0]) for k in self.classes_]
                values=values[:,order]
                if not np.isfinite(values).all() or ((values < 0)|(values > 1)).any():
                    raise ValueError('Invalid TabPFN probabilities')
                return values
            except Exception as exc:
                msg=str(exc).lower()
                retry=any(x in msg for x in ('503','502','504','temporar','timeout','connection reset'))
                retry=retry or ('prepare_train_set_upload' in msg and '400' in msg and 'bad request' in msg)
                if not retry or attempt==5:
                    raise
                self._client=None
                wait=min(15*(2**attempt),120)
                print(f'TabPFN transient service failure; retry in {wait}s',flush=True)
                time.sleep(wait)
    def predict(self,X):
        return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
    def __getstate__(self):
        allowed=('model_path','random_state','categorical_features_indices','thinking_mode',
                 'X_train_','y_train_','classes_','n_features_in_')
        return {key:self.__dict__[key] for key in allowed if key in self.__dict__}
    def __setstate__(self,state):
        self.__dict__.update(state); self.allow_remote=False; self._client=None
    def __sklearn_is_fitted__(self):
        return hasattr(self,'X_train_')


@dataclass
class Spec:
    name: str
    encoding: str
    estimator: Any
    distributions: Optional[dict]
    iterations: int


def model_spec(name, protocol, jobs=4, cloud=False, tabpfn_path='auto'):
    """Search spaces retained from the supplied NT3 nondegenerate v2 core."""
    seed=protocol.seed; iterations=protocol.tuning_iterations
    def pipe(est):
        cont=feature_names(protocol.nt_representation)[:6]
        pre=ColumnTransformer([
            ('continuous',StandardScaler(),cont),('binary','passthrough',BINARY),
            ('DV',OneHotEncoder(categories=[[0,1,2]],handle_unknown='error',
                                sparse_output=False,dtype=np.float32),['DV'])],
            remainder='drop',verbose_feature_names_out=False)
        return Pipeline([('preprocess',pre),('clf',est)])
    encoding='one_hot'; dist=None
    if name=='logistic_regression':
        est=pipe(LogisticRegression(solver='liblinear',max_iter=5000,random_state=seed))
        dist={'clf__C':loguniform(1e-3,1e3),'clf__penalty':['l1','l2'],'clf__class_weight':[None,'balanced']}
    elif name=='cart':
        est=pipe(DecisionTreeClassifier(random_state=seed))
        dist={'clf__criterion':['gini','entropy','log_loss'],'clf__max_depth':[2,3,4,5,6,None],
              'clf__max_leaf_nodes':randint(3,13),'clf__min_samples_leaf':randint(2,41),
              'clf__min_samples_split':randint(4,61),'clf__class_weight':[None,'balanced'],
              'clf__ccp_alpha':uniform(0,.02)}
    elif name in ('random_forest','extra_trees'):
        cls=RandomForestClassifier if name=='random_forest' else ExtraTreesClassifier
        est=pipe(cls(random_state=seed,n_jobs=jobs))
        dist={'clf__n_estimators':randint(300,1001),'clf__max_depth':[None,4,6,8,10,12],
              'clf__min_samples_leaf':randint(1,21),'clf__min_samples_split':randint(2,31),
              'clf__max_features':['sqrt','log2',.5,.8,1.0],
              'clf__class_weight':[None,'balanced','balanced_subsample']}
    elif name=='xgboost':
        from xgboost import XGBClassifier
        est=pipe(XGBClassifier(objective='binary:logistic',eval_metric='logloss',tree_method='hist',
                              random_state=seed,n_jobs=jobs))
        dist={'clf__n_estimators':randint(200,1001),'clf__max_depth':randint(2,7),
              'clf__learning_rate':loguniform(.005,.20),'clf__subsample':uniform(.60,.40),
              'clf__colsample_bytree':uniform(.60,.40),'clf__min_child_weight':randint(1,11),
              'clf__gamma':uniform(0,3),'clf__reg_alpha':loguniform(1e-5,2),
              'clf__reg_lambda':loguniform(.1,10)}
    elif name=='catboost':
        import catboost  # fail immediately if the requested package is absent
        est=CloneSafeCatBoostClassifier(random_seed=seed,thread_count=jobs)
        encoding='native_categorical'; iterations=protocol.catboost_iterations
        dist={'iterations':randint(200,601),'depth':randint(3,8),
              'learning_rate':loguniform(.01,.15),'l2_leaf_reg':loguniform(.5,15),
              'random_strength':uniform(0,2),'border_count':[32,64,128],
              'auto_class_weights':[None,'Balanced']}
    elif name=='lightgbm':
        from lightgbm import LGBMClassifier
        est=pipe(LGBMClassifier(objective='binary',random_state=seed,n_jobs=jobs,verbosity=-1))
        dist={'clf__n_estimators':randint(200,1001),'clf__learning_rate':loguniform(.005,.20),
              'clf__num_leaves':randint(7,64),'clf__max_depth':[-1,3,4,5,6,8,10],
              'clf__min_child_samples':randint(5,51),'clf__subsample':uniform(.60,.40),
              'clf__colsample_bytree':uniform(.60,.40),'clf__reg_alpha':loguniform(1e-5,2),
              'clf__reg_lambda':loguniform(.1,10),'clf__class_weight':[None,'balanced']}
    elif name=='tabpfn':
        if not cloud:
            raise PermissionError('Use --allow-cloud only with authorization for the clinical data')
        import tabpfn_client
        encoding='tabpfn_categorical'; iterations=0
        est=RemoteTabPFN(model_path=tabpfn_path,random_state=seed,allow_remote=True)
    else:
        raise ValueError(f'Unknown model {name}')
    if protocol.smoke:
        # Deliberately different protocol and output directory; never publication data.
        params=est.get_params(deep=True)
        small={k:12 for k in ('clf__n_estimators','iterations') if k in params}
        if small:
            est.set_params(**small)
        dist=None; iterations=0
    return Spec(name,encoding,est,dist,iterations)


def predict_bundle(bundle, frame, apply_override=True, allow_cloud=False):
    schema=bundle['schema']
    X=prepare_features(frame,schema['nt_representation'],schema['encoding'])
    est=bundle['estimator']
    if isinstance(est,RemoteTabPFN):
        est.allow_remote=allow_cloud
    raw=positive_probability(est,X)
    p=calibrate(bundle['calibrator'],raw)
    force=override_mask(frame) if apply_override else np.zeros(len(frame),bool)
    zone=assign_zones(p,bundle['high_threshold'],bundle['low_threshold'],force)
    return pd.DataFrame({'prob_euploid':p,'zone':zone,'override_high':force},index=frame.index)