"""End-to-end leakage-protected experiment with exact dataset and run provenance."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
import numpy as np
from .dataset import ChannelDataset
from .physics import observable_features
from .pilots import comb_ls_estimate
from .evaluation import (holdout_domain,in_domain_split,threshold_from_validation,
                         binary_metrics,block_bootstrap_auc)
from .models import fit_sklearn,fit_coral_mlp
from .provenance import sha256_file,canonical_hash,atomic_json,versions,package_source_sha256
from .statistics import paired_seed_delta


def run_experiment(paths:Sequence[str|Path], output:str|Path, *,
                   holdout:str|None=None, methods:Sequence[str]=('rf','mlp','coral','coral0'),
                   seeds:Sequence[int]=(11,23,47,59,71,83,97,101,109,127),bootstrap:int=200,
                   include_phase:bool=False,observation:str='ideal_cfr',
                   pilot_spacing:int=4,pilot_snr_db:float=12.,observation_seed:int=2026,device:str='cpu',coral_penalty:float=.1)->dict:
    if not paths:raise ValueError('No datasets supplied')
    data=ChannelDataset.concatenate([ChannelDataset.load(p) for p in paths])
    if observation=='ideal_cfr':
        observable=data.h
    elif observation=='comb_ls':
        observable=comb_ls_estimate(data.h,spacing=pilot_spacing,
                                    snr_db=pilot_snr_db,seed=observation_seed)
    else:raise ValueError('observation must be ideal_cfr or comb_ls')
    x=observable_features(observable,include_phase=include_phase)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config={'holdout':holdout,'methods':list(methods),'seeds':list(seeds),
            'bootstrap':bootstrap,'include_phase':include_phase,
            'device':device,'coral_penalty':coral_penalty,
            'observation':observation,'pilot_spacing':pilot_spacing,
            'pilot_snr_db':pilot_snr_db,'observation_seed':observation_seed,'feature_version':'logmag+pdp+moments-v1'}
    manifest={'schema':'loslab.run.v1','config':config,'config_sha256':canonical_hash(config),
              'inputs':[{'path':str(p),'sha256':sha256_file(p)} for p in paths],
              'dependencies':versions(),'package_source_sha256':package_source_sha256(),
              'n_samples':len(data.y),'source_counts':{
                  str(s):int(np.sum(data.source==s)) for s in np.unique(data.source)},
              'domain_counts':{str(s):int(np.sum(data.domain==s)) for s in np.unique(data.domain)},
              'label_semantics':'mixed: RT=geometric, standardized=profile proxy',
              'observation_semantics':('ideal channel, diagnostic upper bound' if observation=='ideal_cfr' else
                   'generic noisy comb pilot estimate, NOT 3GPP DMRS'),
              'study_status':('DEBUG_ONLY' if 'synthetic_debug' in data.source else
                              'SIMULATION_ONLY_UNVALIDATED')}
    atomic_json(output/'manifest.json',manifest)
    results=[]
    for seed in seeds:
        train,val,test=(holdout_domain(data,holdout,seed=seed) if holdout else
                        in_domain_split(data,seed=seed))
        assert not (set(data.group[train])&set(data.group[val]))
        assert not (set(data.group[test])&(set(data.group[train])|set(data.group[val])))
        for method in methods:
            if method in ('rf','mlp','logreg'):
                clf=fit_sklearn(x[train],data.y[train],seed=seed,method=method)
                extra={}
            elif method in ('coral','coral0'):
                clf,extra=fit_coral_mlp(x[train],data.y[train],data.domain[train],
                                      x[val],data.y[val],seed=seed,
                                      penalty=coral_penalty if method=='coral' else 0.,device=device)
            else:raise ValueError(f'Unknown method {method}')
            val_scores=clf.predict_proba(x[val])[:,1]
            threshold=threshold_from_validation(data.y[val],val_scores)
            checkpoint=(output/f'{method}_seed{seed}.pt' if method.startswith('coral') else
                        output/f'{method}_seed{seed}.joblib')
            if method.startswith('coral'):
                clf.save(checkpoint,threshold=threshold,include_phase=include_phase)
            else:
                import joblib
                joblib.dump(clf,checkpoint)
                atomic_json(output/f'{method}_seed{seed}_modelinfo.json',
                            {'format':'loslab.sklearn.v1','threshold':threshold,
                             'include_phase':include_phase,'method':method})
            prob=clf.predict_proba(x[test])[:,1]
            metrics=binary_metrics(data.y[test],prob,threshold=threshold)
            ci=block_bootstrap_auc(data.y[test],prob,data.group[test],seed=seed,
                                   draws=bootstrap)
            result={'seed':int(seed),'method':method,'metrics':metrics,'confidence':ci,
                    'split':{'train_size':len(train),'val_size':len(val),'test_size':len(test),
                             'train_groups':len(set(data.group[train])),
                             'validation_groups':len(set(data.group[val])),
                             'test_groups':len(set(data.group[test])),
                             'train_domains':sorted(set(data.domain[train].tolist())),
                             'test_domains':sorted(set(data.domain[test].tolist()))},'training':extra,'checkpoint_path':str(checkpoint)}
            results.append(result)
            # Keep split indices and probabilities for independently reproducing every metric.
            np.savez_compressed(output/f'{method}_seed{seed}_predictions.npz',
                                train_idx=train,val_idx=val,test_idx=test,
                                y_test=data.y[test],y_val=data.y[val],
                                test_group=np.asarray(data.group[test],dtype=str),
                                test_domain=np.asarray(data.domain[test],dtype=str),
                                p_test=prob,p_val=val_scores,threshold=np.float64(threshold))
        atomic_json(output/'results.json',results)
    aggregated={}
    for method in methods:
        seq=[row['metrics']['auroc'] for row in results if row['method']==method]
        aggregated[method]={'seeds':len(seq),'mean_test_auroc':float(np.mean(seq)),
                            'std_test_auroc':float(np.std(seq,ddof=1)) if len(seq)>1 else None}
    comparisons={}
    for control,treatment in (('mlp','coral'),('coral0','coral')):
        if control in methods and treatment in methods:
            comparisons[f'{treatment}_versus_{control}']=paired_seed_delta(
                results,control,treatment,draws=5000)
    summary={'schema':'loslab.summary.v1','evidence_grade':manifest['study_status'],
             'paired_method_comparisons':comparisons,
             'aggregate_across_fixed_splits':aggregated,
             'warning':'Do not report synthetic-debug results or profile-proxy labels as real LOS measurements.'}
    atomic_json(output/'summary.json',summary)
    return summary
