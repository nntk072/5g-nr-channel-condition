"""Paired, predeclared-seed method comparisons adapted from Aalto thesis workflow."""
from __future__ import annotations
import itertools
import numpy as np


def paired_seed_delta(rows:list[dict], control:str, treatment:str,*,metric:str='auroc',
                      seed:int=1,draws:int=5000)->dict:
    """Compare matched seeds with identical train/val/test and dataset identity.

    Exact sign-flip test is used with <=20 seed pairs. Mean-difference bootstrap
    is descriptive and does not replace independent-scene inference.
    """
    by_method={}
    for method in (control,treatment):
        matching={}
        for r in rows:
            if r['method']==method:
                s=int(r['seed'])
                if s in matching:raise ValueError(f'duplicate seed {s}, method {method}')
                matching[s]=r
        by_method[method]=matching
    a,b=by_method[control],by_method[treatment]
    if set(a)!=set(b) or not a:raise ValueError('Control/treatment seeds differ or empty')
    deltas=[]
    for s in sorted(a):
        # Sizes/counts are necessary but not sufficient; exact split IDs should be
        # verified in saved prediction npz files during independent review.
        if a[s]['split'] != b[s]['split']:
            raise ValueError(f'seed {s} split does not match across methods')
        deltas.append(float(b[s]['metrics'][metric]-a[s]['metrics'][metric]))
    differences=np.asarray(deltas,dtype=np.float64)
    if not np.all(np.isfinite(differences)):raise ValueError('Nonfinite metric delta')
    rng=np.random.default_rng(seed)
    boot=rng.choice(differences,(draws,len(differences)),replace=True).mean(axis=1)
    lo,hi=np.percentile(boot,[2.5,97.5])
    obs=abs(float(differences.mean()))
    if len(differences)<=20:
        # exact randomization: sign flips are valid under a symmetric null
        # difference distribution; do not overinterpret with few seeds.
        abs_means=(abs(float(np.mean(differences*np.array(signs))))
                   for signs in itertools.product([-1,1],repeat=len(differences)))
        p=sum(v>=obs-1e-12 for v in abs_means)/(2**len(differences))
        method_name='exact paired sign-flip'
    else:
        signs=rng.choice([-1.,1.],size=(20000,len(differences)))
        p=float(np.mean(np.abs(np.mean(signs*differences,axis=1))>=obs-1e-12))
        method_name='Monte Carlo paired sign-flip (20000 draws)'
    return {'control':control,'treatment':treatment,'metric':metric,
            'paired_seeds':list(sorted(a)),
            'treatment_minus_control':list(map(float,differences)),
            'mean_delta':float(differences.mean()),
            'mean_delta_bootstrap_ci95':[float(lo),float(hi)],
            'two_sided_sign_flip_p_value':float(p),
            'test':method_name,
            'warning':('Fewer than six seeds: even five consistent wins give p>=0.0625.'
                       if len(differences)<6 else
                       'Seed-level inference cannot replace independent-scene inference.')}
