"""Ablations: classical RF, standard MLP, and domain-regularized MLP.

The new method is an *experimental candidate*, not claimed to outperform MLP.
No CNN is necessary to test the core generalization hypothesis.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def fit_sklearn(x:np.ndarray,y:np.ndarray,seed:int=42,method:str='mlp'):
    if method=='mlp':
        return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,64,32),
                 activation='relu',alpha=1e-3,batch_size=64,early_stopping=True,
                 n_iter_no_change=18,max_iter=300,random_state=seed)).fit(x,y)
    if method=='rf':
        return RandomForestClassifier(n_estimators=250,min_samples_leaf=3,
                     class_weight='balanced_subsample',random_state=seed,n_jobs=-1).fit(x,y)
    if method=='logreg':
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                  class_weight='balanced',random_state=seed)).fit(x,y)
    raise ValueError(f'Unknown sklearn method {method}')


def _build_domain_mlp(input_dim:int):
    from torch import nn
    class Net(nn.Module):
        def __init__(self,d:int):
            super().__init__()
            self.embed=nn.Sequential(nn.Linear(d,128),nn.LayerNorm(128),nn.ReLU(),
                                     nn.Linear(128,64),nn.ReLU())
            self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
        def forward(self,v):
            z=self.embed(v)
            return self.head(z),z
    return Net(input_dim)


def load_coral_checkpoint(path,device:str='cpu'):
    """Reload model/scaling parameters without pickle or training data."""
    import torch
    payload=torch.load(path,map_location=device,weights_only=True)
    if payload.get('format')!='loslab.coral.v1':
        raise ValueError('Invalid checkpoint format')
    network=_build_domain_mlp(payload['n_features']).to(device)
    network.load_state_dict(payload['state_dict']);network.eval()
    center=payload['mean'].to(device);scale=payload['scale'].to(device)
    class LoadedPredictor:
        threshold=float(payload['threshold'])
        include_phase=bool(payload['include_phase'])
        def predict_proba(self,x):
            v=torch.tensor(x,dtype=torch.float32,device=device)
            with torch.no_grad():
                return torch.softmax(network((v-center)/scale)[0],dim=-1).cpu().numpy()
    return LoadedPredictor()


def fit_coral_mlp(x:np.ndarray,y:np.ndarray,domains:np.ndarray,
                  x_val:np.ndarray,y_val:np.ndarray, seed:int=42,
                  penalty:float=.1, epochs:int=150,lr:float=1e-3,
                  batch_per_domain:int=48, device:str='cpu'):
    """Representation-alignment MLP with second-order CORAL penalty.

    Trains on mini-batches sampled by known *training* domain. Validation is
    group-disjoint and only used for early stopping. Domain is NEVER an input.
    With penalty=0, architecture is identical (required ablation).
    """
    import torch
    from torch import nn
    if penalty<0:raise ValueError('penalty must be >=0')
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    scaler=StandardScaler().fit(x)
    xx=torch.tensor(scaler.transform(x),dtype=torch.float32,device=device)
    yy=torch.tensor(y,dtype=torch.long,device=device)
    xv=torch.tensor(scaler.transform(x_val),dtype=torch.float32,device=device)
    yv=torch.tensor(y_val,dtype=torch.long,device=device)
    domains=np.asarray(domains,dtype=str)
    domain_indices=[np.flatnonzero(domains==d) for d in sorted(set(domains))]
    # No method can infer invariance from one training domain.
    if penalty>0 and len(domain_indices)<2:raise ValueError('CORAL needs >=2 training domains')

    model=_build_domain_mlp(x.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    lossfn=nn.CrossEntropyLoss()
    best_state=None;best_loss=float('inf');patience=0
    for epoch in range(epochs):
        model.train()
        # Equal-domain minibatches (without supplying domain as model input).
        subset=np.concatenate([rng.choice(idx,size=batch_per_domain,replace=len(idx)<batch_per_domain)
                               for idx in domain_indices])
        order=rng.permutation(len(subset));subset=subset[order]
        scores,embedding=model(xx[subset]);objective=lossfn(scores,yy[subset])
        if penalty>0:
            selected_domains=domains[subset]
            mats=[]
            for dom in np.unique(selected_domains):
                z=embedding[torch.tensor(selected_domains==dom,device=device)]
                if len(z)<2:continue
                z=z-z.mean(dim=0,keepdim=True)
                mats.append(z.T@z/(len(z)-1))
            if len(mats)>=2:
                alignment=sum(((mats[i]-mats[j])**2).mean()
                              for i in range(len(mats)) for j in range(i+1,len(mats)))
                objective=objective+penalty*alignment
        optimizer.zero_grad();objective.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss=float(lossfn(model(xv)[0],yv).item())
        if val_loss<best_loss-1e-5:
            best_loss=val_loss;best_state=deepcopy(model.state_dict());patience=0
        else:patience+=1
        if patience>=25:break
    if best_state is None:raise RuntimeError('Could not train CORAL MLP')
    model.load_state_dict(best_state);model.eval()

    class Predictor:
        def save(self,path,*,threshold:float,include_phase:bool=False):
            torch.save({'format':'loslab.coral.v1','n_features':x.shape[1],
                        'state_dict':model.state_dict(),
                        'mean':torch.tensor(scaler.mean_,dtype=torch.float32),
                        'scale':torch.tensor(scaler.scale_,dtype=torch.float32),
                        'threshold':float(threshold),'include_phase':bool(include_phase)},str(path))
        def predict_proba(self,value):
            with torch.no_grad():
                v=torch.tensor(scaler.transform(value),dtype=torch.float32,device=device)
                return torch.softmax(model(v)[0],dim=-1).cpu().numpy()
    return Predictor(),{'epoch_count':epoch+1,'best_validation_ce':best_loss,
                        'penalty':penalty,'train_domains':list(sorted(set(domains)))}
