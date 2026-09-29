from __future__ import annotations
import copy, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .utils import flatten_parameters


def evaluate(model, x, y, device='cpu', batch_size=512):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
    model.eval(); preds=[]; probs=[]; ys=[]
    ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False)
    loss_fn=nn.CrossEntropyLoss(reduction='sum'); total_loss=0.0
    with torch.no_grad():
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device)
            logits=model(xb); total_loss += loss_fn(logits,yb).item()
            p=torch.softmax(logits,dim=1)
            preds.extend(torch.argmax(p,dim=1).cpu().numpy().tolist())
            probs.append(p.cpu().numpy()); ys.extend(yb.cpu().numpy().tolist())
    probs=np.concatenate(probs,axis=0); ys=np.asarray(ys); preds=np.asarray(preds)
    precision,recall,f1,_=precision_recall_fscore_support(ys,preds,average='macro',zero_division=0)
    out={'loss':total_loss/max(1,len(ys)),'accuracy':accuracy_score(ys,preds),
         'precision':precision,'recall':recall,'f1':f1}
    try:
        Y=np.eye(probs.shape[1])[ys]
        out['roc_auc']=roc_auc_score(Y,probs,average='macro',multi_class='ovr')
        out['pr_auc']=average_precision_score(Y,probs,average='macro')
    except Exception:
        out['roc_auc']=float('nan'); out['pr_auc']=float('nan')
    return out


def train_local(model_factory, global_vector, x, y, indices, device, epochs, batch_size,
                lr, weight_decay, attack_type='none', num_classes=None, backdoor_rate=0.20,
                fedprox_mu=0.0, scaffold_ci=None, scaffold_c=None):
    model=model_factory().to(device)
    from .utils import load_parameter_vector
    load_parameter_vector(model,global_vector)
    model.train()
    xx=x[indices].copy(); yy=y[indices].copy()
    rng=np.random.default_rng(int(indices[0]) if len(indices) else 0)
    if attack_type=='label_flip' and num_classes:
        yy=(yy+1)%num_classes
    elif attack_type=='backdoor' and num_classes and len(xx)>0:
        m=max(1,int(backdoor_rate*len(xx))); sel=rng.choice(len(xx),m,replace=False)
        k=min(3,xx.shape[1]); xx[sel,:k]=3.0; yy[sel]=0
    ds=TensorDataset(torch.from_numpy(xx.astype(np.float32)),torch.from_numpy(yy.astype(np.int64)))
    dl=DataLoader(ds,batch_size=min(batch_size,max(2,len(ds))),shuffle=True,drop_last=False)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn=nn.CrossEntropyLoss(); start=global_vector.to(device)
    steps=0
    for _ in range(epochs):
        for xb,yb in dl:
            if xb.shape[0] < 2: continue  # BatchNorm safety
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            if fedprox_mu>0:
                cur=torch.cat([p.reshape(-1) for p in model.parameters()])
                loss=loss + 0.5*fedprox_mu*torch.sum((cur-start)**2)
            loss.backward()
            if scaffold_ci is not None and scaffold_c is not None:
                # Apply SCAFFOLD correction directly to parameter gradients.
                off=0
                for p in model.parameters():
                    n=p.numel(); corr=(scaffold_c[off:off+n]-scaffold_ci[off:off+n]).view_as(p).to(device)
                    if p.grad is not None: p.grad.add_(corr)
                    off+=n
            opt.step(); steps+=1
    final=flatten_parameters(model)
    return final-global_vector.cpu(), max(steps,1)
