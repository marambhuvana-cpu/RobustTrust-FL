#!/usr/bin/env python
"""Ablation runner using controlled parameter switches while keeping the same data/seed."""
import argparse,sys,copy
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();base=load_config(a.config);rows=[]
variants={
 'full':{},
 'no_privacy':{'privacy.enabled':False},
 'no_attackshield':{'attackshield.threshold':1.0},
 'no_trust_memory':{'trust.memory':0.0},
 'no_trust_penalty':{'trust.penalty':1.0},
 'no_freshness_decay':{'straggler.freshness_decay':0.0},
 'no_weight_cap':{'aggregation.max_client_weight':1.0},
}
def setdot(d,k,v):
 a,b=k.split('.',1);d[a][b]=v
for name,changes in variants.items():
 cfg=copy.deepcopy(base)
 for k,v in changes.items():setdot(cfg,k,v)
 cfg['output_dir']=str(Path(base['output_dir'])/'ablation'/name)
 r=RobustTrustSimulator(cfg,'robusttrust').run();r['variant']=name;rows.append(r)
out=Path(base['output_dir'])/'ablation';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'ablation.csv',index=False)
