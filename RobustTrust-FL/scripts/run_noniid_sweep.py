#!/usr/bin/env python
import argparse,sys,copy
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--alphas',nargs='+',type=float,default=[10,1,.5,.1])
a=p.parse_args();base=load_config(a.config);rows=[]
for alpha in a.alphas:
  cfg=copy.deepcopy(base);cfg['federated']['dirichlet_alpha']=alpha;cfg['output_dir']=str(Path(base['output_dir'])/'noniid_sweep'/f'alpha_{alpha:g}')
  r=RobustTrustSimulator(cfg,'robusttrust').run();r['dirichlet_alpha']=alpha;rows.append(r)
out=Path(base['output_dir'])/'noniid_sweep';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'noniid_sweep.csv',index=False)
