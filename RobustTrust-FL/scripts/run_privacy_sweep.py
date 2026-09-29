#!/usr/bin/env python
import argparse,sys,copy
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--sigmas',nargs='+',type=float,default=[0,.4,.8,1.2,1.6])
a=p.parse_args();base=load_config(a.config);rows=[]
for sigma in a.sigmas:
  cfg=copy.deepcopy(base);cfg['privacy']['noise_multiplier']=sigma;cfg['privacy']['enabled']=sigma>0
  cfg['output_dir']=str(Path(base['output_dir'])/'privacy_sweep'/f'sigma_{sigma:.2f}')
  r=RobustTrustSimulator(cfg,'robusttrust').run();r['noise_multiplier']=sigma;rows.append(r)
out=Path(base['output_dir'])/'privacy_sweep';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'privacy_sweep.csv',index=False)
