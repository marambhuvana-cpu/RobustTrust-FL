#!/usr/bin/env python
import argparse,sys,copy
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ratios',nargs='+',type=float,default=[.1,.2,.3,.4]);p.add_argument('--attacks',nargs='+',default=['label_flip','sign_flip','gaussian','model_replacement','backdoor','byzantine','mixed'])
a=p.parse_args();base=load_config(a.config);rows=[]
for attack in a.attacks:
  for ratio in a.ratios:
    cfg=copy.deepcopy(base);cfg['attacks']['type']=attack;cfg['attacks']['malicious_ratio']=ratio
    cfg['output_dir']=str(Path(base['output_dir'])/'attack_sweep'/f'{attack}_{ratio:.2f}')
    r=RobustTrustSimulator(cfg,'robusttrust').run();r.update({'attack':attack,'malicious_ratio':ratio});rows.append(r)
out=Path(base['output_dir'])/'attack_sweep';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'attack_sweep.csv',index=False)
