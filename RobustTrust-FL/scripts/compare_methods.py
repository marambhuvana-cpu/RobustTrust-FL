#!/usr/bin/env python
import argparse,sys,copy
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--methods',nargs='+',default=['fedavg','fedprox','median','trimmed_mean','krum','multi_krum','bulyan','trustfed','reputation','robusttrust'])
a=p.parse_args();base=load_config(a.config);rows=[]
for m in a.methods:
    cfg=copy.deepcopy(base);cfg['output_dir']=str(Path(base['output_dir'])/'comparison'/m)
    r=RobustTrustSimulator(cfg,m).run();r['method']=m;rows.append(r)
out=Path(base['output_dir'])/'comparison';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'method_comparison.csv',index=False)
