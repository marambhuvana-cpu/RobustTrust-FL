#!/usr/bin/env python
import argparse,sys,json,copy
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--method',default=None)
a=p.parse_args();base=load_config(a.config);method=a.method or base.get('method','robusttrust');rows=[]
for seed in a.seeds:
    cfg=copy.deepcopy(base);cfg['seed']=seed;cfg['output_dir']=str(Path(base['output_dir'])/f'{method}_seed{seed}')
    r=RobustTrustSimulator(cfg,method).run();r['seed']=seed;rows.append(r)
out=Path(base['output_dir']);out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(out/f'{method}_repeated_summary.csv',index=False)
num=df.select_dtypes('number');summary=pd.DataFrame({'mean':num.mean(),'std':num.std(ddof=1),'ci95_halfwidth':1.96*num.std(ddof=1)/(len(df)**0.5)});summary.to_csv(out/f'{method}_statistics.csv')
print(summary)
