#!/usr/bin/env python
import argparse,sys,copy,time
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--clients',nargs='+',type=int,default=[100,250,500,750,1000])
a=p.parse_args();base=load_config(a.config);rows=[]
for n in a.clients:
  cfg=copy.deepcopy(base);cfg['federated']['num_clients']=n;cfg['output_dir']=str(Path(base['output_dir'])/'scalability'/str(n))
  t=time.perf_counter();r=RobustTrustSimulator(cfg,'robusttrust').run();r.update({'num_clients':n,'wall_clock_s':time.perf_counter()-t});rows.append(r)
out=Path(base['output_dir'])/'scalability';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'scalability.csv',index=False)
