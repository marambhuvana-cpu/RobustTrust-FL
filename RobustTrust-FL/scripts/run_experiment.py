#!/usr/bin/env python
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robusttrust_fl.config import load_config
from robusttrust_fl.simulator import RobustTrustSimulator

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--method',default=None)
a=p.parse_args();cfg=load_config(a.config);method=a.method or cfg.get('method','robusttrust')
# Method-specific output subfolder avoids accidental overwrite.
base=Path(cfg['output_dir']);cfg['output_dir']=str(base/method)
res=RobustTrustSimulator(cfg,method).run();print('\nFinal:',res)
