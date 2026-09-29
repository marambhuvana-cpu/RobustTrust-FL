#!/usr/bin/env python
import argparse
import numpy as np,pandas as pd
from scipy.stats import ttest_rel,wilcoxon
p=argparse.ArgumentParser();p.add_argument('--proposed',required=True);p.add_argument('--baseline',required=True);p.add_argument('--metric',default='accuracy');p.add_argument('--out',default='statistical_comparison.csv');a=p.parse_args()
A=pd.read_csv(a.proposed);B=pd.read_csv(a.baseline);x=A[a.metric].to_numpy(float);y=B[a.metric].to_numpy(float);n=min(len(x),len(y));x=x[:n];y=y[:n];d=x-y
res={'metric':a.metric,'n':n,'proposed_mean':x.mean(),'baseline_mean':y.mean(),'mean_difference':d.mean(),'paired_t_p':ttest_rel(x,y).pvalue,'wilcoxon_p':wilcoxon(x,y).pvalue if np.any(d!=0) else 1.0,'cohen_d':d.mean()/(d.std(ddof=1)+1e-12)}
pd.DataFrame([res]).to_csv(a.out,index=False);print(res)
