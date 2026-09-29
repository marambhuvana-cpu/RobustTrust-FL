from __future__ import annotations
import copy, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from .utils import set_seed, flatten_parameters, load_parameter_vector, ensure_dir
from .models import RobustTrustMLP
from .data import load_tabular_dataset, iid_partition, dirichlet_partition
from .state import ClientState, make_client_profiles
from .training import train_local, evaluate
from .attacks import assign_malicious, apply_update_attack, DATA_ATTACKS
from .privacy import PrivacyGuard
from .attackshield import AttackShield
from .trust import TrustShieldNet
from .straggler import StragglerSync
from .aggregation import fedavg,coordinate_median,trimmed_mean,krum,multi_krum,bulyan,robusttrust
from .audit import VerifiAudit

class RobustTrustSimulator:
    def __init__(self,cfg:dict,method='robusttrust'):
        self.cfg=cfg; self.method=method; self.seed=int(cfg.get('seed',42)); set_seed(self.seed)
        self.device='cuda' if torch.cuda.is_available() and cfg.get('device','auto')!='cpu' else 'cpu'
        d=cfg['dataset']
        self.bundle=load_tabular_dataset(d['name'],d.get('raw_dir','data/raw'),d.get('label_column'),d.get('max_rows'),self.seed,
                                         d.get('synthetic_samples',12000),d.get('synthetic_features',30),d.get('synthetic_classes',5))
        f=cfg['federated']; self.n_clients=int(f['num_clients']); self.fraction=float(f['participation_fraction'])
        if f.get('partition','dirichlet')=='iid': parts=iid_partition(self.bundle.y_train,self.n_clients,self.seed)
        else: parts=dirichlet_partition(self.bundle.y_train,self.n_clients,float(f.get('dirichlet_alpha',0.5)),self.seed)
        profiles=make_client_profiles(self.n_clients,self.seed+17)
        self.clients=[ClientState(i,parts[i],profiles[i]) for i in range(self.n_clients)]
        a=cfg['attacks']; assign_malicious(self.clients,float(a.get('malicious_ratio',0.2)),a.get('type','mixed'),self.seed+31)
        self.input_dim=self.bundle.x_train.shape[1]; self.num_classes=len(np.unique(self.bundle.y_train))
        self.model_factory=lambda: RobustTrustMLP(self.input_dim,self.num_classes,float(cfg['model'].get('dropout',0.30)))
        self.global_model=self.model_factory().to(self.device); self.global_vector=flatten_parameters(self.global_model)
        p=cfg['privacy']; self.privacy=PrivacyGuard(p['clipping'],p['noise_multiplier'],p['delta'],p['max_epsilon'],p.get('enabled',True))
        s=cfg['attackshield']; self.attackshield=AttackShield(s['threshold'],s['weights'],s.get('reference','mean'))
        t=cfg['trust']; self.trustnet=TrustShieldNet(t['weights'],t['memory'],t['penalty'],t['high'],t['low'])
        ss=cfg['straggler']; self.sync=StragglerSync(ss['freshness_decay'],ss['max_age'],ss['min_sync_ratio'],ss['max_wait_s'])
        self.outdir=ensure_dir(cfg['output_dir']); self.audit=VerifiAudit(self.outdir/'audit.jsonl')
        self.rng=np.random.default_rng(self.seed)
        self.scaffold_c=torch.zeros_like(self.global_vector); self.scaffold_ci={i:torch.zeros_like(self.global_vector) for i in range(self.n_clients)}
        self.round_rows=[];self.client_rows=[]

    def _available(self):
        return [c for c in self.clients if self.rng.random()<c.profile.availability]

    def _select(self,available,k):
        if len(available)<=k:return available
        if self.method=='robusttrust':
            explore=float(self.cfg['federated'].get('exploration',0.10))
            scores=np.array([max(1e-6,c.trust*c.reliability*max(c.coordination,1e-3)) for c in available])
            scores=scores/scores.sum(); uniform=np.ones_like(scores)/len(scores); probs=(1-explore)*scores+explore*uniform
            idx=self.rng.choice(len(available),k,replace=False,p=probs)
        else: idx=self.rng.choice(len(available),k,replace=False)
        return [available[i] for i in idx]

    def _quality_scores(self,ids,updates,base_loss):
        gains=[]
        for u in updates:
            m=self.model_factory().to(self.device); load_parameter_vector(m,self.global_vector+u)
            metrics=evaluate(m,self.bundle.x_val,self.bundle.y_val,self.device)
            gains.append(base_loss-metrics['loss'])
        a=np.asarray(gains); lo=a.min(); hi=a.max(); q=np.full(len(a),0.5) if hi-lo<1e-12 else (a-lo)/(hi-lo)
        return {cid:float(np.clip(q[j],0,1)) for j,cid in enumerate(ids)}

    def _aggregate(self, ids, updates, counts, risks, accepted, freshness, quality):
        if not updates:return torch.zeros_like(self.global_vector),np.array([])
        if self.method in ('fedavg','fedprox','scaffold'): return fedavg(updates,counts)
        if self.method=='median':return coordinate_median(updates)
        if self.method=='trimmed_mean':return trimmed_mean(updates,float(self.cfg.get('baseline',{}).get('trim_ratio',0.20)))
        f=max(1,int(round(self.cfg['attacks'].get('malicious_ratio',0.2)*len(updates))))
        f=min(f,max(0,(len(updates)-3)//2))
        if self.method=='krum':return krum(updates,f)
        if self.method=='multi_krum':return multi_krum(updates,f)
        if self.method=='bulyan':return bulyan(updates,f)
        states=[self.clients[cid] for cid in ids]
        if self.method=='reputation':
            scores=np.array([max(1e-9,s.effectiveness) for s in states]);w=scores/scores.sum()
            return torch.sum(torch.stack(updates)*torch.tensor(w,dtype=torch.float32)[:,None],dim=0),w
        if self.method=='trustfed':
            scores=np.array([max(1e-9,s.trust)*(1-risks[j]) for j,s in enumerate(states)]);w=scores/scores.sum()
            return torch.sum(torch.stack(updates)*torch.tensor(w,dtype=torch.float32)[:,None],dim=0),w
        return robusttrust(updates,counts,[s.trust for s in states],[s.reliability for s in states],freshness,quality,risks,accepted,
                           float(self.cfg['aggregation']['max_client_weight']))

    def run(self):
        cfg=self.cfg; f=cfg['federated']; tr=cfg['training']; attack_cfg=cfg['attacks']
        rounds=int(f['rounds']); k=max(1,int(round(self.n_clients*self.fraction))); patience=int(tr.get('early_stopping_patience',10))
        best_loss=float('inf'); stale=0
        for rnd in range(1,rounds+1):
            load_parameter_vector(self.global_model,self.global_vector)
            before=evaluate(self.global_model,self.bundle.x_val,self.bundle.y_val,self.device); base_loss=before['loss']
            available=self._available(); selected=self._select(available,min(k,len(available)))
            raw=[]; ids=[]; counts=[]; delays=[]; privacy_ok=[]; eps=[]; steps_by_id={}
            # local training
            for c in selected:
                c.selected_count+=1
                active_attack=c.attack_type if c.malicious and rnd>=int(attack_cfg.get('activation_round',11)) else 'none'
                data_attack=active_attack if active_attack in DATA_ATTACKS else 'none'
                fedprox_mu=float(tr.get('fedprox_mu',0.01)) if self.method=='fedprox' else 0.0
                ci=self.scaffold_ci[c.client_id] if self.method=='scaffold' else None
                cg=self.scaffold_c if self.method=='scaffold' else None
                u,steps=train_local(self.model_factory,self.global_vector,self.bundle.x_train,self.bundle.y_train,c.indices,self.device,
                                    int(tr['local_epochs']),int(tr['batch_size']),float(tr['learning_rate']),float(tr['weight_decay']),
                                    data_attack,self.num_classes,float(attack_cfg.get('backdoor_rate',0.20)),fedprox_mu,ci,cg)
                steps_by_id[c.client_id]=steps
                raw.append(u);ids.append(c.client_id);counts.append(len(c.indices))
            if not ids: continue
            benign_std=float(torch.std(torch.cat(raw))) if raw else 1e-3
            protected=[]
            for cid,u in zip(ids,raw):
                c=self.clients[cid]; active_attack=c.attack_type if c.malicious and rnd>=int(attack_cfg.get('activation_round',11)) else 'none'
                if active_attack not in DATA_ATTACKS and active_attack!='none':
                    u=apply_update_attack(u,active_attack,self.rng,benign_std,float(attack_cfg.get('sign_scale',5.0)),float(attack_cfg.get('replacement_scale',10.0)))
                pu,ok,e=self.privacy.protect(u,c,self.fraction,c.participated_count+1,self.rng); protected.append(pu);privacy_ok.append(ok);eps.append(e)
                dly=self.sync.simulate_delay(c.profile,pu.numel()*4,len(c.indices),int(tr['local_epochs']),self.rng);delays.append(dly)
            prev={cid:self.clients[cid].previous_update for cid in ids}
            shield=self.attackshield.analyze(ids,protected,prev)
            quality_map=self._quality_scores(ids,protected,base_loss)
            accepted=[];risks=[];fresh=[];usable=[]; sync_info={}; trust_info={}
            for j,cid in enumerate(ids):
                c=self.clients[cid]; a=shield[cid]
                # non-robust baselines do not use AttackShield to filter, but we still log its diagnostic output
                is_acc=a['accepted'] if self.method in ('robusttrust','trustfed') else True
                risks.append(a['risk'])
                # update running state; robust trust logic is maintained even when baseline selected for comparable diagnostics
                ti=self.trustnet.update(c,quality_map[cid],a['risk'],a['accepted']); trust_info[cid]=ti
                # virtual round-age from max wait deadline
                age=max(0,int(delays[j]//max(self.sync.max_wait,1e-6)))
                si=self.sync.assess(c,delays[j],age,is_acc); sync_info[cid]=si
                ok=bool(is_acc and si['temporal_eligible'] and privacy_ok[j]) if self.method=='robusttrust' else True
                accepted.append(ok);fresh.append(si['freshness']);usable.append(ok)
            # enforce bounded sync for robusttrust; if below ratio, use whatever valid arrived by deadline, without wall-clock waiting
            sync_ratio=sum(usable)/max(1,len(ids))
            if self.method=='robusttrust' and sync_ratio<self.sync.min_sync:
                # fallback: retain AttackShield/privacy-valid clients with freshest updates to reach minimum when possible
                order=np.argsort([-fresh[j] for j in range(len(ids))])
                target=math.ceil(self.sync.min_sync*len(ids))
                for j in order:
                    if sum(accepted)>=target:break
                    if shield[ids[j]]['accepted'] and privacy_ok[j] and sync_info[ids[j]]['temporal_eligible']:accepted[j]=True
            agg,w=self._aggregate(ids,protected,counts,risks,accepted,fresh,[quality_map[c] for c in ids])
            candidate=self.global_vector+agg
            cand_model=self.model_factory().to(self.device);load_parameter_vector(cand_model,candidate)
            cand=evaluate(cand_model,self.bundle.x_val,self.bundle.y_val,self.device)
            finite=bool(torch.isfinite(candidate).all()); tol=float(tr.get('validation_loss_tolerance',0.20))
            accept_candidate=finite and cand['loss'] <= base_loss*(1+tol)
            if self.method!='robusttrust': accept_candidate=finite
            if accept_candidate:self.global_vector=candidate
            # SCAFFOLD control update after successful local round
            if self.method=='scaffold':
                deltas=[]
                lr=float(tr['learning_rate'])
                for cid,u in zip(ids,raw):
                    ci_old=self.scaffold_ci[cid]; steps=max(1,steps_by_id[cid])
                    ci_new=ci_old-self.scaffold_c-u/(-steps*lr)  # (global-local)/(K lr) = -u/(K lr)
                    deltas.append(ci_new-ci_old); self.scaffold_ci[cid]=ci_new
                if deltas:self.scaffold_c=self.scaffold_c+torch.mean(torch.stack(deltas),dim=0)*(len(ids)/self.n_clients)
            # update histories
            for j,cid in enumerate(ids):
                c=self.clients[cid]; c.participated_count+=1; c.success_count+=int(accepted[j]); c.previous_update=protected[j].clone(); c.last_generated_round=rnd
                c.reliability=0.8*c.reliability+0.2*float(accepted[j]); c.participation=c.participated_count/max(1,c.selected_count)
                c.contribution_history.append(quality_map[cid]); c.effectiveness=float(np.mean(c.contribution_history[-10:]))
                self.client_rows.append({'round':rnd,'client_id':cid,'malicious':c.malicious,'attack_type':c.attack_type,'risk':risks[j],
                                         'trust':c.trust,'quality':quality_map[cid],'reliability':c.reliability,'participation':c.participation,
                                         'freshness':fresh[j],'delay_s':delays[j],'accepted':accepted[j],'privacy_epsilon':eps[j],
                                         'aggregation_weight':float(w[j]) if len(w)>j else 0.0})
            load_parameter_vector(self.global_model,self.global_vector)
            testm=evaluate(self.global_model,self.bundle.x_test,self.bundle.y_test,self.device)
            # malicious detection metrics from AttackShield diagnostic labels
            truth=np.array([self.clients[c].malicious and rnd>=int(attack_cfg.get('activation_round',11)) for c in ids],dtype=bool)
            pred=np.array([not shield[c]['accepted'] for c in ids],dtype=bool)
            tp=int(np.sum(truth&pred));fn=int(np.sum(truth&~pred));fp=int(np.sum(~truth&pred));tn=int(np.sum(~truth&~pred))
            adr=tp/max(1,tp+fn);fpr=fp/max(1,fp+tn)
            row={'round':rnd,'method':self.method,**testm,'val_loss':cand['loss'],'candidate_accepted':accept_candidate,
                 'selected_clients':len(ids),'accepted_clients':int(sum(accepted)),'sync_ratio':float(sum(accepted)/len(ids)),
                 'mean_trust':float(np.mean([self.clients[c].trust for c in ids])),'mean_risk':float(np.mean(risks)),
                 'attack_detection_rate':adr,'attack_fpr':fpr,'mean_privacy_epsilon':float(np.mean(eps))}
            self.round_rows.append(row)
            audit_record={'round':rnd,'method':self.method,'selected_clients':ids,
                          'accepted_clients':[ids[j] for j,a in enumerate(accepted) if a],
                          'rejected_clients':[ids[j] for j,a in enumerate(accepted) if not a],
                          'risk_scores':{str(cid):risks[j] for j,cid in enumerate(ids)},
                          'trust_scores':{str(cid):self.clients[cid].trust for cid in ids},
                          'freshness':{str(cid):fresh[j] for j,cid in enumerate(ids)},
                          'aggregation_weights':{str(cid):float(w[j]) if len(w)>j else 0.0 for j,cid in enumerate(ids)},
                          'privacy_epsilon':{str(cid):eps[j] for j,cid in enumerate(ids)},'candidate_accepted':accept_candidate}
            self.audit.append(audit_record,self.global_vector)
            print(f"round={rnd:03d} method={self.method:12s} acc={testm['accuracy']:.4f} f1={testm['f1']:.4f} ADR={adr:.3f} trust={row['mean_trust']:.3f}")
            if cand['loss']<best_loss-1e-5:best_loss=cand['loss'];stale=0
            else:stale+=1
            if stale>=patience and rnd>=int(attack_cfg.get('activation_round',11))+5:break
        return self._save()

    def _save(self):
        rdf=pd.DataFrame(self.round_rows);cdf=pd.DataFrame(self.client_rows)
        rdf.to_csv(self.outdir/'round_metrics.csv',index=False);cdf.to_csv(self.outdir/'client_metrics.csv',index=False)
        final=rdf.iloc[-1].to_dict() if len(rdf) else {}
        import json
        final['audit_verified']=self.audit.verify(); final['rounds_completed']=len(rdf)
        (self.outdir/'final_metrics.json').write_text(json.dumps(final,indent=2,default=float),encoding='utf-8')
        self._plots(rdf,cdf)
        return final

    def _plots(self,rdf,cdf):
        if rdf.empty:return
        import matplotlib.pyplot as plt
        for col,name,ylabel in [('accuracy','accuracy_vs_round.png','Accuracy'),('mean_risk','risk_vs_round.png','Mean adversarial risk')]:
            plt.figure();plt.plot(rdf['round'],rdf[col]);plt.xlabel('Communication round');plt.ylabel(ylabel);plt.tight_layout();plt.savefig(self.outdir/name,dpi=180);plt.close()
        if not cdf.empty:
            g=cdf.groupby(['round','malicious'])['trust'].mean().unstack()
            plt.figure()
            for c in g.columns:plt.plot(g.index,g[c],label=('Malicious' if c else 'Benign'))
            plt.xlabel('Communication round');plt.ylabel('Mean trust');plt.legend();plt.tight_layout();plt.savefig(self.outdir/'trust_vs_round.png',dpi=180);plt.close()
