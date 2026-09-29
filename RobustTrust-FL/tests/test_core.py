import numpy as np, torch
from robusttrust_fl.attackshield import AttackShield
from robusttrust_fl.trust import TrustShieldNet
from robusttrust_fl.straggler import StragglerSync
from robusttrust_fl.aggregation import robusttrust
from robusttrust_fl.state import ClientState,ClientProfile


def profile():return ClientProfile(1,20,50,0.01,5,1.0)

def test_attackshield_bounds():
    u=[torch.ones(10),torch.ones(10)*1.01,-torch.ones(10)*5]
    out=AttackShield().analyze([0,1,2],u,{0:None,1:None,2:None})
    assert all(0<=v['risk']<=1 for v in out.values())
    assert out[2]['risk']>=out[0]['risk']

def test_trust_good_increases_bad_decreases():
    c=ClientState(0,np.arange(10),profile())
    t=TrustShieldNet(); old=c.trust;t.update(c,1,0,True);assert c.trust>old
    old=c.trust;t.update(c,0,1,False);assert c.trust<old

def test_freshness_monotonic():
    s=StragglerSync();c=ClientState(0,np.arange(10),profile())
    vals=[s.assess(c,1,a,True)['freshness'] for a in range(4)]
    assert vals==sorted(vals,reverse=True)
    assert not s.assess(c,1,4,True)['temporal_eligible']

def test_weights_normalized():
    ups=[torch.ones(4),torch.ones(4)*2,torch.ones(4)*3]
    agg,w=robusttrust(ups,[10,10,10],[.9,.8,.7],[1,1,1],[1,1,1],[1,1,1],[0,.1,.2],[True]*3,.6)
    assert np.isclose(w.sum(),1.0);assert np.all(w>=0)
