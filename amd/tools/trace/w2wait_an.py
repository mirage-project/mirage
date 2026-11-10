# w2wait_an.py DIR: W2 items vs their expert's W13 completion: started before it (waited), how long; per start window
import numpy as np, sys
d0=sys.argv[1]
tr=np.load(d0+'/moe_trace_gpu0.npy').astype(np.uint64)
t0=int(open(d0+'/start_gpu0.txt').read().split()[0])
R=[]
for w in range(min(tr.shape[0],1024)):
    for i in range(32):
        st=int(tr[w,i,0]); m=int(tr[w,i,1])
        if not st: break
        k=(m>>32)&0xFF; d=(m&0xFFFFFFFF)/1e3; s=(st-t0)/1e3; sl=(m>>40)&0xFFFF
        if k in (1,2,4): R.append((s,s+d,k,sl,w))
R=np.array(R); w13=R[R[:,2]!=2]; w2=R[R[:,2]==2]
done={s:w13[w13[:,3]==s][:,1].max() for s in np.unique(w13[:,3])}
dn=np.array([done[s] for s in w2[:,3]])
wait=np.clip(dn-w2[:,0],0,None)
print('W2 items %d; started before their expert was done: %d; wait med %.1f us among those'%(len(w2),(wait>0).sum(),np.median(wait[wait>0]) if (wait>0).any() else 0))
for lo in range(40,76,4):
    m=(w2[:,0]>=lo)&(w2[:,0]<lo+4)
    if m.sum()==0: continue
    print('  W2 start %2d-%2d: n %4d, waiting %4d (wait med %4.1f), dur med %4.1f, dur after expert done med %4.1f'%(lo,lo+4,m.sum(),(wait[m]>0).sum(),np.median(wait[m][wait[m]>0]) if (wait[m]>0).any() else 0,np.median(w2[m,1]-w2[m,0]),np.median(w2[m,1]-np.maximum(w2[m,0],dn[m]))))
# ready W2 items not yet started, over time
for t in range(44,66,2):
    ready=(dn<=t)&(w2[:,0]>t)
    print('  t=%d: W2 items whose expert is done but not started: %4d; W2 items running: %4d'%(t,ready.sum(),((w2[:,0]<=t)&(w2[:,1]>t)).sum()))
