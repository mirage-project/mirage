# idle_an.py DIR: per 2-us bin, how many waves are (a) in a W13 unit, (b) in a W2 item, (c) between their last W13 unit
# and first W2 item, (d) before their first unit, (e) after their last item
import numpy as np, sys
d0=sys.argv[1]
tr=np.load(d0+'/moe_trace_gpu0.npy').astype(np.uint64)
t0=int(open(d0+'/start_gpu0.txt').read().split()[0])
W=[]
for w in range(min(tr.shape[0],1024)):
    it=[]
    for i in range(32):
        st=int(tr[w,i,0]); m=int(tr[w,i,1])
        if not st: break
        k=(m>>32)&0xFF; d=(m&0xFFFFFFFF)/1e3; s=(st-t0)/1e3
        if k in (1,2,4): it.append((s,s+d,k))
    if it: W.append(sorted(it))
print('t   W13  W2  gap13->2  pre  post  (waves of %d)'%len(W))
for t in np.arange(20,76,2.0):
    a=b=c=pre=post=0
    for it in W:
        in13=any(s<=t<e and k!=2 for s,e,k in it); in2=any(s<=t<e and k==2 for s,e,k in it)
        if in13: a+=1
        elif in2: b+=1
        else:
            l13=max([e for s,e,k in it if k!=2],default=None); f2=min([s for s,e,k in it if k==2],default=None)
            if it[0][0]>t: pre+=1
            elif it[-1][1]<=t and max(e for s,e,k in it)<=t: post+=1
            elif l13 is not None and f2 is not None and l13<=t<f2: c+=1
    print('%4.0f %4d %4d %5d %5d %5d'%(t,a,b,c,pre,post))
