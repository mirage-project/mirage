# last_an.py DIR: the moe drain: W2 items by start time near the end, the last 25 items to end (start, dur, die, slot,
# the wave's previous item end), and per die the last item end
import numpy as np, sys
d0=sys.argv[1]
tr=np.load(d0+'/moe_trace_gpu0.npy').astype(np.uint64)
t0=int(open(d0+'/start_gpu0.txt').read().split()[0])
R=[]
for w in range(min(tr.shape[0],1024)):
    prev=None
    for i in range(32):
        st=int(tr[w,i,0]); m=int(tr[w,i,1])
        if not st: break
        k=(m>>32)&0xFF; d=(m&0xFFFFFFFF)/1e3; s=(st-t0)/1e3; sl=(m>>40)&0xFFFF
        if k in (1,2,4):
            R.append((s,s+d,k,sl,w,(w//4)%8,prev if prev is not None else -1)); prev=s+d
R=np.array(R)
w2=R[R[:,2]==2]
o=np.argsort(w2[:,1])[-25:]
print('last 25 W2 items: start end dur die slot prev_end')
for r in w2[o]: print('  %6.1f %6.1f %5.1f die %d slot %3d prev %6.1f'%(r[0],r[1],r[1]-r[0],r[5],r[3],r[6]))
print('per die last end:',[round(w2[w2[:,5]==d][:,1].max(),1) for d in range(8)])
print('per die W2 items:',[int((w2[:,5]==d).sum()) for d in range(8)])
we=[R[R[:,4]==w][:,1].max() for w in np.unique(R[:,4])]
print('wave end pct 50/90/99/max',[round(np.percentile(we,p),1) for p in (50,90,99,100)])
for lo in range(60,76,2):
    m=(w2[:,0]>=lo)&(w2[:,0]<lo+2)
    if m.sum(): print('  W2 start %d-%d n %3d dur med %.1f max %.1f'%(lo,lo+2,m.sum(),np.median(w2[m,1]-w2[m,0]),(w2[m,1]-w2[m,0]).max()))
