# tail_an.py DIR: the W2 tail: per expert slot, its last W13 end, its W2 items' start/end; waves' end times
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
R=np.array(R)
w13=R[R[:,2]!=2]; w2=R[R[:,2]==2]
last13=w13[:,1].max(); end=w2[:,1].max()
print('last W13 end %.1f  last W2 end %.1f'%(last13,end))
slots=np.unique(R[:,3])
rows=[]
for s in slots:
    a=w13[w13[:,3]==s]; b=w2[w2[:,3]==s]
    if len(a)==0 or len(b)==0: continue
    rows.append((s,a[:,1].max(),len(a),b[:,0].min(),np.median(b[:,0]),b[:,1].max(),len(b),np.median(b[:,1]-b[:,0])))
rows.sort(key=lambda r:r[1])
print('slot  W13done nW13 | W2 first-start med-start last-end nW2 med-dur')
for r in rows[-14:]:
    print('%4d  %6.1f %4d | %6.1f %6.1f %6.1f %4d %5.1f'%r)
# waves: end of last item
we=np.array([R[R[:,4]==w][:,1].max() for w in np.unique(R[:,4])])
print('wave end pct 50/90/99/max', [round(np.percentile(we,p),1) for p in (50,90,99,100)])
# W2 items ending after last13: how many started before last13
late=w2[w2[:,1]>last13]
print('W2 items ending after last W13 end: %d; started before it %d; their med dur %.1f'%(len(late),(late[:,0]<last13).sum(),np.median(late[:,1]-late[:,0])))
for lo in range(int(last13)-4,int(end)+1,2):
    m=(w2[:,0]>=lo)&(w2[:,0]<lo+2)
    if m.sum(): print('  W2 start %d-%d: n %3d med dur %.1f'%(lo,lo+2,m.sum(),np.median(w2[m,1]-w2[m,0])))
