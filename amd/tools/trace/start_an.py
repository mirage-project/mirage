# start_an.py DIR...: moe start phase per wave from trace kinds 10 (task start -> z_q in LDS) and 11 (-> routing table),
# then the first unit's start (the take); grouped by the CU's moe task start (us after the start barrier)
import numpy as np, sys
for d0 in sys.argv[1:]:
    tr=np.load(d0+'/moe_trace_gpu0.npy').astype(np.uint64)
    t0=int(open(d0+'/start_gpu0.txt').read().split()[0])
    rows=[]
    for w in range(min(tr.shape[0],1024)):
        k10=k11=None; first=None
        for i in range(32):
            st=int(tr[w,i,0]); m=int(tr[w,i,1])
            if not st: break
            k=(m>>32)&0xFF; d=(m&0xFFFFFFFF)/1e3; s=(st-t0)/1e3
            if k==10: k10=(s,s+d)
            elif k==11: k11=(s,s+d)
            elif k in (1,2,4) and (first is None or s<first): first=s
        if k10 and k11 and first is not None: rows.append((k10[0],k10[1],k11[1],first))
    if not rows:
        print(d0,'no start-phase entries'); continue
    r=np.array(rows)
    print('##',d0.split('/')[-1],'waves',len(r))
    for lo,hi in ((0,5),(5,20),(20,30),(30,45)):
        m=(r[:,0]>=lo)&(r[:,0]<hi)
        if m.sum()==0: continue
        md=lambda x: np.median(x)
        print('  task start %2d-%2d: %4d waves | start med %5.1f | z_q in LDS %5.1f (+%4.1f) | table %5.1f (+%4.1f) | first unit %5.1f (+%4.1f)'%(
            lo,hi,m.sum(),md(r[m,0]),md(r[m,1]),md(r[m,1]-r[m,0]),md(r[m,2]),md(r[m,2]-r[m,1]),md(r[m,3]),md(r[m,3]-r[m,2])))
