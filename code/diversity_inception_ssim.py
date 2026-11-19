import numpy as np, pandas as pd, torch, itertools, sys
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pytorch_fid.inception import InceptionV3
from scipy.stats import wilcoxon
IMG='/tmp/v3out/PAPER023041_results/images'
tex=pd.read_csv('/mnt/user-data/uploads/a0cc4e57-texture_list.csv')
net=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
M=['Procedural','fBM_p_B-spline','fBM_p_FF_no_warp','Hybrid-FF_Proposed','fBM_p_Opt_no_warp','Hybrid-Opt']
rows=[]
torch.set_num_threads(8)
for m in M:
    for t in tex.texture_id:
        ims=[np.asarray(Image.open(f"{IMG}/256_{m}_{s}_{t.replace('.jpg','.png')}"),np.float32)/255 for s in range(3)]
        x=torch.tensor(np.stack(ims))[:,None].repeat(1,3,1,1)
        with torch.no_grad(): f=net(x)[0].squeeze(-1).squeeze(-1)
        f=torch.nn.functional.normalize(f,dim=1)
        pairs=list(itertools.combinations(range(3),2))
        cos=np.mean([1-float(f[a]@f[b]) for a,b in pairs])
        ss=np.mean([1-ssim(ims[a],ims[b],data_range=1.0) for a,b in pairs])
        rows.append(dict(method=m,texture_id=t,category=t.split('_')[0],div_incep=cos,div_1mssim=ss))
    print(m, flush=True)
df=pd.DataFrame(rows); df.to_csv('/tmp/div/diversity.csv',index=False)
print(df.groupby('method')[['div_incep','div_1mssim']].agg(['mean','std']).round(4).to_string())
def holm(ps):
    ps=np.asarray(ps); o=np.argsort(ps); adj=np.empty_like(ps); run=0
    for r,i in enumerate(o): run=max(run,min(1,(len(ps)-r)*ps[i])); adj[i]=run
    return adj
res=[]
for a,b in [('fBM_p_B-spline','Procedural'),('Hybrid-FF_Proposed','fBM_p_FF_no_warp'),('Hybrid-Opt','fBM_p_Opt_no_warp')]:
    for k in ('div_incep','div_1mssim'):
        A=df[df.method==a].set_index('texture_id')[k]; B=df[df.method==b].set_index('texture_id')[k].loc[A.index]
        st,p=wilcoxon(A,B); res.append(dict(A=a,B=b,metric=k,meanA=A.mean(),meanB=B.mean(),rel=(A.mean()/B.mean()-1)*100,median_diff=np.median(A-B),W=st,p=p))
r=pd.DataFrame(res); r['p_holm']=holm(r.p.values); print(r.round(5).to_string())
r.to_csv('/tmp/div/diversity_tests.csv',index=False)
