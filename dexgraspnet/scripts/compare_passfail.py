#!/usr/bin/env python3
"""Render sim-pass vs sim-fail grasps of one object to see the failure mode.
Headless matplotlib. Run from DexGrasp_KIST/grasp_generation."""
import os, sys, argparse
sys.path.append(os.path.realpath("."))
import numpy as np, torch, transforms3d
import trimesh as tm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

tn=["WRJTx","WRJTy","WRJTz"]; rn=["WRJRx","WRJRy","WRJRz"]
jn=["robot0:FFJ3","robot0:FFJ2","robot0:FFJ1","robot0:FFJ0","robot0:MFJ3","robot0:MFJ2","robot0:MFJ1","robot0:MFJ0","robot0:RFJ3","robot0:RFJ2","robot0:RFJ1","robot0:RFJ0","robot0:LFJ4","robot0:LFJ3","robot0:LFJ2","robot0:LFJ1"]

def hp(qpos, dev):
    rot=transforms3d.euler.euler2mat(*[qpos[n] for n in rn])
    r6=rot[:,:2].T.ravel().tolist()
    return torch.tensor([qpos[n] for n in tn]+r6+[qpos[n] for n in jn],dtype=torch.float,device=dev).unsqueeze(0)

def addmesh(ax,V,F,c,a):
    ax.add_collection3d(Poly3DCollection(V[F],alpha=a,facecolor=c,edgecolor="none"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npy",required=True); ap.add_argument("--mesh_path",required=True)
    ap.add_argument("--object_code",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    from utils.kistar_model import HandModel
    hm=HandModel("kistar/kistar.urdf","kistar","kistar/contact_points.json","kistar/penetration_points.json",n_surface_points=0,device="cpu")
    data=np.load(a.npy,allow_pickle=True)
    sim=np.array([g["simulated"] for g in data])
    pidx=np.where(sim)[0][:3]; fidx=np.where(~sim)[0][:3]
    obj0=tm.load(os.path.join(a.mesh_path,a.object_code,"coacd","decomposed.obj"),force="mesh",process=False)
    fig=plt.figure(figsize=(18,6))
    rows=[("SIM-PASS",pidx,"#2ca02c"),("SIM-FAIL",fidx,"#d62728")]
    for ri,(lbl,idxs,col) in enumerate(rows):
        for ci,gi in enumerate(idxs):
            g=data[gi]; qpos=g["qpos"]; sc=float(g["scale"])
            hm.set_parameters(hp(qpos,"cpu")); hmesh=hm.get_trimesh_data(0)
            om=obj0.copy(); om.apply_scale(sc)
            ax=fig.add_subplot(2,3,ri*3+ci+1,projection="3d")
            addmesh(ax,np.asarray(hmesh.vertices),np.asarray(hmesh.faces),"#5b9bd5",0.4)
            addmesh(ax,np.asarray(om.vertices),np.asarray(om.faces),"#ff8c42",1.0)
            allv=np.vstack([np.asarray(hmesh.vertices),np.asarray(om.vertices)])
            c=(allv.min(0)+allv.max(0))/2; r=(allv.max(0)-allv.min(0)).max()/2*1.05
            ax.set_xlim(c[0]-r,c[0]+r);ax.set_ylim(c[1]-r,c[1]+r);ax.set_zlim(c[2]-r,c[2]+r)
            ax.view_init(elev=20,azim=45); ax.set_axis_off()
            ax.set_title(f"{lbl} #{gi} Epen={g['E_pen']:.3f}",color=col,fontsize=10)
    fig.suptitle(f"{a.object_code}: sim-pass (top) vs sim-fail (bottom)",fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.96])
    os.makedirs(os.path.dirname(a.out),exist_ok=True); fig.savefig(a.out,dpi=110,bbox_inches="tight")
    print("Saved",a.out)

if __name__=="__main__": main()
