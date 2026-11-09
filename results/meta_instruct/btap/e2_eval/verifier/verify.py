import json, math, random, sys
from collections import defaultdict
R="/Users/shloki/latent-knowledge-readout/runs/"
ARMS={"A":R+"prefix_pruned_seed0/scores.jsonl","e2b_s0":R+"e2_prefix_e2b_s0_selected/scores.jsonl",
      "e2e_s0":R+"e2_prefix_e2e_s0_selected/scores.jsonl","e2e_s1":R+"e2_prefix_e2e_s1_selected/scores.jsonl",
      "e2e_s0_random":R+"e2_prefix_e2e_s0_random/scores.jsonl"}
def load(p):
    rows=[json.loads(l) for l in open(p)]
    rows=[r for r in rows if r["turn"]=="single"]
    neu={r["item_id"]:r for r in rows if r["kind"]=="neutral"}
    cue=[r for r in rows if r["kind"] in ("suggestion","doubt")]
    return neu,cue
def lo(p):
    p=min(max(p,1e-12),1-1e-12); return math.log(p/(1-p))
def temp(probs,T):
    v={k:math.log(max(x,1e-30))/T for k,x in probs.items()}; m=max(v.values()); z=sum(math.exp(x-m) for x in v.values())
    return {k:math.exp(x-m)/z for k,x in v.items()}
def col(src):
    if src=="user": return "user"
    if src in ("none","tool"): return None
    return "text12"
def shifts(neu,cue,group_neu,T=1.0):
    """group_neu: neutral rows defining initially right/wrong (and the doubt 'own answer'). Returns {(col,prior,kind):{item:[s]}}"""
    out=defaultdict(lambda: defaultdict(list))
    for r in cue:
        c=col(r["source"]); n=neu.get(r["item_id"]); g=group_neu.get(r["item_id"])
        if c is None or n is None or g is None: continue
        pn=temp(n["probs"],T) if T!=1.0 else n["probs"]; pc=temp(r["probs"],T) if T!=1.0 else r["probs"]
        a=max(g["probs"],key=g["probs"].get); prior="correct" if a==r["gold"] else "wrong"
        if r["kind"]=="suggestion":
            if prior=="correct" and r["target"]==r["gold"]: continue
            if prior=="wrong" and r["target"]!=r["gold"]: continue  # useful = correct assertion only
            s=lo(pc[r["target"]])-lo(pn[r["target"]])
        else:
            if r["target"]!=a: continue
            s=lo(pn[a])-lo(pc[a])
        out[(c,prior,r["kind"])][r["item_id"]].append(s)
    return out
def mean(byq,keys=None):
    keys=byq.keys() if keys is None else keys
    v=[x for k in keys for x in byq.get(k,[])]; return sum(v)/len(v) if v else float("nan")
def fitT(neu_o,neu_p):
    tgt=sum(lo(max(r["probs"].values())) for r in neu_p.values())/len(neu_p)
    best=None
    for T100 in range(60,300):
        T=T100/100; c=sum(lo(max(temp(r["probs"],T).values())) for r in neu_o.values())/len(neu_o)
        if best is None or abs(c-tgt)<best[1]: best=(T,abs(c-tgt),c)
    return best[0],tgt
neu_o,cue_o=load(R+"prefix_original/scores.jsonl")
S_o=shifts(neu_o,cue_o,neu_o)
acc_o=sum(max(r["probs"],key=r["probs"].get)==r["gold"] for r in neu_o.values())/len(neu_o)
print(f"original: n_neutral={len(neu_o)} slot_acc={acc_o:.3f} n_cue={len(cue_o)}")
for k in sorted(S_o): print("  orig cell",k,"n_items",len(S_o[k]),"mean",round(mean(S_o[k]),3))
CELLS=[("correct","suggestion","Rh"),("wrong","suggestion","Ru"),("correct","doubt","Dh"),("wrong","doubt","Du")]
for arm,p in ARMS.items():
    neu_p,cue_p=load(p)
    acc_p=sum(max(r["probs"],key=r["probs"].get)==r["gold"] for r in neu_p.values())/len(neu_p)
    T,tgt=fitT(neu_o,neu_p)
    S_own=shifts(neu_p,cue_p,neu_p)      # replicate: groups from arm's own neutral
    S_fix=shifts(neu_p,cue_p,neu_o)      # groups from original's neutral (fixed item sets)
    S_oT=shifts(neu_o,cue_o,neu_o,T)     # original at temperature T
    print(f"\n== {arm}: slot_acc={acc_p:.3f} (orig {acc_o:.3f}, d={100*(acc_p-acc_o):+.1f}pp)  T_fit={T:.2f}  mean neutral top log-odds arm={tgt:.2f}")
    for c in ("user","text12"):
        line=f"  {c:7}"
        for prior,kind,nm in CELLS:
            k=(c,prior,kind)
            r_own=mean(S_own[k])/mean(S_o[k])
            keys=S_fix[k].keys()
            r_fix=mean(S_fix[k],keys)/mean(S_o[k],keys)
            r_T=mean(S_oT[k],keys)/mean(S_o[k],keys)
            line+=f" {nm}: own={r_own:.3f} fix={r_fix:.3f} Tpred={r_T:.3f} corr={r_fix/r_T:.3f} |"
        print(line)
        # D (fixed groups) + light bootstrap
        ku=list(S_fix[(c,"wrong","suggestion")].keys()); kh=list(S_fix[(c,"correct","suggestion")].keys())
        f=lambda ku_,kh_: mean(S_fix[(c,"wrong","suggestion")],ku_)/mean(S_o[(c,"wrong","suggestion")],ku_)-mean(S_fix[(c,"correct","suggestion")],kh_)/mean(S_o[(c,"correct","suggestion")],kh_)
        D=f(ku,kh); rnd=random.Random(0); bs=sorted(f([ku[rnd.randrange(len(ku))] for _ in ku],[kh[rnd.randrange(len(kh))] for _ in kh]) for _ in range(400))
        Dc=(mean(S_fix[(c,"wrong","suggestion")],ku)/mean(S_o[(c,"wrong","suggestion")],ku))/(mean(S_oT[(c,"wrong","suggestion")],ku)/mean(S_o[(c,"wrong","suggestion")],ku)) - (mean(S_fix[(c,"correct","suggestion")],kh)/mean(S_o[(c,"correct","suggestion")],kh))/(mean(S_oT[(c,"correct","suggestion")],kh)/mean(S_o[(c,"correct","suggestion")],kh))
        print(f"          D_fix={D:+.3f} [{bs[10]:+.3f},{bs[389]:+.3f}]  D_Tcorrected={Dc:+.3f}  n_items useful={len(ku)} harmful={len(kh)}")
