"""Pure cached arithmetic for the frozen P04 probe; no GPU calls."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

OUT=Path("/home/chenglong/P04-work")
METHODS=("G0","pixel","react","simple","oac")
K=52


def top(table,method):
    return [gid for gid,row in sorted(table.items(),key=lambda pair:(-pair[1][method],pair[0]))[:K]]


def overlap(a,b):
    return len(set(a)&set(b))


def main():
    with gzip.open(OUT/"per_view_gaussian.csv.gz","rt",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    reports=[]
    for scene in ("android","room"):
        grouped={c:{} for c in ("A","B","C","D")}
        viewrows={c:[] for c in grouped}
        for row in rows:
            if row["scene"]!=scene:
                continue
            cond=row["condition"]
            gid=int(row["gaussian_id"])
            if gid not in grouped[cond]:
                grouped[cond][gid]={k:float(row[k]) for k in (*METHODS,"n","A","support")}
            viewrows[cond].append(row)
        for cond in grouped:
            v=viewrows[cond]
            n=len(v)
            projected=sum(int(r["z"]) for r in v)
            contributed=sum(int(r["z"]) and float(r["b"])>1e-12 for r in v)
            accepted=sum(int(r["z"]) and float(r["bm"])>1e-12 for r in v)
            positive=[float(r["u"]) for r in v if int(r["z"]) and float(r["b"])>1e-12]
            reports.append(dict(scene=scene,condition=cond,kind="sets",
                probe_view_pairs=n,projected=projected,contributed=contributed,accepted=accepted,
                projected_no_contribution=projected-contributed,
                contribution_no_accept=contributed-accepted,
                mean_u_given_contribution=float(np.mean(positive)),
                median_u_given_contribution=float(np.median(positive)),
                p10_u_given_contribution=float(np.percentile(positive,10))))
        for comparison,left,right in (("A_to_C","A","C"),("C_to_D","C","D")):
            for method in METHODS:
                a=top(grouped[left],method)
                b=top(grouped[right],method)
                ids=sorted(grouped[left])
                ra={gid:i for i,gid in enumerate(sorted(ids,key=lambda j:(-grouped[left][j][method],j)))}
                rb={gid:i for i,gid in enumerate(sorted(ids,key=lambda j:(-grouped[right][j][method],j)))}
                rho=float(np.corrcoef([ra[j] for j in ids],[rb[j] for j in ids])[0,1])
                changes=[abs(grouped[right][j][method]-grouped[left][j][method]) for j in ids]
                reports.append(dict(scene=scene,comparison=comparison,kind="ranking",method=method,
                    top_k=K,top_overlap=overlap(a,b),spearman_rank=rho,
                    median_absolute_score_change=float(np.median(changes)),
                    max_absolute_score_change=float(max(changes)),
                    new_top_ids=";".join(map(str,sorted(set(b)-set(a))))))
        for cond in grouped:
            finite_positive=[r for r in grouped[cond].values() if r["G0"]>0]
            reports.append(dict(scene=scene,condition=cond,kind="incremental",
                oac_vs_original_top_overlap=overlap(top(grouped[cond],"oac"),top(grouped[cond],"G0")),
                oac_vs_pixel_top_overlap=overlap(top(grouped[cond],"oac"),top(grouped[cond],"pixel")),
                oac_vs_react_top_overlap=overlap(top(grouped[cond],"oac"),top(grouped[cond],"react")),
                oac_vs_simple_top_overlap=overlap(top(grouped[cond],"oac"),top(grouped[cond],"simple")),
                low_support_count=sum(r["support"]<3 for r in grouped[cond].values()),
                zero_A_count=sum(r["A"]<=1e-12 for r in grouped[cond].values()),
                oac_max_over_G0=max((r["oac"]/r["G0"] for r in finite_positive),default=1),
                oac_max_absolute_increment=max((r["oac"]-r["G0"] for r in grouped[cond].values()),default=0)))
        left,right=grouped["C"],grouped["D"]
        low=[gid for gid,r in left.items() if r["A"]<1]
        changed=[(gid,right[gid]["oac"]-left[gid]["oac"]) for gid in low]
        reports.append(dict(scene=scene,comparison="C_to_D",kind="low_evidence_risk",
            low_A_lt1_count=len(low),low_A_max_oac_absolute_increase=max([x[1] for x in changed],default=0),
            low_A_positive_increase_count=sum(x[1]>1e-12 for x in changed),
            low_A_max_oac_multiplier=max((right[j]["oac"]/left[j]["oac"] for j in low if left[j]["oac"]>0),default=1),
            low_A_new_oac_top_count=len((set(top(right,"oac"))-set(top(left,"oac"))) & set(low)),
            total_oac_top_entries=len(set(top(right,"oac"))-set(top(left,"oac")))))
    (OUT/"analysis.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for row in reports:
        if row["kind"] in ("sets","incremental","low_evidence_risk") or row.get("method") in ("G0","simple","oac"):
            print(row)


if __name__=="__main__":
    main()
