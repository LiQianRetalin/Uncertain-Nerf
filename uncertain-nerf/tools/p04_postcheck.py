"""Bounded P04 reference, no-op gradient check, and paired overhead audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from p04_oac_diagnostic import GSPLAT, SCENES, WEIGHT_SHA, Ledger, masks_for, run_probe, save_json, sha
from gsplat import rasterization
from puri_gs.dino_features import FeatureCache
from puri_gs.semantic_mask import StaticResponsibilityHead, masked_photo_loss


def tensor_sha(splats):
    return hashlib.sha256(b"".join(splats[k].detach().cpu().numpy().tobytes() for k in sorted(splats))).hexdigest()


def main():
    out=Path("/home/chenglong/P04-work")
    manifest=json.loads((out/"diagnostic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="DIAGNOSTIC_COMPLETE"
    ledger=Ledger(out)
    sys.path.insert(0,str(GSPLAT/"examples"))
    from datasets.colmap import Parser, Dataset
    from fused_ssim import fused_ssim
    overhead=[]
    validation=[]
    integrity=[]
    for scene_info in manifest["scenes"]:
        scene=scene_info["scene"]
        run,data_dir,feature_dir=SCENES[scene]
        ledger.scene=scene
        name=scene_info["selected"][0]["image_name"]
        ledger.view=name
        split=json.loads((run/"dataset_split.json").read_text())
        source=Parser(str(data_dir),factor=4,normalize=True,test_every=8)
        trainset=Dataset(source,split="train",val_every=0,train_keyword=split["train_keyword"],test_keyword=split["test_keyword"])
        local=next(j for j,i in enumerate(trainset.indices) if source.image_names[int(i)]==name)
        data=trainset[local]
        device=torch.device("cuda:0")
        checkpoint=torch.load(run/"ckpts/ckpt_29999_rank0.pt",map_location="cpu",weights_only=True)
        splats=torch.nn.ParameterDict({k:torch.nn.Parameter(v.to(device)) for k,v in checkpoint["splats"].items()})
        head=StaticResponsibilityHead().to(device)
        head.load_state_dict(torch.load(run/"aux/mask_head_step29999.pt",map_location="cpu",weights_only=True),strict=True)
        head.eval().requires_grad_(False)
        hist=torch.load(run/"aux/residual_hist_step29999.pt",map_location="cpu",weights_only=True)
        assert isinstance(hist,dict) and "histogram" in hist
        cache=FeatureCache(feature_dir,expected_weight_sha256=WEIGHT_SHA)
        target=data["image"][None].to(device).float()/255
        height,width=target.shape[1:3]
        masks,_=masks_for(head,cache,name,height,width,True,device,ledger)
        assert scene_info["selected"][0]["image_sha256"]==sha(Path(source.image_paths[int(trainset.indices[local])]))
        initial=tensor_sha(splats)
        def render(parts, w, h, K):
            return rasterization(parts["means"],parts["quats"],torch.exp(parts["scales"]),
                 torch.sigmoid(parts["opacities"]),torch.cat((parts["sh0"],parts["shN"]),1),
                 torch.linalg.inv(data["camtoworld"][None].to(device)),K,w,h,
                 packed=False,absgrad=True,sh_degree=3,near_plane=.01,far_plane=1e10)
        K=data["K"][None].to(device)
        ledger.condition="reference64"
        first=ledger.call("fine_forward",lambda:render(splats,width,height,K))
        visible=torch.where((first[2]["radii"][0]>0).all(-1))[0]
        sample=visible[torch.linspace(0,len(visible)-1,64,device=device).long()]
        small=torch.nn.ParameterDict({k:torch.nn.Parameter(v[sample].detach().clone()) for k,v in splats.items()})
        Ksmall=K.clone(); Ksmall[:,0,:]*=64/width; Ksmall[:,1,:]*=64/height
        rgb64,alpha64,info64=ledger.call("reference64_forward",lambda:render(small,64,64,Ksmall))
        one=torch.ones((64,64),device=device,dtype=torch.bool)
        _,probe_alpha=ledger.call("reference64_probe_forward",lambda:run_probe(info64,{"A":one,"C":one,"D":one},torch.arange(64,device=device),device))
        delta=(probe_alpha-alpha64[0,:,:,0]).abs()
        validation.append(dict(scene=scene,reference_gaussians=64,width=64,height=64,
                               alpha_max_abs=float(delta.max()),alpha_mean_abs=float(delta.mean()),
                               pixels_over_2e4=int((delta>2e-4).sum())))
        assert float(delta.max())<2e-3 and int((delta>2e-4).sum())<=1
        def gradient_pair(with_probe):
            for parameter in splats.values():
                parameter.grad=None
            rgb,_,info=ledger.call("fine_forward",lambda:render(splats,width,height,K))
            if with_probe:
                ids=visible[torch.linspace(0,len(visible)-1,512,device=device).long()]
                ledger.call("exact_probe_forward",lambda:run_probe(info,masks,ids,device))
            loss,_,_=masked_photo_loss(rgb,target,masks["A"][None,None].float(),fused_ssim_fn=fused_ssim)
            ledger.call("photo_backward",lambda:loss.backward())
            return rgb.detach().clone(),info["means2d"].absgrad.detach().clone(),float(loss.detach())
        ledger.condition="diagnostic_off"
        rgb0,g0,l0=gradient_pair(False)
        ledger.condition="diagnostic_on"
        rgb1,g1,l1=gradient_pair(True)
        validation[-1].update(render_max_abs=float((rgb0-rgb1).abs().max()),
                              absgrad_max_abs=float((g0-g1).abs().max()),loss_abs=abs(l0-l1))
        save_json(out/f"reference_validation_{scene}.json",validation[-1])
        assert validation[-1]["render_max_abs"]==0 and validation[-1]["absgrad_max_abs"]<1e-8 and l0==l1
        ids=visible[torch.linspace(0,len(visible)-1,512,device=device).long()]
        for repeat in range(4):  # one warmup plus three fixed measured pairs
            ledger.condition="overhead_off"
            torch.cuda.reset_peak_memory_stats()
            _,_,_=ledger.call("fine_forward",lambda:render(splats,width,height,K))
            off_sec=float(ledger.rows[-1]["seconds"])
            off_peak=torch.cuda.max_memory_allocated()
            ledger.condition="overhead_on"
            torch.cuda.reset_peak_memory_stats()
            _,_,info=ledger.call("fine_forward",lambda:render(splats,width,height,K))
            on_render_sec=float(ledger.rows[-1]["seconds"])
            ledger.call("exact_probe_forward",lambda:run_probe(info,masks,ids,device))
            on_probe_sec=float(ledger.rows[-1]["seconds"])
            on_peak=torch.cuda.max_memory_allocated()
            overhead.append(dict(scene=scene,view=name,repeat=repeat,warmup=int(repeat==0),
                                 original_forward_seconds=off_sec,diagnostic_render_seconds=on_render_sec,
                                 diagnostic_probe_seconds=on_probe_sec,
                                 diagnostic_total_seconds=on_render_sec+on_probe_sec,
                                 original_peak_bytes=off_peak,diagnostic_peak_bytes=on_peak,
                                 extra_peak_bytes=on_peak-off_peak))
        final=tensor_sha(splats)
        assert final==initial
        paths=[run/"ckpts/ckpt_29999_rank0.pt",run/"aux/mask_head_step29999.pt",run/"aux/residual_hist_step29999.pt"]
        integrity.append(dict(scene=scene,gaussian_tensor_before=initial,gaussian_tensor_after=final,
                              checkpoint_sha256_after=sha(paths[0]),mask_head_sha256_after=sha(paths[1]),
                              residual_hist_sha256_after=sha(paths[2]),
                              checkpoint_unchanged=sha(paths[0])==scene_info["checkpoint_sha256"],
                              mask_head_unchanged=sha(paths[1])==scene_info["mask_head_sha256"],
                              residual_hist_unchanged=sha(paths[2])==scene_info["residual_hist_sha256"]))
        charged=sum(int(r["charged"]) for r in ledger.rows)
        save_json(out/"status.json",dict(stage="POSTCHECK_RUNNING",scene=scene,charged_calls=charged,
                 remaining_calls=300-charged,next_checkpoint="other scene or report"))
    with (out/"overhead.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(overhead[0]));writer.writeheader();writer.writerows(overhead)
    save_json(out/"reference_validation.json",validation)
    save_json(out/"state_integrity.json",integrity)
    charged=sum(int(r["charged"]) for r in ledger.rows)
    save_json(out/"status.json",dict(stage="POSTCHECK_COMPLETE",charged_calls=charged,
              remaining_calls=300-charged,next_checkpoint="independent report and zip"))


if __name__=="__main__":
    main()
