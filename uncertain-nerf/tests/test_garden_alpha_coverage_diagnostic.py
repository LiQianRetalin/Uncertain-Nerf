from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tools.diagnose_puri_gs_garden_alpha_coverage import (
    SELECTED_IMAGES, TEST_NAMES, TEST_NAMES_SHA256, alpha_interaction, background_identity_error,
    choose_diagnosis, composited_alpha, excess_error_decomposition, names_sha256,
    next_mechanism, opacity_support_contract, refuse_existing_output,
    source_has_zero_training_contract, support_decomposition, top_connected_component,
    validate_alpha, _render, _save_selected_npz,
)


def test_fixed_test_names_and_alpha_transmittance_contract():
    assert len(TEST_NAMES) == 24
    assert names_sha256(TEST_NAMES) == TEST_NAMES_SHA256
    assert composited_alpha([0.2, 0.5]) == pytest.approx(0.6)
    alpha = np.array([[0.0, 0.6, 1.0]], np.float32)
    assert validate_alpha(alpha)["transmittance_identity_max_abs_error"] == 0
    with pytest.raises(RuntimeError): validate_alpha(np.array([1.01]))


def test_black_white_identity_and_native_ones_probe_agree():
    alpha = np.array([[0.2, 0.9]], np.float32)
    black = np.array([[[.1,.2,.3],[.2,.2,.2]]],np.float32)
    white = black + (1-alpha)[...,None]
    assert background_identity_error(black,white,alpha) <= 1e-7
    ones_probe = 1-(1-np.array(.2))*(1-np.array(.5))
    assert ones_probe == pytest.approx(composited_alpha([.2,.5]))


def test_fixed_opacity_is_renderer_value_and_does_not_mutate_logits():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([-4.0,0.0,4.0]); original=logits.clone()
    effective,before=opacity_support_contract(logits)
    assert torch.equal(logits,original) and torch.equal(before,original)
    assert torch.equal(effective,torch.full_like(logits,.5))
    assert not torch.allclose(torch.sigmoid(effective),effective)


def test_synthetic_renderer_probes_preserve_geometry_and_sh_alpha(monkeypatch):
    torch = pytest.importorskip("torch")
    calls=[]
    def fake_rasterization(**kwargs):
        calls.append(kwargs)
        height,width=kwargs["height"],kwargs["width"]
        alpha=kwargs["opacities"].mean().expand(1,height,width,1).clone()
        rgb=alpha.expand(1,height,width,3).clone()
        if kwargs.get("backgrounds") is not None:
            rgb=rgb+(1-alpha)*kwargs["backgrounds"][:,None,None,:]
        if kwargs["render_mode"]=="RGB+ED":
            rgb=torch.cat([rgb,torch.ones((1,height,width,1))],dim=-1)
        return rgb,alpha,{"radii":torch.ones((1,kwargs["means"].shape[0]))}
    rendering=types.ModuleType("gsplat.rendering"); rendering.rasterization=fake_rasterization
    gsplat=types.ModuleType("gsplat"); gsplat.rendering=rendering
    monkeypatch.setitem(sys.modules,"gsplat",gsplat); monkeypatch.setitem(sys.modules,"gsplat.rendering",rendering)
    splats={
        "means":torch.tensor([[1.,2.,3.],[4.,5.,6.]]), "quats":torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]),
        "scales":torch.zeros((2,3)), "opacities":torch.tensor([-2.,2.]),
        "sh0":torch.zeros((2,1,3)), "shN":torch.zeros((2,15,3)),
    }
    original={key:value.clone() for key,value in splats.items()}
    data={"image":torch.zeros((2,3,3),dtype=torch.uint8),"camtoworld":torch.eye(4),"K":torch.eye(3)}
    standard=_render(splats,data,device="cpu"); sh0=_render(splats,data,device="cpu",sh_degree=0)
    support=_render(splats,data,device="cpu",support=True)
    assert np.array_equal(standard["alpha"],sh0["alpha"])
    assert support["alpha"].mean()==pytest.approx(.5)
    assert calls[-1]["sh_degree"] is None and torch.equal(calls[-1]["opacities"],torch.full((2,),.5))
    for key in ("means","quats","scales","opacities"):
        assert torch.equal(splats[key],original[key])


def test_top_roi_uses_row_zero_eight_connectivity_and_largest_component():
    mask=np.zeros((5,7),bool); mask[0,0]=True; mask[1,1]=True
    mask[0,4:6]=True; mask[1,3:6]=True; mask[2,4]=True
    roi,status=top_connected_component(mask)
    assert status["roi_status"]=="nonempty" and roi[2,4] and not roi[0,0]
    empty,empty_status=top_connected_component(np.zeros((2,2),bool))
    assert not empty.any() and empty_status["roi_status"]=="empty"


def test_excess_error_decomposition_uses_registered_denominators():
    b1e=np.zeros(4); tare=np.array([1.,2.,3.,4.]); b1a=np.array([.9,.9,.9,.2]); tara=np.array([.1,.5,.9,.1])
    got=excess_error_decomposition(b1e,tare,b1a,tara)
    assert got["S_eligible"]["value"]==pytest.approx(6/10)
    assert got["S_cov"]["value"]==pytest.approx(1/6)
    assert got["S_transition"]["value"]==pytest.approx(2/6)
    assert got["S_high"]["value"]==pytest.approx(3/6)
    empty=excess_error_decomposition(np.ones(2),np.zeros(2),np.ones(2),np.ones(2))
    assert empty["S_eligible"]["status"]=="not_applicable"


def test_support_subtype_interaction_and_delta_direction():
    b1e=np.zeros(3); tare=np.ones(3); b1a=np.ones(3); tara=np.zeros(3)
    low=support_decomposition(b1e,tare,b1a,tara,np.array([.1,.1,.9]))
    assert low["S_low_support"]["value"]==pytest.approx(2/3)
    assert low["secondary_subtype"]=="LOW_FIXED_OPACITY_SUPPORT"
    b1=np.array([1.,0.]); dg=np.array([.6,.2]); mask=np.array([.7,.4]); ru=np.array([.1,.9])
    assert np.allclose(alpha_interaction(b1,dg,mask,ru),np.array([-.2,.3]))
    ru_a=np.array([.2,.4]); align=np.array([.3,.2]); tar=np.array([.1,.5])
    assert np.allclose(align-ru_a,[.1,-.2]) and np.allclose(tar-align,[-.2,.3])


def test_label_priority_and_next_mechanism():
    def block(e,c,h,t):
        return {"S_eligible":{"value":e},"S_cov":{"value":c},"S_high":{"value":h},"S_transition":{"value":t}}
    controls=block(.1,.1,.1,.8)
    assert choose_diagnosis(block(.9,.6,.1,.3),block(.9,.6,.1,.3),controls,True)=="ALPHA_COVERAGE_HOLE_SUPPORTED"
    assert choose_diagnosis(block(.9,.3,.3,.4),block(.9,.3,.3,.4),controls,True)=="MIXED_FAILURE_SUPPORTED"
    assert choose_diagnosis(block(.9,.1,.7,.2),block(.9,.1,.7,.2),controls,True)=="HIGH_ALPHA_RENDERING_ERROR_SUPPORTED"
    assert choose_diagnosis(block(.4,.8,.1,.1),block(.9,.8,.1,.1),controls,True)=="DIAGNOSIS_INCONCLUSIVE"
    assert choose_diagnosis(block(.9,.1,.2,.7),block(.9,.1,.2,.7),controls,True)=="DIAGNOSIS_INCONCLUSIVE"
    assert next_mechanism("ALPHA_COVERAGE_HOLE_SUPPORTED","LOW_FIXED_OPACITY_SUPPORT")=="coverage_topology"


def test_output_refusal_json_arrays_and_zero_training_static_contract(tmp_path):
    existing=tmp_path/"existing"; existing.mkdir()
    with pytest.raises(FileExistsError): refuse_existing_output(existing)
    source=Path(__file__).parents[1]/"tools"/"diagnose_puri_gs_garden_alpha_coverage.py"
    contract=source_has_zero_training_contract(source)
    assert all(contract.values()), contract
    tree=ast.parse(source.read_text(encoding="utf-8"))
    imports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    assert "run_puri_gs" not in imports and "puri_gs.ru_training" not in imports
    text=source.read_text(encoding="utf-8")
    assert "optimizer_step_count\": 0" in text and "torch.inference_mode()" in text
    assert "SELECTED_IMAGES = FAILURE_IMAGES + CONTROL_IMAGES" in text
    selected={name:{"alpha":np.zeros((2,2),np.float32)} for name in SELECTED_IMAGES}
    output=tmp_path/"selected.npz"; _save_selected_npz(output,selected)
    with np.load(output) as payload:
        assert {key.split("__",1)[0]+".JPG" for key in payload} == {Path(name).stem+".JPG" for name in SELECTED_IMAGES}
    wrong=dict(selected); wrong["DSC00000.JPG"]={"alpha":np.zeros((1,1))}
    with pytest.raises(RuntimeError): _save_selected_npz(tmp_path/"wrong.npz",wrong)
