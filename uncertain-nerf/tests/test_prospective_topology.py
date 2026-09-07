import copy
from types import SimpleNamespace

import numpy as np
import torch

from puri_gs.prospective_topology import (
    SparseFootprint,
    alpha_increment,
    append_birth_rows,
    benefit_harm,
    celf_acceptance_order,
    compute_default_grow_masks,
    exhaustive_acceptance_order,
    initialize_birth_geometry,
    projection_jacobian,
    initialize_footprint_scores,
    pair_births_with_clones,
    rotation_matrix_to_wxyz,
    support_increment,
    validate_parameter_state_shapes,
)


def footprint(candidate_id, y0, x0, values):
    values = torch.as_tensor(values, dtype=torch.float32)
    return SparseFootprint(candidate_id, y0, y0 + values.shape[0], x0, x0 + values.shape[1], values)


def test_alpha_and_support_product_identities_and_finite_difference():
    torch.manual_seed(4)
    A, F, G = (torch.rand(8, dtype=torch.float64) for _ in range(3))
    opacity = 0.1
    assert torch.allclose(1 - (1 - A) * (1 - opacity * G) - A, alpha_increment(A, opacity, G))
    assert torch.allclose(1 - (1 - F) * (1 - 0.5 * G) - F, support_increment(F, G))
    epsilon = 1e-7
    plus = 1 - (1 - A) * (1 - epsilon * G)
    derivative = (plus - A) / epsilon
    assert torch.max(torch.abs(derivative - (1 - A) * G)) < 1e-8


def test_benefit_harm_partition_and_monotonicity():
    support = torch.zeros(4, 4)
    evidence = torch.zeros(4, 4); evidence[:, :2] = 1
    item = footprint(0, 0, 0, torch.ones(4, 4))
    initialize_footprint_scores([item], support, support, evidence, initial_opacity=0.1)
    b0, h0 = benefit_harm(support, evidence, item)
    total = support_increment(support, item.gaussian).mean()
    assert torch.allclose(b0 + h0, total)
    updated = support.clone(); updated[:, :2] = 0.5
    b1, h1 = benefit_harm(updated, evidence, item)
    assert b1 <= b0 and h1 == h0
    assert float(b1 - h1) <= float(b0 - h0)


def test_celf_matches_exhaustive_with_overlapping_tiles():
    support = torch.zeros(5, 5)
    evidence = torch.ones(5, 5)
    items = [
        footprint(2, 0, 0, torch.ones(3, 3)),
        footprint(1, 1, 1, torch.full((3, 3), 0.8)),
        footprint(3, 3, 3, torch.ones(2, 2)),
    ]
    initialize_footprint_scores(items, support, support, evidence, initial_opacity=0.1)
    assert celf_acceptance_order(items, support, evidence, 3) == exhaustive_acceptance_order(items, support, evidence, 3)


def test_birth_must_pareto_dominate_replaced_clone():
    support = torch.zeros(4, 4)
    evidence = torch.zeros(4, 4); evidence[:, :3] = 1
    birth = footprint(10, 0, 0, torch.ones(4, 3))
    clone = footprint(7, 0, 2, torch.full((4, 2), 0.4))
    initialize_footprint_scores([birth, clone], support, support, evidence, initial_opacity=0.1)
    replacements, _ = pair_births_with_clones([birth], [clone], support, evidence, limit=1)
    assert len(replacements) == 1
    chosen = replacements[0]
    assert chosen.benefit > chosen.clone_benefit
    assert chosen.harm <= chosen.clone_harm


def test_default_grow_masks_are_pure_and_match_formula():
    params = {"scales": torch.nn.Parameter(torch.log(torch.tensor([[0.005, 0.005, 0.005], [0.02, 0.02, 0.02]])))}
    state = {"grad2d": torch.tensor([1.0, 1.0]), "count": torch.tensor([1000.0, 1000.0]), "scene_scale": 1.0}
    strategy = SimpleNamespace(grow_grad2d=0.0006, grow_scale3d=0.01, refine_scale2d_stop_iter=0, grow_scale2d=0.05)
    before_params, before_state = copy.deepcopy(params), copy.deepcopy(state)
    masks = compute_default_grow_masks(params, state, strategy, 10000)
    assert masks.clone.tolist() == [True, False]
    assert masks.split.tolist() == [False, True]
    assert torch.equal(params["scales"], before_params["scales"])
    assert torch.equal(state["grad2d"], before_state["grad2d"])


def test_pure_grow_masks_match_real_gsplat_golden_action_counts():
    from gsplat.strategy import DefaultStrategy

    strategy = DefaultStrategy(absgrad=True, grow_grad2d=0.0006, grow_scale3d=0.01)
    shapes = {"means": (3,), "scales": (3,), "quats": (4,), "opacities": (), "sh0": (1, 3), "shN": (15, 3)}
    params = {name: torch.nn.Parameter(torch.zeros((2, *shape))) for name, shape in shapes.items()}
    params["scales"].data[:] = torch.log(torch.tensor([[0.005] * 3, [0.02] * 3]))
    optimizers = {name: torch.optim.Adam([param], lr=1e-3) for name, param in params.items()}
    state = {"grad2d": torch.tensor([1.0, 1.0]), "count": torch.tensor([1000.0, 1000.0]), "scene_scale": 1.0}
    masks = compute_default_grow_masks(params, state, strategy, 10000)
    golden_clone, golden_split = strategy._grow_gs(params, optimizers, state, 10000)
    assert golden_clone == int(masks.clone.sum()) == 1
    assert golden_split == int(masks.split.sum()) == 1


def test_birth_geometry_targets_half_patch_and_is_finite():
    point = np.array([0.0, 0.0, 5.0])
    c2ws, Ks = [], []
    for center in ((-1, 0, 0), (1, 0, 0), (0, -1, 0)):
        matrix = np.eye(4); matrix[:3, 3] = center; c2ws.append(matrix)
        Ks.append(np.array([[30.0, 0, 18.0], [0, 30.0, 18.0], [0, 0, 1.0]]))
    scales, quat = initialize_birth_geometry(point, c2ws, Ks)
    assert np.isfinite(scales).all() and np.isfinite(quat).all()
    assert np.isclose(np.linalg.norm(quat), 1.0)
    from gsplat.utils import normalized_quat_to_rotmat
    rotation = normalized_quat_to_rotmat(torch.from_numpy(quat)[None])[0].numpy()
    covariance = rotation @ np.diag(np.exp(scales) ** 2) @ rotation.T
    projected_radii = [
        np.sqrt(np.linalg.eigvalsh(projection_jacobian(point, c2w, K) @ covariance @ projection_jacobian(point, c2w, K).T).max())
        for c2w, K in zip(c2ws, Ks)
    ]
    assert np.isclose(np.median(projected_radii), 0.5, atol=1e-5)


def test_wxyz_quaternion_round_trips_gsplat_rotation_convention():
    from gsplat.utils import normalized_quat_to_rotmat

    source = torch.tensor([[0.3, -0.2, 0.7, 0.6]], dtype=torch.float64)
    source = torch.nn.functional.normalize(source, dim=-1)
    rotation = normalized_quat_to_rotmat(source)[0].numpy()
    converted = torch.from_numpy(rotation_matrix_to_wxyz(rotation))[None]
    recovered = normalized_quat_to_rotmat(converted)[0].numpy()
    assert np.allclose(recovered, rotation, atol=1e-10)


def test_append_birth_rows_zeroes_optimizer_and_strategy_state():
    shapes = {"means": (3,), "scales": (3,), "quats": (4,), "opacities": (), "sh0": (1, 3), "shN": (15, 3)}
    params = {name: torch.nn.Parameter(torch.zeros((2, *shape))) for name, shape in shapes.items()}
    optimizers = {name: torch.optim.Adam([param], lr=1e-3) for name, param in params.items()}
    loss = sum(value.square().sum() for value in params.values()); loss.backward()
    for optimizer in optimizers.values(): optimizer.step(); optimizer.zero_grad(set_to_none=True)
    state = {"grad2d": torch.ones(2), "count": torch.ones(2), "scene_scale": 1.0}
    rows = {name: torch.ones((1, *shape)) for name, shape in shapes.items()}
    append_birth_rows(params, optimizers, state, rows)
    validate_parameter_state_shapes(params, optimizers, state)
    assert len(params["means"]) == 3
    assert state["grad2d"][-1] == 0 and state["count"][-1] == 0
    for name, optimizer in optimizers.items():
        for key, value in optimizer.state[params[name]].items():
            if key != "step": assert torch.count_nonzero(value[-1]) == 0
