#!/usr/bin/env python3
"""GPU check that RU-PART's sparse analytic footprint matches gsplat 1.5.3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from gsplat import rasterization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.prospective_topology import project_gaussians, sparse_footprints_from_projection


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the gsplat footprint validation")
    device = torch.device("cuda")
    means = torch.tensor([[0.08, -0.04, 3.0], [-0.12, 0.10, 4.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.94, 0.1, 0.2, 0.25]], device=device)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.tensor([[0.12, 0.08, 0.06], [0.10, 0.14, 0.07]], device=device)
    viewmat = torch.eye(4, device=device)
    K = torch.tensor([[30.0, 0.0, 18.0], [0.0, 30.0, 18.0], [0.0, 0.0, 1.0]], device=device)
    ids = torch.arange(len(means), device=device)
    errors = []
    for index in range(len(means)):
        _, alpha, _ = rasterization(
            means[index:index + 1], quats[index:index + 1], scales[index:index + 1],
            torch.full((1,), 0.5, device=device), torch.zeros((1, 3), device=device),
            viewmat[None], K[None], 36, 36, packed=False, rasterize_mode="classic",
        )
        radii, means2d, _, conics = project_gaussians(
            means[index:index + 1], quats[index:index + 1], scales[index:index + 1], viewmat, K
        )
        footprint = sparse_footprints_from_projection(ids[index:index + 1], means2d, conics, radii)[0]
        analytic = torch.zeros((36, 36), device=device)
        analytic[footprint.y0:footprint.y1, footprint.x0:footprint.x1] = 0.5 * footprint.gaussian
        errors.append(float(torch.max(torch.abs(alpha[0, ..., 0] - analytic))))
    maximum = max(errors)
    result = {"gsplat_version": "1.5.3", "candidate_count": len(errors), "max_absolute_alpha_error": maximum, "formula_implementation_match": maximum <= 2e-6}
    print(json.dumps(result, indent=2))
    if not result["formula_implementation_match"]:
        print("FORMULA_IMPLEMENTATION_MISMATCH")
        return 3
    print("PURI-GS-RU-PART-PROJECTION-MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
