import torch

from v5.camera import distort_simple_radial, undistort_simple_radial
from v5.data import pixels_to_rays


def test_simple_radial_round_trip_and_gradient():
    normalized = torch.tensor(
        [[0.0, 0.0], [0.25, -0.15], [-0.55, 0.35]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radial_k = torch.tensor([-0.12, 0.08, -0.05], dtype=torch.float64)
    distorted = distort_simple_radial(normalized, radial_k)
    recovered = undistort_simple_radial(distorted, radial_k)
    assert torch.allclose(recovered, normalized, atol=1.0e-10, rtol=1.0e-10)
    recovered.square().sum().backward()
    assert normalized.grad is not None
    assert torch.isfinite(normalized.grad).all()


def test_pixels_to_rays_removes_simple_radial_distortion():
    pose = torch.eye(4, dtype=torch.float64)[:3].unsqueeze(0)
    intrinsic = torch.tensor(
        [[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    normalized = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    radial_k = torch.tensor([0.1], dtype=torch.float64)
    distorted = distort_simple_radial(normalized, radial_k)
    xs = 50.0 + 100.0 * distorted[:, 0]
    ys = 40.0 + 100.0 * distorted[:, 1]
    _, rays_d = pixels_to_rays(
        pose,
        intrinsic,
        torch.tensor([0]),
        ys,
        xs,
        radial_distortion=radial_k,
    )
    expected = torch.tensor([[0.3, 0.2, -1.0]], dtype=torch.float64)
    assert torch.allclose(rays_d, expected, atol=1.0e-10, rtol=1.0e-10)
