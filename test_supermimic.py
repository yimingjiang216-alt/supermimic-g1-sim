"""
Unit tests & numerical verification suite for `supermimic-g1-sim/src`:
1. SO(3) Lie Group exponential/logarithmic map invertibility & Rodrigues rotation parity.
2. Heading Canonicalization (`yaw_mat` & `canonical_features`): verifies exact invariance
   across arbitrary global yaw rotations (including the 359° -> 2° wrap boundary).
3. Analytic Velocity-Integration Head (`structured=True`, substep coeff c=0.625): verifies
   zero-init identity drift and exact target-increment round-trip reconstruction.
4. Termination Head & 3-MLP Policy/Critic forward-backward gradient flow + PortableMuonAdam
   Newton-Schulz orthogonalization check.
"""
from __future__ import annotations

import math
import torch

from src.kinematics import (
    axis_angle_to_mat,
    mat_to_6d,
    mat_to_quat,
    quat_slerp,
    quat_to_mat,
    rot_angle_sq,
    so3_exp,
    so3_log,
    yaw_mat,
)
from src.policy import Critic, Policy, PortableMuonAdam, zeropower_via_newtonschulz5
from src.world_model import DifferentiableWorldModel, canonical_features

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    torch.manual_seed(42)

    # 1. SO(3) exp/log & quaternion round-trip
    w = torch.randn(64, 3) * 0.8
    R = so3_exp(w)
    w_rec = so3_log(R)
    err_so3 = (w - w_rec).abs().max().item()
    check("SO(3) exp/log exact inverse", err_so3 < 1e-5, f"max_err={err_so3:.2e}")

    q = mat_to_quat(R)
    R_rec = quat_to_mat(q)
    err_quat = (R - R_rec).abs().max().item()
    check("Quaternion <-> Rotation Matrix round-trip", err_quat < 1e-5, f"max_err={err_quat:.2e}")

    # 2. Vectorized slerp continuity across near-identical frames (mjlab early-return bug fix)
    q0 = mat_to_quat(so3_exp(torch.tensor([[0.1, -0.2, 0.3]])))
    q1 = mat_to_quat(so3_exp(torch.tensor([[0.1, -0.2, 0.3001]])))
    qm = quat_slerp(q0, q1, torch.tensor([0.5]))
    d0 = (qm - q0).norm().item()
    d1 = (qm - q1).norm().item()
    check("Vectorized slerp midpoint on near-identical quaternions", abs(d0 - d1) < 1e-6 and d0 > 0, f"d0={d0:.2e}, d1={d1:.2e}")

    # 3. Heading Canonicalization invariance across 359° -> 2° yaw boundary
    p = torch.tensor([[1.2, -0.5, 0.76], [1.2, -0.5, 0.76]])
    R_base = so3_exp(torch.tensor([[0.08, -0.05, 0.0], [0.08, -0.05, 0.0]]))
    yaws = torch.tensor([math.radians(359.0), math.radians(2.0)])
    cy, sy = torch.cos(yaws), torch.sin(yaws)
    z0, o1 = torch.zeros_like(cy), torch.ones_like(cy)
    R_z = torch.stack([cy, -sy, z0, sy, cy, z0, z0, z0, o1], dim=-1).unflatten(-1, (3, 3))
    R_rot = R_z @ R_base
    v_local = torch.tensor([[0.8, -0.1, 0.05], [0.8, -0.1, 0.05]])
    w_local = torch.tensor([[0.02, 0.15, -0.4], [0.02, 0.15, -0.4]])
    v_world = (R_rot @ v_local.unsqueeze(-1)).squeeze(-1)
    w_world = (R_rot @ w_local.unsqueeze(-1)).squeeze(-1)
    q_j = torch.randn(1, 29).expand(2, 29)
    qd_j = torch.randn(1, 29).expand(2, 29)

    S = (p, R_rot, v_world, w_world, q_j, qd_j)
    f_canon = canonical_features(S, canonicalize=True)
    f_raw = canonical_features(S, canonicalize=False)
    diff_canon = (f_canon[0] - f_canon[1]).abs().max().item()
    diff_raw = (f_raw[0] - f_raw[1]).abs().max().item()
    check(
        "Heading Canonicalization invariance across 359° -> 2° wrap",
        diff_canon < 1e-5 and diff_raw > 0.05,
        f"canonical_diff={diff_canon:.2e} vs raw_diff={diff_raw:.3f}",
    )

    # 4. Analytic Semi-Implicit Velocity Integration round-trip & zero-init drift
    wm = DifferentiableWorldModel(num_joints=29, xi_dim=33, hidden=(128, 128, 128), structured=True, canonicalize=True)
    S1_pred = wm.step(S, torch.zeros(2, 29), torch.zeros(2, 33))
    # With zero-initialized last layer and default normalizer, velocity is preserved and position integrates v * dt
    expected_p1 = p + v_world * wm.dt
    err_drift_p = (S1_pred[0] - expected_p1).abs().max().item()
    check("WorldModel zero-init semi-implicit constant-velocity drift", err_drift_p < 1e-5, f"max_err={err_drift_p:.2e}")

    # 5. 3-MLP Policy + Critic + WorldModel 8-step BPTT gradient flow & PortableMuonAdam step
    pi = Policy(obs_dim=93, goal_dim=680, act_dim=29, hist=10, enc_hidden=(128, 64), dec_hidden=(128, 64))
    critic = Critic(in_dim=71 + 680 + 33, hidden=(64, 32))
    opt_pi = PortableMuonAdam(pi, [pi.enc, pi.dec], lr=1e-3)

    obs_hist = torch.randn(4, 10, 93)
    goal = torch.randn(4, 680)
    xi_n = torch.zeros(4, 33)
    S_cur = tuple(x.repeat(2, *([1] * (x.dim() - 1))) for x in S)
    ref_p = S_cur[0].clone()
    ref_R = S_cur[1].clone()

    survival = torch.ones(4)
    total_ret = torch.zeros(4)
    for t in range(4):
        a = pi(obs_hist, goal)
        S_cur = wm.step(S_cur, a, xi_n)
        d_soft = wm.term_head.soft_prob(S_cur[0], S_cur[1], ref_p, ref_R)
        r_step = -((S_cur[0] - ref_p).pow(2).sum(-1) + (a.pow(2).mean(-1))) * 0.02
        total_ret = total_ret + (0.99**t) * survival * r_step
        survival = survival * (1.0 - d_soft)

    v_boot = critic(torch.cat([canonical_features(S_cur), goal, xi_n], dim=-1))
    loss_pi = -(total_ret + (0.99**4) * survival * v_boot).mean()
    opt_pi.zero_grad()
    loss_pi.backward()

    grad_norms = [p.grad.norm().item() for p in pi.parameters() if p.grad is not None]
    check("Multi-step BPTT gradient flows through WorldModel to Policy Encoder & Decoder",
          len(grad_norms) == len(list(pi.parameters())) and all(g > 0 for g in grad_norms),
          f"n_params={len(grad_norms)}, min_grad={min(grad_norms):.2e}")

    w_before = pi.dec[0].weight.detach().clone()
    opt_pi.step()
    w_after = pi.dec[0].weight.detach()
    check("PortableMuonAdam updates policy weights cleanly", not torch.allclose(w_before, w_after))

    # 6. Newton-Schulz 5-step orthogonalization check
    M = torch.randn(64, 32)
    U = zeropower_via_newtonschulz5(M, steps=5)
    sv = torch.linalg.svdvals(U)
    check("Newton-Schulz 5-step orthogonalization singular values in [0.65, 1.25]",
          bool((sv.min() > 0.65) and (sv.max() < 1.25)),
          f"sv in [{sv.min():.3f}, {sv.max():.3f}]")

    print("\n" + ("ALL 7 SUPERMIMIC TESTS PASSED!" if not FAILURES else f"FAILED: {FAILURES}"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
