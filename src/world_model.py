"""
SuperMimic Differentiable World Model:
1. Heading Canonicalization (eliminating 359° -> 2° yaw discontinuity)
2. 3-Layer LayerNorm-MLP Dynamics (8.82M params at 2048 width, zero-init residual head)
3. Two Heads:
   - Analytic Head: Semi-implicit velocity integration (c = 0.625) + Differentiable FK
     for exact 14-body 3D poses, 8-term tracking rewards, and proprioceptive observations.
   - Termination Head: Learned 64x64 BCE head & Analytic Soft-Sigmoid gate for
     survival-weighted imagination rollouts c_{t+1} = c_t * (1 - d_{t+1}).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kinematics import G1Kinematics, mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, so3_exp, so3_log, yaw_mat

CONTROL_DT = 0.02
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT  # 0.1 per step


class RunningNorm(nn.Module):
    """In-place running mean/variance normalizer (no host-device sync)."""

    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        if n == 0:
            return
        m = x.mean(dim=0)
        v = x.var(dim=0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        new_mean = self.mean + delta * (n / tot)
        new_var = (self.var * self.count + v * n + delta.pow(2) * self.count * n / tot) / tot
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps * self.eps)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps * self.eps)


def build_ln_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, zero_last: bool = False) -> nn.Sequential:
    """LayerNorm-MLP (LNP): Linear -> LayerNorm -> SiLU -> ... -> Linear."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.extend([nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()])
        d = h
    last = nn.Linear(d, out_dim)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


def canonical_features(S: tuple[torch.Tensor, ...], canonicalize: bool = True) -> torch.Tensor:
    """
    Heading-normalized state features:
      - root height z (1)
      - 6D tilt in heading frame H^T R (6)
      - body-frame linear velocity R^T v (3)
      - body-frame angular velocity R^T w (3)
      - joint positions q (J) and velocities qd (J)
    If canonicalize=False (ablation), raw world-frame orientation and velocities are used.
    """
    p, R, v, w, q, qd = S
    if not canonicalize:
        return torch.cat([p[..., 2:3], mat_to_6d(R), v, w, q, qd], dim=-1)
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    tilt_6d = mat_to_6d(H.transpose(-1, -2) @ R)
    v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
    w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
    return torch.cat([p[..., 2:3], tilt_6d, v_b, w_b, q, qd], dim=-1)


class TerminationHead(nn.Module):
    """
    Learned 2-layer Termination Head predicting P(terminated | height_err, tilt_err)
    plus analytic soft-sigmoid termination probability for differentiable rollouts.
    """

    def __init__(self, hidden: tuple[int, ...] = (64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.height_thr = height_thr
        self.tilt_thr = tilt_thr
        self.net = build_ln_mlp(4, hidden, 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def features(self, p: torch.Tensor, R: torch.Tensor, ref_p: torch.Tensor, ref_R: torch.Tensor) -> torch.Tensor:
        e_z = ref_p[..., 2] - p[..., 2]
        e_tilt = ref_R[..., 2, 2] - R[..., 2, 2]
        return torch.stack([e_z, e_tilt], dim=-1)

    def forward(self, p: torch.Tensor, R: torch.Tensor, ref_p: torch.Tensor, ref_R: torch.Tensor) -> torch.Tensor:
        """Return termination logits (...,)."""
        feats = self.features(p, R, ref_p, ref_R) * self.scale
        x = torch.cat([feats, feats.abs()], dim=-1)
        return self.net(x).squeeze(-1)

    def soft_prob(self, p: torch.Tensor, R: torch.Tensor, ref_p: torch.Tensor, ref_R: torch.Tensor) -> torch.Tensor:
        """Differentiable soft termination probability d_hat in [0, 1]."""
        e_z = (ref_p[..., 2] - p[..., 2]).abs()
        e_tilt = (ref_R[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1.0 - torch.sigmoid((e_z - self.height_thr) / 0.02)) * (
            1.0 - torch.sigmoid((e_tilt - self.tilt_thr) / 0.04)
        )
        return 1.0 - alive


class DifferentiableWorldModel(nn.Module):
    """
    SuperMimic Differentiable World Model with:
      - Heading Canonicalization (`canonicalize=True`)
      - Analytic Semi-Implicit Velocity Integration Head (`structured=True`, substep coeff c=0.625)
      - Learned & Analytic Termination Head
    """

    DRIFT_W = 0.625  # (decimation + 1) / (2 * decimation) for 4 physics substeps

    def __init__(
        self,
        num_joints: int = 29,
        xi_dim: int = 33,
        hidden: tuple[int, ...] = (2048, 2048, 2048),
        structured: bool = True,
        canonicalize: bool = True,
        dt: float = CONTROL_DT,
    ):
        super().__init__()
        self.num_j = num_joints
        self.xi_dim = xi_dim
        self.structured = structured
        self.canonicalize = canonicalize
        self.dt = dt

        self.in_dim = (1 + 6 + 3 + 3 + 2 * num_joints) + num_joints + xi_dim
        self.out_dim = 3 + 3 + 3 + 3 + 2 * num_joints  # dp(3), dth(3), dv(3), dw(3), dq(J), dqd(J)

        self.net = build_ln_mlp(self.in_dim, hidden, self.out_dim, zero_last=True)
        self.term_head = TerminationHead()
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim)
        self.inc_norm = RunningNorm(self.out_dim)

    def _drift(self, S: tuple[torch.Tensor, ...], v1: torch.Tensor, w1: torch.Tensor, qd1: torch.Tensor):
        """Semi-implicit trapezoidal position/orientation drift from old and new velocities."""
        p, R, v, w, q, qd = S
        a = self.dt * (1.0 - self.DRIFT_W)
        b = self.dt * self.DRIFT_W
        if self.canonicalize:
            H = yaw_mat(R)
            gp = (H.transpose(-1, -2) @ (a * v + b * v1).unsqueeze(-1)).squeeze(-1)
            gth = (R.transpose(-1, -2) @ (a * w + b * w1).unsqueeze(-1)).squeeze(-1)
        else:
            gp = a * v + b * v1
            gth = a * w + b * w1
        gq = a * qd + b * qd1
        return gp, gth, gq

    def increments(self, S0: tuple[torch.Tensor, ...], S1: tuple[torch.Tensor, ...]) -> torch.Tensor:
        p0, R0, v0, w0, q0, qd0 = S0
        p1, R1, v1, w1, q1, qd1 = S1
        if self.canonicalize:
            H = yaw_mat(R0)
            Rt = R0.transpose(-1, -2)
            dp = (H.transpose(-1, -2) @ (p1 - p0).unsqueeze(-1)).squeeze(-1)
            dth = so3_log(Rt @ R1)
            dv = (Rt @ (v1 - v0).unsqueeze(-1)).squeeze(-1)
            dw = (Rt @ (w1 - w0).unsqueeze(-1)).squeeze(-1)
        else:
            dp = p1 - p0
            dth = so3_log(R0.transpose(-1, -2) @ R1)
            dv = v1 - v0
            dw = w1 - w0
        return torch.cat([dp, dth, dv, dw, q1 - q0, qd1 - qd0], dim=-1)

    def targets(self, S0: tuple[torch.Tensor, ...], S1: tuple[torch.Tensor, ...]) -> torch.Tensor:
        inc = self.increments(S0, S1)
        if not self.structured:
            return inc
        J = self.num_j
        dp, dth, dv, dw, dq, dqd = inc.split((3, 3, 3, 3, J, J), dim=-1)
        gp, gth, gq = self._drift(S0, S1[2], S1[3], S1[5])
        return torch.cat([dp - gp, dth - gth, dv, dw, dq - gq, dqd], dim=-1)

    def step(self, S: tuple[torch.Tensor, ...], a: torch.Tensor, xi_n: torch.Tensor) -> tuple[torch.Tensor, ...]:
        p, R, v, w, q, qd = S
        x = torch.cat([canonical_features(S, self.canonicalize), a, xi_n], dim=-1)
        raw = self.net(self.in_norm(x))
        d = raw * self.out_norm.std() + self.out_norm.mean
        J = self.num_j
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, J, J), dim=-1)

        if self.canonicalize:
            v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
            w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        else:
            v1 = v + dv
            w1 = w + dw
        qd1 = qd + dqd

        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + gp, dth + gth, dq + gq

        if self.canonicalize:
            H = yaw_mat(R)
            p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        else:
            p1 = p + dp
        R1 = R @ so3_exp(dth)
        q1 = q + dq
        return (p1, R1, v1, w1, q1, qd1)
