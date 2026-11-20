"""
SuperMimic 3-MLP Policy & Critic Architecture + Pure-PyTorch Muon Optimizer:
- GoalEncoder MLP: 10 future reference frames -> 32-D tanh motion identity latent z
- PolicyDecoder MLP: latent z + 10-step proprioceptive history -> 29-D joint targets
  (with optional Reference-Action Skip connection)
- Critic MLP: (256, 128) value network trained with TD(lambda=0.95) GAE targets
- MuonAdam: Newton-Schulz 5-step orthogonalization on hidden weight matrices
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .world_model import RunningNorm, build_ln_mlp


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Works on any PyTorch 2.x version and any CUDA compute capability (including T4 SM 7.5).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class PortableMuonAdam:
    """
    Muon on hidden 2D weight matrices of MLPs (rescaled to AdamW RMS),
    Adam on input/output layers and bias vectors. Compatible with all PyTorch versions.
    """

    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float = 2e-4, betas=(0.9, 0.95)):
        hidden_ids = set()
        for m in mlps:
            linears = [x for x in m.modules() if isinstance(x, nn.Linear)]
            for l in linears[1:-1]:
                hidden_ids.add(id(l.weight))
        self.muon_params = [p for p in model.parameters() if id(p) in hidden_ids and p.requires_grad]
        self.adam_params = [p for p in model.parameters() if id(p) not in hidden_ids and p.requires_grad]
        self.lr = lr
        self.momentum = 0.95
        self.state = {p: torch.zeros_like(p) for p in self.muon_params}
        self.adam = torch.optim.Adam(self.adam_params, lr=lr, betas=betas)

    @property
    def param_groups(self):
        return self.adam.param_groups

    def set_lr(self, lr: float):
        self.lr = lr
        for g in self.adam.param_groups:
            g["lr"] = lr

    def zero_grad(self, set_to_none: bool = True):
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        for p in self.muon_params:
            if p.grad is None:
                continue
            buf = self.state[p]
            buf.lerp_(p.grad, 1.0 - self.momentum)
            g = p.grad.lerp(buf, self.momentum)
            u = zeropower_via_newtonschulz5(g, steps=5)
            # Match AdamW RMS update scale: 0.2 * sqrt(max(A, B))
            scale = 0.2 * (max(p.shape[-2], p.shape[-1]) ** 0.5)
            p.add_(u, alpha=-self.lr * scale)
        self.adam.step()


class Policy(nn.Module):
    """
    Goal-Conditioned Motion Tracking Policy:
      z = tanh(GoalEncoder(goal_norm(g_t))) in R^32
      a_t = PolicyDecoder([z, obs_norm(o_{t-H+1:t})]) + optional ref_skip
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int = 29,
        hist: int = 10,
        z_dim: int = 32,
        enc_hidden: tuple[int, ...] = (2048, 1024, 512, 256),
        dec_hidden: tuple[int, ...] = (2048, 1024, 512, 256),
        ref_skip: tuple[int, torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.hist = hist
        self.ref_off = None
        if ref_skip is not None:
            self.ref_off = ref_skip[0]
            self.register_buffer("ref_q0", ref_skip[1].clone().float())
            self.register_buffer("ref_alpha", ref_skip[2].clone().float())

        self.enc = build_ln_mlp(goal_dim, enc_hidden, z_dim)
        self.dec = build_ln_mlp(z_dim + obs_dim * hist, dec_hidden, act_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()

        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def encode(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.enc(self.goal_norm(goal)))

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        o = self.obs_norm(obs_hist).flatten(-2)
        z = self.encode(goal)
        a = self.dec(torch.cat([z, o], dim=-1))
        if self.ref_off is not None:
            q_ref = goal[..., self.ref_off : self.ref_off + a.shape[-1]]
            a = a + (q_ref - self.ref_q0) / self.ref_alpha
        return a


class Critic(nn.Module):
    """Compact 2-layer Value Network V_psi(s_t, g_t, xi) -> scalar."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 128)):
        super().__init__()
        self.norm = RunningNorm(in_dim)
        self.net = build_ln_mlp(in_dim, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)
