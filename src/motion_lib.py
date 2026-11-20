"""
LAFAN1 Reference Motion Library & Return-Based 1s Bin Curriculum Reset Sampler (5:4:1 Mix).

Implements:
- 30fps LAFAN1 G1 CSV loading + vectorized quaternion slerp resampling to 50Hz
- Central-difference root linear/angular velocity & joint velocity computation
- Precomputed 14-body 3D poses/velocities via Differentiable G1Kinematics
- K=10 future-frame relative goal vector construction (with heading-canonical dp & 6D rotation)
- Return-based 1-second bin sampling (50% uniform + 40% softmax(-25 G_b / G_max) + 10% memory)
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
import numpy as np
import torch

from .kinematics import G1Kinematics, mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat
from .world_model import CONTROL_DT, MAX_STEP_REWARD

BIN_FRAMES = 50       # 1.0 second per bin at 50Hz
GOAL_Q_OFFSET = 15    # [dp(3), rot6d(6), v_local(3), w_local(3), q(29), 0.1*qd(29)]


def _central_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


class MotionLibrary:
    """Multi-clip 50Hz reference motion library with return-based 1s bin curriculum sampling."""

    def __init__(self, csv_files: list[str | Path], kin: G1Kinematics, device: str = "cpu"):
        self.device = device
        self.names = [Path(f).stem for f in csv_files]
        ps, qs, js, lens = [], [], [], []

        for f in csv_files:
            raw = np.loadtxt(f, delimiter=",")
            p = torch.tensor(raw[:, 0:3], dtype=torch.float32)
            q_xyzw = raw[:, 3:7]
            quat = torch.tensor(np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=-1), dtype=torch.float32)
            j = torch.tensor(raw[:, 7:36], dtype=torch.float32)
            src_fps = 30.0
            T_src = p.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            alpha = (t - i0.float()).clamp(0.0, 1.0)
            p_50 = p[i0] * (1.0 - alpha[:, None]) + p[i0 + 1] * alpha[:, None]
            j_50 = j[i0] * (1.0 - alpha[:, None]) + j[i0 + 1] * alpha[:, None]
            q_50 = quat_slerp(quat[i0], quat[i0 + 1], alpha)
            ps.append(p_50)
            qs.append(q_50)
            js.append(j_50)
            lens.append(n)

        self.clip_len = torch.tensor(lens, dtype=torch.long, device=device)
        self.clip_start = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long)]), dim=0
        ).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _central_diff(p, CONTROL_DT)
            qd = _central_diff(j, CONTROL_DT)
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2.0 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p)
            root_R.append(R)
            root_v.append(v)
            root_w.append(w)
            jq.append(j)
            jqd.append(qd)

        cat = lambda xs: torch.cat(xs, dim=0).to(device)
        self.root_p = cat(root_p)
        self.root_R = cat(root_R)
        self.root_v = cat(root_v)
        self.root_w = cat(root_w)
        self.q = cat(jq)
        self.qd = cat(jqd)

        with torch.no_grad():
            self.body_p, self.body_R, self.body_v, self.body_w = kin.forward(
                self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd
            )

        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)

        # 1-second bins for 5:4:1 Return-Based Curriculum Sampling
        bin_clip, bin_start = [], []
        for c, length in enumerate(lens):
            nb = int(np.ceil(length / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, dtype=torch.long, device=device)
        self.bin_start = torch.tensor(bin_start, dtype=torch.long, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    def goal_dim(self, K: int = 10) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.q.shape[-1])

    def goal(self, f: torch.Tensor, K: int = 10) -> torch.Tensor:
        """Construct K-frame future goal vector in the reference heading frame."""
        end = self.clip_end[self.clip_id[f]]
        fut = torch.minimum(f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device), end.unsqueeze(-1))
        H = yaw_mat(self.root_R[f])
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat([dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1], dim=-1)
        return feats.reshape(f.shape[0], -1)

    def sample_curriculum_bins(self, n: int, bin_G: torch.Tensor, bin_w_avg: torch.Tensor, gamma: float = 0.99) -> tuple[torch.Tensor, torch.Tensor]:
        """
        5:4:1 Return-Based Reset Sampling (SuperMimic Appendix D.1):
          - 50% uniform across all 1s bins
          - 40% proportional to softmax(-25 * G_b / G_max) focusing on hard low-return bins
          - 10% historical average softmax weights to prevent catastrophic forgetting
        """
        g_max = MAX_STEP_REWARD / (1.0 - gamma)
        soft = torch.softmax(-25.0 * bin_G / g_max, dim=0)
        bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / self.num_bins + 0.4 * soft + 0.1 * (bin_w_avg / bin_w_avg.sum().clamp(min=1e-8))
        bins = torch.multinomial(probs, n, replacement=True)
        start = self.bin_start[bins]
        end = torch.minimum(start + BIN_FRAMES, self.clip_end[self.bin_clip[bins]])
        frames = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return frames, bins
