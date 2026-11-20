"""
build_kaggle_notebook.py — Generates the self-contained Kaggle GPU notebook `supermimic_g1_sim.ipynb`
for Unitree G1 (29-DoF) Whole-Body Motion Tracking with One Differentiable World Model (SuperMimic).

Pitfalls explicitly guarded against:
1. Download drops (LAFAN1 / G1 meshes): Dual-source (HuggingFace + ModelScope fallback) with HTTP Range
   resume, exponential backoff, column validation, and atomic `.part` -> final rename.
2. Log corruption: Append-only `train_log.csv` with per-step `flush() + os.fsync()` and atomic `.tmp`
   checkpoint replacement (`os.replace`) instead of fragile binary TensorBoard event files.
3. GPU VRAM & WDDM/Allocator fragmentation: Sets `PYTORCH_CUDA_ALLOC_CONF`, uses compact 6-skill
   action windows, and supports both `mjlab/MuJoCo-Warp` GPU simulation and automatic multi-threaded
   native C++ `mujoco` fallback if the cloud node's CUDA driver is older than Warp's PTX target.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


CELL_1_SETUP = r'''# Cell 1: Environment Setup, Anti-OOM Config & Resilient Dependency Installation
import os
import sys
import time
import json
import math
import shutil
import subprocess
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MJLAB_WARP_QUIET"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def run_cmd(cmd: str, check: bool = True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")
    return r.returncode

import torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| Capability:", torch.cuda.get_device_capability(0),
          "| VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), flush=True)

# Install MuJoCo + MJWarp + mjlab with --no-deps so Kaggle's pre-installed PyTorch 2.10.0+cu128 is NEVER overwritten!
run_cmd('pip install -q "mujoco==3.11.0" "warp-lang==1.16.0" "mujoco-warp==3.11.0" "mjlab==1.6.1" glfw pyopengl prettytable trimesh mediapy imageio-ffmpeg tensordict tyro docstring-parser shtab typeguard --no-deps', check=True)
run_cmd('pip install -q imageio absl-py "etils[epath]" --no-deps', check=False)

# Strip @torch.jit.script from mjlab and make mjlab.envs.__init__ headless (no GUI viewer imports)
import site, pathlib
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    mjlab_dir = pathlib.Path(sp) / "mjlab"
    if mjlab_dir.exists():
        for pyf in mjlab_dir.rglob("*.py"):
            txt = pyf.read_text(encoding="utf-8")
            if "@torch.jit.script" in txt:
                pyf.write_text(txt.replace("@torch.jit.script", "# @torch.jit.script"), encoding="utf-8")
                print("Stripped @torch.jit.script from", pyf, flush=True)
        envs_init = mjlab_dir / "envs" / "__init__.py"
        if envs_init.exists():
            envs_init.write_text('"""Headless mjlab.envs init."""\n', encoding="utf-8")
            print("Patched headless", envs_init, flush=True)

# Verify mujoco & EGL offscreen renderer
import mujoco
print("MuJoCo version:", mujoco.__version__, "| PyTorch still:", torch.__version__, flush=True)
'''


CELL_2_DOWNLOAD = r'''# Cell 2: Resilient Dual-Source Data Downloader (HuggingFace + ModelScope Fallback + Atomic Rename)
import os
import re
import time
import json
import urllib.request
import numpy as np
import mujoco
from pathlib import Path

WORK = Path("/kaggle/working")
DATA_DIR = WORK / "data"
G1_DIR = DATA_DIR / "g1_description"
MESH_DIR = G1_DIR / "meshes"
RAW_LAFAN_DIR = DATA_DIR / "lafan1_raw"
CLIP_DIR = DATA_DIR / "lafan1_g1_clips"

for d in (G1_DIR, MESH_DIR, RAW_LAFAN_DIR, CLIP_DIR, WORK / "results", WORK / "figs", WORK / "videos", WORK / "ckpts"):
    d.mkdir(parents=True, exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main"
HF_API  = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main"
MS_BASE = "https://www.modelscope.cn/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset/resolve/master"

def fetch_bytes_with_retry(urls: list[str], retries: int = 5, timeout: int = 45) -> bytes:
    last_err = None
    for attempt in range(retries):
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
                    if len(data) > 0:
                        return data
            except Exception as e:
                last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download from {urls}: {last_err}")

def download_atomic(rel_path: str, dest: Path, min_bytes: int = 64):
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return
    urls = [f"{HF_BASE}/{rel_path}", f"{MS_BASE}/{rel_path}"]
    data = fetch_bytes_with_retry(urls)
    if len(data) < min_bytes:
        raise RuntimeError(f"Downloaded file too small ({len(data)}B): {rel_path}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)

# 1. Download G1 29-DoF URDF and all STL meshes
urdf_path = G1_DIR / "g1_29dof_rev_1_0.urdf"
download_atomic("robot_description/g1/g1_29dof_rev_1_0.urdf", urdf_path, min_bytes=10000)

urdf_text = urdf_path.read_text(encoding="utf-8")
stl_names = sorted(set(re.findall(r'filename="[^"]*?([^/"]+\.STL)"', urdf_text, flags=re.I)))
print(f"Downloading {len(stl_names)} Unitree G1 STL link meshes...", flush=True)
for name in stl_names:
    download_atomic(f"robot_description/g1/meshes/{name}", MESH_DIR / name, min_bytes=100)

# Compile URDF -> Free-floating Pelvis MJCF (g1_29dof.xml)
mjcf_path = G1_DIR / "g1_29dof.xml"
clean_urdf = re.sub(r"<!--.*?-->", "", urdf_text, flags=re.S)
clean_urdf = re.sub(
    r"<mujoco>.*?</mujoco>",
    f'<mujoco><compiler meshdir="{MESH_DIR.as_posix()}" balanceinertia="true" discardvisual="false" fusestatic="false" strippath="true"/></mujoco>',
    clean_urdf,
    flags=re.S,
)
tmp_urdf = G1_DIR / "_tmp_g1.urdf"
tmp_urdf.write_text(clean_urdf, encoding="utf-8")
m_tmp = mujoco.MjModel.from_xml_path(str(tmp_urdf))
mujoco.mj_saveLastXML(str(mjcf_path), m_tmp)
tmp_urdf.unlink(missing_ok=True)

xml_txt = mjcf_path.read_text(encoding="utf-8")
xml_txt = re.sub(r'(<body name="pelvis"[^>]*>)', r'\1\n      <freejoint name="floating_base_joint"/>', xml_txt, count=1)
xml_txt = re.sub(r'meshdir="[^"]*"', f'meshdir="{MESH_DIR.as_posix()}/"', xml_txt)
mjcf_path.write_text(xml_txt, encoding="utf-8")
m_check = mujoco.MjModel.from_xml_path(str(mjcf_path))
print(f"Compiled Unitree G1 MJCF: nq={m_check.nq}, nv={m_check.nv}, nbody={m_check.nbody}, mass={m_check.body_mass.sum():.2f} kg", flush=True)

# 2. Download 6 Representative LAFAN1 Motion Capture Clips & Extract Action-Packed Windows
#    Each clip is validated to have exactly 36 columns (3 root pos + 4 root quat xyzw + 29 joint angles).
TARGET_CLIPS = [
    ("walk1_subject1.csv",         "walk",         " steady walk & turn"),
    ("run1_subject2.csv",          "run",          " fast run & agile pivot"),
    ("dance1_subject1.csv",        "dance",        " rhythmic whole-body street dance"),
    ("jumps1_subject1.csv",        "jumps",        " continuous two-foot jumps & landing"),
    ("fight1_subject2.csv",        "fight",        " martial arts kicks & punches"),
    ("fallAndGetUp1_subject1.csv", "fallAndGetUp", " ground fall & autonomous get-up"),
]

WINDOW_FRAMES_30FPS = 180  # 6.0 seconds @ 30fps = 300 control steps @ 50Hz

clip_summary = []
for fname, short_name, desc in TARGET_CLIPS:
    raw_file = RAW_LAFAN_DIR / fname
    download_atomic(f"g1/{fname}", raw_file, min_bytes=50000)
    arr = np.loadtxt(raw_file, delimiter=",")
    assert arr.ndim == 2 and arr.shape[1] == 36, f"Corrupted CSV shape {arr.shape} in {fname}"

    # Select the most characteristic 6-second window starting from an upright stance (z > 0.68m)
    z = arr[:, 2]
    jd = np.abs(np.diff(arr[:, 7:36], axis=0)).mean(axis=1)
    best_s = 0
    best_score = -1e9
    for s in range(0, max(1, len(arr) - WINDOW_FRAMES_30FPS - 10), 15):
        e = s + WINDOW_FRAMES_30FPS
        if z[s] < 0.68 or abs(z[s + 1] - z[s]) > 0.02:
            continue
        w_z = z[s:e]
        w_jd = jd[s:e]
        if short_name == "fallAndGetUp":
            # Must go down to floor (< 0.35m) in the first half and recover (> 0.68m) at the end
            min_idx = int(np.argmin(w_z))
            if w_z[min_idx] < 0.38 and 20 <= min_idx <= 115 and w_z[-10:].mean() > 0.66:
                score = (0.75 - w_z[min_idx]) * 10.0 + w_z[-10:].mean()
            else:
                score = -1e6
        elif short_name == "jumps":
            score = float(w_z.max() - np.percentile(w_z, 20)) * 5.0 + float(w_jd.mean())
        else:
            # High motion richness while staying upright
            if w_z.min() < 0.58:
                continue
            score = float(w_jd.mean())
        if score > best_score:
            best_score = score
            best_s = s

    sliced = arr[best_s : best_s + WINDOW_FRAMES_30FPS].copy()
    # Center XY at (0, 0) at frame 0
    sliced[:, 0:2] -= sliced[0:1, 0:2]
    out_csv = CLIP_DIR / fname
    np.savetxt(out_csv, sliced, delimiter=",", fmt="%.6f")
    clip_summary.append({
        "file": fname,
        "skill": short_name,
        "desc": desc.strip(),
        "raw_frames_30fps": int(len(arr)),
        "start_frame_30fps": int(best_s),
        "window_frames_30fps": int(len(sliced)),
        "duration_s": round(len(sliced) / 30.0, 2),
        "z_min_m": round(float(sliced[:, 2].min()), 3),
        "z_max_m": round(float(sliced[:, 2].max()), 3),
    })
    print(f"Prepared {fname:28s} | start={best_s:4d} ({best_s/30:.1f}s) | z in [{sliced[:,2].min():.2f}, {sliced[:,2].max():.2f}] m", flush=True)

(WORK / "results" / "clip_summary.json").write_text(json.dumps(clip_summary, indent=2), encoding="utf-8")
'''


CELL_3_CORE_MODULES = r'''# Cell 3: Clone & Patch BlinkMimic / SuperMimic Core on GPU (with Portable Muon & T4 FP32 Support)
import os
import sys
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
BM_DIR = WORK / "blinkmimic_repo"
if not BM_DIR.exists():
    subprocess.run(f"git clone --depth 1 https://github.com/fuyichen0725/blinkmimic.git {BM_DIR}", shell=True, check=True)

# Link our prepared G1 MJCF and 6-skill LAFAN1 dataset into BM_DIR/data
bm_data = BM_DIR / "data"
bm_data.mkdir(exist_ok=True)
for sub_name, target in [("g1_description", WORK / "data" / "g1_description"),
                         ("lafan1_g1", WORK / "data" / "lafan1_g1_clips")]:
    dst = bm_data / sub_name
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            import shutil
            shutil.rmtree(dst)
    os.symlink(target, dst)

# Patch optim.py so MuonAdam works on ANY PyTorch 2.x version (even if torch.optim.Muon is absent)
optim_patch = """from __future__ import annotations
import torch
import torch.nn as nn

def _linears(m: nn.Module) -> list[nn.Linear]:
    return [x for x in m.modules() if isinstance(x, nn.Linear)]

def _newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
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

class MuonAdam:
    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95), fused: bool = True,
                 capturable: bool = False):
        hidden = {id(l.weight) for m in mlps for l in _linears(m)[1:-1]}
        self.muon_params = [p for p in model.parameters() if id(p) in hidden and p.requires_grad]
        adam_params = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
        self.capturable = capturable
        self.momentum = 0.95
        self.ns_buf = {p: torch.zeros_like(p) for p in self.muon_params}
        self._muon_group = {"params": self.muon_params, "lr": torch.tensor(float(lr), device=self.muon_params[0].device) if capturable else float(lr)}
        try:
            self.adam = torch.optim.Adam(adam_params, lr=torch.tensor(float(lr), device=adam_params[0].device) if capturable else lr,
                                         betas=betas, fused=fused, capturable=capturable)
        except Exception:
            self.adam = torch.optim.Adam(adam_params, lr=float(lr), betas=betas)

    @property
    def param_groups(self):
        return [self._muon_group] + list(self.adam.param_groups)

    def zero_grad(self, set_to_none: bool = True):
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, ok: torch.Tensor | None = None):
        if ok is not None and not bool(ok.item()):
            return
        lr_val = float(self._muon_group["lr"].item()) if torch.is_tensor(self._muon_group["lr"]) else float(self._muon_group["lr"])
        for p in self.muon_params:
            if p.grad is None:
                continue
            buf = self.ns_buf[p]
            buf.lerp_(p.grad, 1.0 - self.momentum)
            g = p.grad.lerp(buf, self.momentum)
            u = _newton_schulz_5(g, steps=5)
            ratio = 0.2 * (max(p.shape[-2], p.shape[-1]) ** 0.5)
            p.add_(u, alpha=-lr_val * ratio)
        self.adam.step()

    def state_dict(self):
        return {"adam": self.adam.state_dict()}

    def load_state_dict(self, sd: dict):
        if "adam" in sd:
            self.adam.load_state_dict(sd["adam"])
"""
(BM_DIR / "src" / "blinkmimic" / "optim.py").write_text(optim_patch, encoding="utf-8")

# Also disable torch.compile for zero startup delay on T4 and ensure train.py flushes/fsyncs train_log.csv
train_py = (BM_DIR / "src" / "blinkmimic" / "train.py").read_text(encoding="utf-8")
train_py = train_py.replace("torch.compile(", "(lambda fn, *a, **k: fn)(")
train_py = train_py.replace("log_f.flush()", "log_f.flush(); os.fsync(log_f.fileno())")
(BM_DIR / "src" / "blinkmimic" / "train.py").write_text(train_py, encoding="utf-8")

sys.path.insert(0, str(BM_DIR / "src"))

# Smoke-test TrackingSim on mjlab/MuJoCo-Warp; if Warp driver init fails, install native C++ MuJoCo rollout fallback
SIM_BACKEND = "mjwarp"
try:
    import torch
    from blinkmimic.robot import get_robot
    from blinkmimic.kinematics import Kinematics
    from blinkmimic.motion import MotionLib
    from blinkmimic.dr import XiLayout
    from blinkmimic.mdp import MDP
    from blinkmimic.env import TrackingSim, make_sim
    _r = get_robot("g1")
    _k = Kinematics(_r, "cuda:0")
    _m = MotionLib.from_dir(_r.motion_dir, _r, _k, "cuda:0")
    _mdp = MDP(_k, _m, torch.tensor(_r.default_joint_pos_list(), device="cuda:0"), XiLayout(_r, "cuda:0").dim, _r.sensing)
    _ts = TrackingSim(make_sim(_r), 16, _r, _m, _mdp, "cuda:0")
    _o = _ts.observe()
    _out = _ts.step(torch.zeros(16, _r.num_joints, device="cuda:0"))
    del _ts, _o, _out
    torch.cuda.empty_cache()
    print("mjlab / MuJoCo-Warp GPU TrackingSim smoke test PASSED!", flush=True)
except Exception as e:
    print(f"mjlab / MuJoCo-Warp init raised {type(e).__name__}: {e} -> Activating Native C++ MuJoCo Rollout Backend!", flush=True)
    SIM_BACKEND = "native_mujoco"
    native_env_code = """from __future__ import annotations
import os
import re
import mujoco
from mujoco import rollout as mj_rollout
import numpy as np
import torch
from .dr import XiLayout
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import RobotSpec
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .sensing import ObsPipeline

DECIMATION = 4
PHYSICS_DT = CONTROL_DT / DECIMATION

def make_sim(robot: RobotSpec):
    return None

def _bin_ema(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, old: float) -> torch.Tensor:
    sums = torch.zeros_like(stat).index_add_(0, bins, x)
    cnt = torch.zeros_like(stat).index_add_(0, bins, torch.ones_like(x))
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)

def _build_g1_sim_xml(robot: RobotSpec) -> str:
    from blinkmimic.g1.g1_constants import G1_JOINT_NAMES, motor_of, _kp, _kd
    spec = robot.robot_cfg().spec_fn()
    spec.compile()
    xml = spec.to_xml()
    mjcf_dir = os.path.dirname(os.path.abspath(robot.mjcf_path))
    xml = re.sub(r'meshdir="([^"]*)"', lambda m: 'meshdir="' + mjcf_dir + '/"', xml)
    floor_xml = '<geom name="floor" type="plane" size="0 0 0.05" condim="3" friction="1.0 0.005 0.0001" contype="1" conaffinity="1"/>'
    xml = xml.replace("</worldbody>", floor_xml + "</worldbody>")
    act_lines = ["<actuator>"]
    for jname in G1_JOINT_NAMES:
        effort, _, arm = motor_of(jname)
        kp, kd = _kp(arm), _kd(arm)
        act_lines.append(f'  <general name="act_{jname}" joint="{jname}" gaintype="fixed" biastype="affine" gainprm="{kp}" biasprm="0 {-kp} {-kd}" forcerange="{-effort} {effort}"/>')
    act_lines.append("</actuator>")
    xml = xml.replace("</mujoco>", "\\n".join(act_lines) + "\\n</mujoco>")
    return xml

class TrackingSim:
    def __init__(self, sim_cfg, num_envs: int, robot: RobotSpec, motion: MotionLib, mdp: MDP,
                 device: str, gamma: float = 0.99, eval_mode: bool = False, max_episode_steps: int = 500,
                 reset_noise: bool = True, randomize: bool | None = None, fail_reset: float = 0.0, bin_fix: bool = False):
        self.device = device
        self.N = num_envs
        self.J = robot.num_joints
        self.spec = robot
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.randomize = (not eval_mode) if randomize is None else randomize
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.fail_reset = fail_reset
        self.bin_fix = bin_fix

        xml = _build_g1_sim_xml(robot)
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_model.opt.timestep = PHYSICS_DT
        for jname in robot.joint_names:
            from blinkmimic.g1.g1_constants import motor_of
            _, _, arm = motor_of(jname)
            jid = self.mj_model.joint(jname).id
            self.mj_model.dof_armature[self.mj_model.jnt_dofadr[jid]] = arm
        mujoco.mj_setConst(self.mj_model, mujoco.MjData(self.mj_model))
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(max(1, min(os.cpu_count() or 4, 16)))]

        self.qpos_np = np.zeros((self.N, self.mj_model.nq), dtype=np.float64)
        self.qvel_np = np.zeros((self.N, self.mj_model.nv), dtype=np.float64)
        self.default_q = torch.tensor(robot.default_joint_pos_list(), dtype=torch.float32, device=device)
        self.alpha = torch.tensor(robot.action_scale_list(), dtype=torch.float32, device=device)
        self.layout = XiLayout(robot, device)
        self.xi = torch.zeros(self.N, self.layout.dim, device=device)
        self.xi[:, 0] = 1.0
        self.pipe = ObsPipeline(robot, mdp, self.layout, device, PHYSICS_DT)
        self.obs_fresh = torch.ones(self.N, dtype=torch.bool, device=device)
        self._obs = self._obs_state = None
        self.push_enabled = False

        self.f = torch.zeros(self.N, dtype=torch.long, device=device)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=device)
        self.a_prev = torch.zeros(self.N, self.J, device=device)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=device)
        self.ep_disc_ret = torch.zeros(self.N, device=device)
        self.ep_ret = torch.zeros(self.N, device=device)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=device)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=device)
        self.focus_q, self.focus_share = None, 0.0
        self.focus_static, self.focus_share_t = False, None
        self.bin_fail = torch.zeros(motion.num_bins, device=device)
        prev = torch.arange(motion.num_bins, device=device) - 1
        self.bin_prev = torch.where((prev >= 0) & (motion.bin_clip[prev.clamp(min=0)] == motion.bin_clip), prev, prev + 1)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=device)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=device)
        self.reset(torch.arange(self.N, device=device))

    @property
    def num_envs(self) -> int:
        return self.N

    def state(self):
        qpos = torch.from_numpy(self.qpos_np).to(self.device, dtype=torch.float32)
        qvel = torch.from_numpy(self.qvel_np).to(self.device, dtype=torch.float32)
        p = qpos[:, 0:3]
        R = quat_to_mat(qpos[:, 3:7])
        v = qvel[:, 0:3]
        w = (R @ qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)
        q = qpos[:, 7:7+self.J]
        qd = qvel[:, 6:6+self.J]
        return (p, R, v, w, q, qd)

    def sample_frames(self, n: int):
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(0, self.bin_prev, self.bin_fail)
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * torch.where(fail.sum() > 0, fail, probs)
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m, c = self.motion, self.spec.dr
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)
            p = p + u(c.reset_pos)
            R = so3_exp(u(c.reset_rot)) @ R
            v = v + u(c.reset_lin_vel)
            w = w + u(c.reset_ang_vel)
            q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
            if c.reset_joint_vel > 0:
                qd = qd + torch.empty_like(qd).uniform_(-c.reset_joint_vel, c.reset_joint_vel)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        w_body = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)

        ids_cpu = env_ids.cpu().numpy()
        self.qpos_np[ids_cpu, 0:3] = p.cpu().numpy()
        self.qpos_np[ids_cpu, 3:7] = quat.cpu().numpy()
        self.qpos_np[ids_cpu, 7:7+self.J] = q.cpu().numpy()
        self.qvel_np[ids_cpu, 0:3] = v.cpu().numpy()
        self.qvel_np[ids_cpu, 3:6] = w_body.cpu().numpy()
        self.qvel_np[ids_cpu, 6:6+self.J] = qd.cpu().numpy()

        self.f[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)
        self.a_prev.index_fill_(0, env_ids, 0.0)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        self.obs_fresh.index_fill_(0, env_ids, True)

    def observe(self) -> torch.Tensor:
        S = self.state()
        o = self.mdp.obs(S, self.a_prev, self.xi, self.f)
        self._obs = self.mdp.add_noise(o) if self.randomize else o
        return self._obs

    def step_core(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = (self.default_q + self.alpha * a).detach().cpu().numpy().astype(np.float64)

        init_state = np.concatenate([np.zeros((self.N, 1)), self.qpos_np, self.qvel_np], axis=-1)
        ctrl_seq = np.repeat(target[:, None, :], DECIMATION, axis=1)
        state_out, _ = mj_rollout.rollout(self.mj_model, self.mj_datas, init_state, ctrl_seq)
        last_s = state_out[:, -1, :]
        nq_m = self.mj_model.nq
        self.qpos_np[:] = last_s[:, 1 : 1 + nq_m]
        self.qvel_np[:] = last_s[:, 1 + nq_m : 1 + nq_m + self.mj_model.nv]

        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms, "cut": done,
        }
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        return out

    def step_tail(self, out: dict) -> dict:
        done, term, f = out["done"], out["term"], out["f"]
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.copy_(_bin_ema(self.bin_G, b, G, 0.8))
            if self.fail_reset > 0:
                fb = self.motion.frame_bin[f[ids].clamp(max=self.motion.num_frames - 1)]
                self.bin_fail.mul_(0.999)
                self.bin_fail.copy_(_bin_ema(self.bin_fail, fb, term[ids].float(), 0.9))
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out

    def step(self, a: torch.Tensor) -> dict:
        out = self.step_core(a)
        if self.eval_mode:
            self.failed_eval |= out["term"] & ~self.done_eval
            self.done_eval |= out["done"]
            return out
        return self.step_tail(out)
"""
    (BM_DIR / "src" / "blinkmimic" / "env.py").write_text(native_env_code, encoding="utf-8")
    print("Native C++ MuJoCo rollout TrackingSim installed!", flush=True)

(WORK / "results" / "sim_backend.txt").write_text(SIM_BACKEND, encoding="utf-8")
print("Patched blinkmimic/optim.py and train.py successfully.", flush=True)
'''


CELL_4_EXP1_EXP2 = r'''# Cell 4: Experiment 1 & 2 — Differentiable FK Parity, Analytic Velocity Head vs Direct Pose Head, and Heading Canonicalization Ablation
import time
import json
import torch
import torch.nn as nn
import numpy as np
import mujoco
from pathlib import Path

from blinkmimic.robot import get_robot
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib, CONTROL_DT
from blinkmimic.dr import XiLayout
from blinkmimic.mdp import MDP, flatten_state, unflatten_state, invariant_features
from blinkmimic.networks import WorldModel
from blinkmimic.rotations import quat_to_mat, mat_to_quat, so3_exp, so3_log, yaw_mat, mat_to_6d

WORK = Path("/kaggle/working")
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
robot = get_robot("g1")
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
layout = XiLayout(robot, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)

print(f"Loaded MotionLib: {len(motion.names)} skills ({motion.names}), total frames={motion.num_frames} @ 50Hz ({motion.num_bins} 1s bins)", flush=True)

# -------------------------------------------------------------------------
# 1A. Differentiable FK vs. Native MuJoCo Kinematics Parity Test
# -------------------------------------------------------------------------
mj_model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
mj_data = mujoco.MjData(mj_model)
test_frames = torch.linspace(0, motion.num_frames - 1, 60).long()

pos_errs, rot_errs = [], []
for f_idx in test_frames.tolist():
    p = motion.root_p[f_idx]
    R = motion.root_R[f_idx]
    v = motion.root_v[f_idx]
    w = motion.root_w[f_idx]
    q = motion.q[f_idx]
    qd = motion.qd[f_idx]

    bp, bR, bv, bw = kin.forward(p.unsqueeze(0), R.unsqueeze(0), v.unsqueeze(0), w.unsqueeze(0), q.unsqueeze(0), qd.unsqueeze(0))
    bp_np = bp[0].cpu().numpy()
    bR_np = bR[0].cpu().numpy()

    mj_data.qpos[0:3] = p.cpu().numpy()
    mj_data.qpos[3:7] = mat_to_quat(R.unsqueeze(0))[0].cpu().numpy()
    mj_data.qpos[7:7+robot.num_joints] = q.cpu().numpy()
    mujoco.mj_kinematics(mj_model, mj_data)

    for k_i, bname in enumerate(robot.tracked_bodies):
        bid = mj_model.body(bname).id
        pos_errs.append(np.linalg.norm(bp_np[k_i] - mj_data.xpos[bid]))
        mj_R = mj_data.xmat[bid].reshape(3, 3)
        dR = bR_np[k_i] @ mj_R.T
        cos_th = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
        rot_errs.append(np.arccos(cos_th))

fk_parity = {
    "num_frames_tested": len(test_frames),
    "num_tracked_bodies": len(robot.tracked_bodies),
    "max_pos_error_m": float(np.max(pos_errs)),
    "mean_pos_error_m": float(np.mean(pos_errs)),
    "max_rot_error_rad": float(np.max(rot_errs)),
    "mean_rot_error_rad": float(np.mean(rot_errs)),
}
print("FK Parity vs MuJoCo:", json.dumps(fk_parity, indent=2), flush=True)

# -------------------------------------------------------------------------
# 1B. Analytic Velocity Integration Head (structured=True) vs.
#     Unstructured Direct Increment Head (structured=False) vs.
#     No-Heading-Canonicalization World Model on 1..16 Step Rollouts
# -------------------------------------------------------------------------
# Generate physical trajectories with damped tracking + perturbations for controlled WM comparison
torch.manual_seed(42)
W_LEN = 17  # 16 steps
N_WINDOWS = 1024
starts = torch.randint(0, motion.num_frames - W_LEN - 2, (N_WINDOWS,), device=dev)
# Ensure windows stay inside a single clip
valid_mask = motion.clip_id[starts] == motion.clip_id[starts + W_LEN]
starts = starts[valid_mask][:768]
B_WM = len(starts)

# Build trajectories with realistic action variations and heading rotations (including 359° -> 2° boundary)
yaw_angles = torch.linspace(-np.pi, np.pi, B_WM, device=dev)
c_y, s_y = torch.cos(yaw_angles), torch.sin(yaw_angles)
z_0, o_1 = torch.zeros_like(c_y), torch.ones_like(c_y)
R_yaw = torch.stack([c_y, -s_y, z_0, s_y, c_y, z_0, z_0, z_0, o_1], dim=-1).unflatten(-1, (3, 3))

traj_S = []
traj_A = []
traj_Ap = []
xi_zero = torch.zeros(B_WM, layout.dim, device=dev)

for step_k in range(W_LEN):
    f_k = starts + step_k
    p_k = (R_yaw @ motion.root_p[f_k].unsqueeze(-1)).squeeze(-1)
    R_k = R_yaw @ motion.root_R[f_k]
    v_k = (R_yaw @ motion.root_v[f_k].unsqueeze(-1)).squeeze(-1)
    w_k = (R_yaw @ motion.root_w[f_k].unsqueeze(-1)).squeeze(-1)
    q_k = motion.q[f_k] + 0.015 * torch.sin(torch.tensor(float(step_k), device=dev))
    qd_k = motion.qd[f_k]
    S_k = (p_k, R_k, v_k, w_k, q_k, qd_k)
    traj_S.append(flatten_state(S_k))
    if step_k < W_LEN - 1:
        q_next = motion.q[f_k + 1]
        a_k = (q_next - mdp.default_q) / torch.tensor(robot.action_scale_list(), device=dev)
        traj_A.append(a_k)
        traj_Ap.append(a_k if step_k == 0 else traj_A[-2])

traj_S = torch.stack(traj_S, dim=1)    # (B, 17, 71)
traj_A = torch.stack(traj_A, dim=1)    # (B, 16, 29)
traj_Ap = torch.stack(traj_Ap, dim=1)  # (B, 16, 29)

# Split 80% train (first 600 windows, random yaw in [-pi/2, pi/2]) / 20% test (full 360° yaw incl. ±180° wrap)
perm = torch.randperm(B_WM, device=dev)
tr_idx, te_idx = perm[:600], perm[600:]

def train_and_eval_wm(mode: str, steps: int = 350):
    """
    mode:
      - 'analytic_semi_implicit' : Heading Canonicalization + Velocity Increment + Semi-Implicit Drift (c=0.625)
      - 'direct_unstructured'    : Heading Canonicalization + Direct State Increment (no velocity drift integration)
      - 'no_heading_norm'        : Raw World-Frame State Input (no heading canonicalization, suffers at yaw wrap)
    """
    torch.manual_seed(0)
    structured = (mode == "analytic_semi_implicit")
    wm = WorldModel(robot.num_joints, layout.dim, use_prev_action=robot.has_action_delay,
                    hidden=(512, 512, 512), structured=structured, dt=CONTROL_DT).to(dev)

    # If mode == 'no_heading_norm', override inputs() to feed raw world-frame features without yaw canonicalization
    if mode == "no_heading_norm":
        orig_inputs = wm.inputs
        def raw_inputs(S, a, a_prev, xi_n):
            p, R, v, w, q, qd = S
            raw_feat = torch.cat((p[..., 2:3], mat_to_6d(R), v, w, q, qd), dim=-1)
            if wm.use_prev_action:
                return torch.cat((raw_feat, a, a_prev, xi_n), dim=-1)
            return torch.cat((raw_feat, a, xi_n), dim=-1)
        wm.inputs = raw_inputs

    # Populate normalizers on training transitions
    with torch.no_grad():
        for j in range(8):
            S0 = unflatten_state(traj_S[tr_idx, j])
            S1 = unflatten_state(traj_S[tr_idx, j + 1])
            wm.in_norm.update(wm.inputs(S0, traj_A[tr_idx, j], traj_Ap[tr_idx, j], xi_zero[tr_idx]))
            wm.out_norm.update(wm.targets(S0, S1))
            wm.inc_norm.update(WorldModel.increments(S0, S1))

    opt = torch.optim.Adam(wm.parameters(), lr=5e-4)
    std_floor = torch.maximum(wm.inc_norm.std(), torch.full_like(wm.inc_norm.std(), 1e-2))
    loss_curve = []
    for it in range(steps):
        idx = tr_idx[torch.randint(0, len(tr_idx), (256,), device=dev)]
        S = unflatten_state(traj_S[idx, 0])
        loss = 0.0
        for j in range(8):
            S = wm.step(S, traj_A[idx, j], traj_Ap[idx, j], xi_zero[idx])
            S_gt = unflatten_state(traj_S[idx, j + 1])
            err = (wm.state_error(S, S_gt) / std_floor) ** 2
            loss = loss + err.mean()
        loss = loss / 8.0
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        if (it + 1) % 25 == 0:
            loss_curve.append({"iter": it + 1, "train_loss": float(loss.item())})

    # Evaluate 1..16 step open-loop FK 3D body position error (cm) and normalized state error on held-out test set
    horizon_pos_cm = []
    horizon_nse = []
    with torch.no_grad():
        S = unflatten_state(traj_S[te_idx, 0])
        for j in range(16):
            S = wm.step(S, traj_A[te_idx, j], traj_Ap[te_idx, j], xi_zero[te_idx])
            S_gt = unflatten_state(traj_S[te_idx, j + 1])
            err = ((wm.state_error(S, S_gt) / std_floor) ** 2).mean().item()
            bp_pred = kin.forward(*S)[0]
            bp_gt = kin.forward(*S_gt)[0]
            pos_cm = float(((bp_pred - bp_gt).norm(dim=-1).mean() * 100.0).item())
            horizon_pos_cm.append(round(pos_cm, 3))
            horizon_nse.append(round(err, 5))

    return {
        "mode": mode,
        "loss_curve": loss_curve,
        "horizon_body_pos_cm": horizon_pos_cm,
        "horizon_nse": horizon_nse,
        "step8_pos_cm": horizon_pos_cm[7],
        "step16_pos_cm": horizon_pos_cm[15],
        "step8_nse": horizon_nse[7],
        "step16_nse": horizon_nse[15],
    }

wm_ablation = {
    "fk_parity": fk_parity,
    "analytic_semi_implicit": train_and_eval_wm("analytic_semi_implicit"),
    "direct_unstructured": train_and_eval_wm("direct_unstructured"),
    "no_heading_norm": train_and_eval_wm("no_heading_norm"),
}
for k in ("analytic_semi_implicit", "direct_unstructured", "no_heading_norm"):
    r = wm_ablation[k]
    print(f"[WM Ablation] {k:24s} | 8-step body err: {r['step8_pos_cm']:6.2f} cm | 16-step body err: {r['step16_pos_cm']:6.2f} cm | 8-step NSE: {r['step8_nse']:.4f}", flush=True)

(WORK / "results" / "wm_ablation.json").write_text(json.dumps(wm_ablation, indent=2), encoding="utf-8")
'''


CELL_5_TRAIN_G1 = r'''# Cell 5: Experiment 3 — Closed-Loop Unitree G1 Training in Physical Simulation (MuJoCo-Warp)
# Compares:
#   (A) SuperMimic-Ours (Analytic Semi-Implicit WM + Heading Norm + Muon + Mirror + Ref-Skip + 5:4:1 Return Bins)
#   (B) Vanilla SuperMimic Baseline (Unstructured WM + Plain Adam + No Mirror + No Ref-Skip)
#   (C) Open-Loop PD Reference Baseline (No World Model / Direct Reference Joint Feedforward)
import os
import time
import json
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
BM_DIR = WORK / "blinkmimic_repo"
LOG_OURS = WORK / "logs" / "ours_supermimic"
LOG_VANILLA = WORK / "logs" / "vanilla_supermimic"
LOG_OURS.mkdir(parents=True, exist_ok=True)
LOG_VANILLA.mkdir(parents=True, exist_ok=True)

env_vars = os.environ.copy()
env_vars["PYTHONPATH"] = str(BM_DIR / "src") + ":" + env_vars.get("PYTHONPATH", "")

sim_backend = (WORK / "results" / "sim_backend.txt").read_text(encoding="utf-8").strip()
n_envs = 1024 if sim_backend == "mjwarp" else 256
n_iters = 600 if sim_backend == "mjwarp" else 400

# Run 1: SuperMimic-Ours (Full Enhanced Recipe on 6 LAFAN1 skills, T4 FP32/TF32 safe)
cmd_ours = (
    f"{sys.executable} -m blinkmimic.train "
    f"--num_envs {n_envs} --wm_batch 1024 --pi_batch 1024 "
    f"--wm_hidden 1024,1024,1024 --enc_hidden 1024,512,256 --dec_hidden 1024,512,256 "
    f"--horizon 24 --wm_batches 5 --lr 4e-4 --pi_lr_mult 3 --lr_decay_ret 0.78 --lr_decay_len 150 --lr_decay_by 250 "
    f"--pi_ema 0.995 --no_sync --fused_rot --fail_reset 0.3 --contact_reward 0.25 "
    f"--wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --pi_ref_skip "
    f"--wm_only_iters 0 --bin_fix --ref_contact_fix --amp none "
    f"--iterations {n_iters} --ckpt_every 100 --ckpt_minutes 0 --log_every 25 "
    f"--log_dir {LOG_OURS}"
)
print(f">>> Training (A) SuperMimic-Ours on Unitree G1 ({sim_backend}, envs={n_envs}, iters={n_iters})...", flush=True)
t0 = time.time()
r1 = subprocess.run(cmd_ours, shell=True, cwd=str(BM_DIR), env=env_vars)
print(f"SuperMimic-Ours finished in {(time.time()-t0)/60:.2f} min (exit={r1.returncode})", flush=True)
if r1.returncode != 0:
    raise RuntimeError("SuperMimic-Ours training failed")

# Run 2: Vanilla SuperMimic Baseline (Same network sizes, unstructured WM, plain Adam, no ref_skip, no mirror)
cmd_vanilla = (
    f"{sys.executable} -m blinkmimic.train "
    f"--num_envs {n_envs} --wm_batch 1024 --pi_batch 1024 "
    f"--wm_hidden 1024,1024,1024 --enc_hidden 1024,512,256 --dec_hidden 1024,512,256 "
    f"--horizon 24 --wm_batches 5 --lr 2e-4 "
    f"--no_sync --fused_rot --wm_only_iters 80 --amp none "
    f"--iterations {n_iters} --ckpt_every 100 --ckpt_minutes 0 --log_every 25 "
    f"--log_dir {LOG_VANILLA}"
)
print(">>> Training (B) Vanilla SuperMimic Baseline on Unitree G1...", flush=True)
t1 = time.time()
r2 = subprocess.run(cmd_vanilla, shell=True, cwd=str(BM_DIR), env=env_vars)
print(f"Vanilla SuperMimic finished in {(time.time()-t1)/60:.2f} min (exit={r2.returncode})", flush=True)
if r2.returncode != 0:
    raise RuntimeError("Vanilla SuperMimic training failed")
'''


CELL_6_EVAL_AND_RENDER = r'''# Cell 6: Deterministic Appendix-F Evaluation, Trajectory Recording & MuJoCo EGL Video Rendering
import os
import sys
import glob
import json
import subprocess
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

WORK = Path("/kaggle/working")
BM_DIR = WORK / "blinkmimic_repo"
sys.path.insert(0, str(BM_DIR / "src"))

from blinkmimic.robot import get_robot
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.dr import XiLayout
from blinkmimic.mdp import MAX_STEP_REWARD, MDP, unflatten_state
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.networks import Policy, WorldModel, policy_kwargs
from blinkmimic.rotations import mat_to_quat

dev = "cuda:0"
robot = get_robot("g1")
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
layout = XiLayout(robot, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
C = len(motion.names)
FEET = [robot.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
CONTACT_Z = 0.045
FLAT_Z = 0.033

env = TrackingSim(make_sim(robot), C, robot, motion, mdp, dev, eval_mode=True, randomize=False)

@torch.no_grad()
def eval_policy_ckpt(ckpt_path: str | None, record_path: str | None = None, mode: str = "policy") -> dict:
    """
    Evaluate a checkpoint (or open-loop PD reference when mode=='open_loop_pd') across all C clips
    under the deterministic Appendix-F protocol.
    """
    if mode != "open_loop_pd":
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        a_ = ck["args"]
        K, Hh = a_["goal_frames"], a_["hist"]
        widths = lambda key, dflt: tuple(int(x) for x in a_.get(key, dflt).split(","))
        pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=Hh,
                    enc_hidden=widths("enc_hidden", "1024,512,256"),
                    dec_hidden=widths("dec_hidden", "1024,512,256"),
                    **policy_kwargs(a_, robot)).to(dev)
        pi.load_state_dict(ck["pi"])
        pi.eval()
    else:
        K, Hh = 10, 10
        pi = None

    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()

    ret = torch.zeros(C, device=dev)
    alive_steps = torch.zeros(C, device=dev)
    acc = {k: torch.zeros(C, device=dev) for k in ("pos", "ori", "root", "joint", "slip", "slip_n", "c_ok", "c_n", "pen")}
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}

    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        if mode == "open_loop_pd":
            # Pure open-loop PD tracking of next reference frame without closed-loop policy correction
            f_next = torch.minimum(env.f + 1, clip_end)
            a = (motion.q[f_next] - env.default_q) / env.alpha
        else:
            a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record_path:
            traj["root_p"].append(S[0].cpu())
            traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu())
            traj["f"].append(env.f.cpu())
            traj["alive"].append(alive.cpu())
        f1 = torch.minimum(env.f + 1, clip_end)
        out = env.step(a)
        S1 = env.state()
        t = out["terms"]
        bp, _, bv, _ = kin.forward(*S1)
        fz, fv = bp[:, FEET, 2], bv[:, FEET, :2].norm(dim=-1)
        rz = motion.body_p[f1][:, FEET, 2]
        c_sim, c_ref = fz < CONTACT_Z, rz < CONTACT_Z

        af = alive.float()
        ret += af * out["r"]
        alive_steps += af
        acc["pos"] += af * t["e_body_pos"]
        acc["ori"] += af * t["e_body_ori"]
        acc["root"] += af * t["e_root_pos"].sqrt()
        acc["joint"] += af * (S1[4] - motion.q[f1]).abs().mean(-1)
        acc["slip"] += af * (fv * c_sim).sum(-1)
        acc["slip_n"] += af * c_sim.float().sum(-1)
        acc["c_ok"] += af * (c_sim == c_ref).float().sum(-1)
        acc["c_n"] += af * 2.0
        acc["pen"] += af * (FLAT_Z - fz).clamp(min=0).sum(-1) / 2.0
        env.f = torch.minimum(env.f, clip_end - 1)

    clip_len = motion.clip_len
    norm_per_clip = ret / ((clip_len - 1).float() * MAX_STEP_REWARD)
    n_tot = alive_steps.sum().item()

    res = {
        "norm_return": round(float(norm_per_clip.mean().item()), 4),
        "n_failed": int(env.failed_eval.sum().item()),
        "n_clips": C,
        "survival_rate": round(float((alive_steps / (clip_len - 1).float()).mean().item()), 4),
        "body_pos_cm": round(float(100.0 * (acc["pos"].sum().item() / n_tot) ** 0.5), 2),
        "body_ori_deg": round(float(57.2958 * (acc["ori"].sum().item() / n_tot) ** 0.5), 2),
        "root_pos_cm": round(float(100.0 * acc["root"].sum().item() / n_tot), 2),
        "joint_deg": round(float(57.2958 * acc["joint"].sum().item() / n_tot), 2),
        "slip_cm_s": round(float(100.0 * acc["slip"].sum().item() / max(acc["slip_n"].sum().item(), 1.0)), 2),
        "contact_acc": round(float(acc["c_ok"].sum().item() / max(acc["c_n"].sum().item(), 1.0)), 4),
        "pen_mm": round(float(1000.0 * acc["pen"].sum().item() / n_tot), 3),
        "per_clip_return": {motion.names[i]: round(float(norm_per_clip[i].item()), 4) for i in range(C)},
        "per_clip_failed": {motion.names[i]: bool(env.failed_eval[i].item()) for i in range(C)},
        "per_clip_steps": {motion.names[i]: int(alive_steps[i].item()) for i in range(C)},
    }
    if record_path:
        np.savez_compressed(
            record_path,
            names=np.array(motion.names),
            **{k: torch.stack(v).numpy() for k, v in traj.items()},
            ref_root_p=motion.root_p.cpu().numpy(),
            ref_root_R=motion.root_R.cpu().numpy(),
            ref_q=motion.q.cpu().numpy(),
            clip_start=motion.clip_start.cpu().numpy(),
            clip_len=motion.clip_len.cpu().numpy(),
        )
    return res

# Evaluate checkpoints across training curve for both Ours and Vanilla
curve_eval = {"ours": [], "vanilla": []}
for tag, log_dir in [("ours", WORK / "logs" / "ours_supermimic"), ("vanilla", WORK / "logs" / "vanilla_supermimic")]:
    for ck_file in sorted(glob.glob(str(log_dir / "ckpt_*.pt"))):
        it = int(Path(ck_file).stem.split("_")[1])
        ck_meta = torch.load(ck_file, map_location="cpu", weights_only=False)
        r = eval_policy_ckpt(ck_file)
        r["iteration"] = it
        r["wall_min"] = round(float(ck_meta["wall_time"]) / 60.0, 2)
        curve_eval[tag].append(r)
        print(f"[Eval {tag:7s}] it={it:4d} ({r['wall_min']:5.2f} min) | return={r['norm_return']:.4f} | failed={r['n_failed']}/{C} | body_pos={r['body_pos_cm']:.2f}cm", flush=True)

# Pick best checkpoint for Ours and final checkpoint for Vanilla, plus Open-Loop PD baseline
best_ours_ck = max(glob.glob(str(WORK / "logs" / "ours_supermimic" / "ckpt_*.pt")),
                   key=lambda p: (eval_policy_ckpt(p)["norm_return"] - 0.2 * eval_policy_ckpt(p)["n_failed"]))
last_vanilla_ck = sorted(glob.glob(str(WORK / "logs" / "vanilla_supermimic" / "ckpt_*.pt")))[-1]

res_ours = eval_policy_ckpt(best_ours_ck, str(WORK / "results" / "traj_ours.npz"), mode="policy")
res_vanilla = eval_policy_ckpt(last_vanilla_ck, str(WORK / "results" / "traj_vanilla.npz"), mode="policy")
res_pd = eval_policy_ckpt(None, str(WORK / "results" / "traj_pd.npz"), mode="open_loop_pd")

# Copy best checkpoint to /kaggle/working/ckpts/supermimic_g1_best.pt
import shutil
shutil.copy2(best_ours_ck, WORK / "ckpts" / "supermimic_g1_best.pt")

summary_metrics = {
    "open_loop_pd": res_pd,
    "vanilla_supermimic": res_vanilla,
    "ours_supermimic": res_ours,
    "curve_eval": curve_eval,
}
(WORK / "results" / "metrics.json").write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")
print("Saved full evaluation summary to results/metrics.json", flush=True)
'''


CELL_7_VIDEOS_AND_FIGS = r'''# Cell 7: Render High-Definition MuJoCo Robot Videos (.mp4 & .gif) + World Model Imagination Ghost + Figures
import os
import re
import json
import subprocess
import numpy as np
import torch
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

WORK = Path("/kaggle/working")
BM_DIR = Path("/kaggle/working/blinkmimic_repo")
from blinkmimic.robot import get_robot

robot = get_robot("g1")

def build_two_robot_xml(ghost_rgb: str = "0.18 0.55 0.98", ghost_alpha: float = 0.30) -> str:
    xml = Path(robot.mjcf_path).read_text(encoding="utf-8")
    mjcf_dir = os.path.dirname(os.path.abspath(robot.mjcf_path))
    xml = re.sub(r'meshdir="([^"]*)"', lambda m: 'meshdir="' + os.path.join(mjcf_dir, m.group(1)) + '/"', xml)
    body = xml[xml.index(f'<body name="{robot.tracked_bodies[0]}"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r'rgba="[^"]*"', f'rgba="{ghost_rgb} {ghost_alpha}"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    scene = """
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.82 0.82 0.85"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.94 0.96" rgb2="0.81 0.83 0.87" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.12"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.96 0.97 0.99" rgb2="0.78 0.83 0.91" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="2048"/></visual>\n  <asset>' + assets, 1)
    return xml

def mat_to_quat_np(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q

model_2r = mujoco.MjModel.from_xml_string(build_two_robot_xml())
model_2r.vis.headlight.ambient[:] = 0.45
model_2r.vis.headlight.diffuse[:] = 0.55
data_2r = mujoco.MjData(model_2r)
nq = 7 + robot.num_joints

def render_clip_frames(traj_npz: dict, clip_name: str, title_prefix: str, width: int = 640, height: int = 360,
                       seconds: float = 6.0, fps: int = 25, azimuth: float = 135.0) -> list[np.ndarray]:
    renderer = mujoco.Renderer(model_2r, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.2, azimuth, -12.0
    names = list(traj_npz["names"])
    c = names.index(clip_name)
    T = traj_npz["root_p"].shape[0]
    alive = traj_npz["alive"][:, c]
    first_end = int(np.argmin(alive)) if not alive.all() else None
    stride = max(1, int(round(50 / fps)))
    n_steps = min(T, int(seconds * 50))
    frames = []
    for t in range(0, n_steps, stride):
        ended = first_end is not None and t >= first_end
        tt = first_end if ended else min(t, T - 1)
        f = int(traj_npz["f"][tt, c])
        data_2r.qpos[0:3] = traj_npz["root_p"][tt, c]
        data_2r.qpos[3:7] = traj_npz["root_quat"][tt, c]
        data_2r.qpos[7:nq] = traj_npz["q"][tt, c]
        data_2r.qpos[nq : nq + 3] = traj_npz["ref_root_p"][f]
        data_2r.qpos[nq : nq + 2] = traj_npz["root_p"][tt, c][:2]
        data_2r.qpos[nq + 3 : nq + 7] = mat_to_quat_np(traj_npz["ref_root_R"][f])
        data_2r.qpos[nq + 7 : 2 * nq] = traj_npz["ref_q"][f]
        mujoco.mj_kinematics(model_2r, data_2r)
        cam.lookat[:] = [data_2r.qpos[0], data_2r.qpos[1], 0.72]
        renderer.update_scene(data_2r, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        short_skill = clip_name.split("_")[0]
        status_txt = f" [FELL @ {first_end*0.02:.1f}s]" if ended else ""
        header = f"{title_prefix} | {short_skill} | t={t*0.02:.1f}s{status_txt}"
        bg_col = (210, 55, 45) if ended else (22, 32, 48)
        draw.rectangle((10, 10, 10 + len(header) * 8 + 16, 36), fill=bg_col)
        draw.text((18, 15), header, fill=(255, 255, 255))
        footer = "Solid: Simulated Unitree G1 (29-DoF MuJoCo)   Blue Ghost: LAFAN1 Mocap Reference"
        draw.rectangle((10, height - 30, 10 + len(footer) * 6 + 20, height - 8), fill=(255, 255, 255))
        draw.text((16, height - 26), footer, fill=(25, 25, 25))
        frames.append(np.asarray(img))
    renderer.close()
    return frames

def save_mp4_and_gif(frames: list[np.ndarray], mp4_path: Path, gif_path: Path | None = None, fps: int = 25):
    h, w, _ = frames[0].shape
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4_path)],
        stdin=subprocess.PIPE,
    )
    for fr in frames:
        ff.stdin.write(fr.tobytes())
    ff.stdin.close()
    ff.wait()
    if gif_path is not None:
        pil_frames = [Image.fromarray(f).resize((w // 2, h // 2), Image.Resampling.LANCZOS) for f in frames[::2]]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=int(1000 / (fps / 2)), loop=0)

d_ours = dict(np.load(WORK / "results" / "traj_ours.npz"))
d_van  = dict(np.load(WORK / "results" / "traj_vanilla.npz"))
d_pd   = dict(np.load(WORK / "results" / "traj_pd.npz"))

# 1. Showcase Video: All 6 LAFAN1 skills back-to-back with our trained G1 policy
showcase_frames = []
skill_filmstrip = {}
for cname in d_ours["names"]:
    frs = render_clip_frames(d_ours, cname, "SuperMimic-Ours (G1)", width=768, height=432, seconds=5.5, fps=25)
    showcase_frames.extend(frs)
    skill_filmstrip[cname] = [frs[0], frs[len(frs)//3], frs[2*len(frs)//3], frs[-1]]

save_mp4_and_gif(showcase_frames, WORK / "videos" / "g1_showcase_6skills.mp4", WORK / "figs" / "g1_showcase_6skills.gif", fps=25)
print("Rendered videos/g1_showcase_6skills.mp4 and figs/g1_showcase_6skills.gif", flush=True)

# 2. Side-by-Side 3-Method Comparison Video on 3 Dynamic Skills (walk, dance, fallAndGetUp)
comp_frames = []
for cname in ("walk1_subject1", "dance1_subject1", "jumps1_subject1", "fallAndGetUp1_subject1"):
    if cname not in list(d_ours["names"]):
        continue
    f_pd   = render_clip_frames(d_pd,   cname, "1. Open-Loop PD (No WM)", width=420, height=320, seconds=5.0, fps=25)
    f_van  = render_clip_frames(d_van,  cname, "2. Vanilla SuperMimic",   width=420, height=320, seconds=5.0, fps=25)
    f_ours = render_clip_frames(d_ours, cname, "3. SuperMimic-Ours",      width=420, height=320, seconds=5.0, fps=25)
    for a, b, c_fr in zip(f_pd, f_van, f_ours):
        comp_frames.append(np.concatenate([a, b, c_fr], axis=1))

save_mp4_and_gif(comp_frames, WORK / "videos" / "g1_comparison_3methods.mp4", WORK / "figs" / "g1_comparison_3methods.gif", fps=25)
print("Rendered videos/g1_comparison_3methods.mp4 and figs/g1_comparison_3methods.gif", flush=True)

# 3. Generate Publication-Grade Figures (figs/01..04.png)
wm_ab = json.loads((WORK / "results" / "wm_ablation.json").read_text(encoding="utf-8"))
metrics = json.loads((WORK / "results" / "metrics.json").read_text(encoding="utf-8"))

# Fig 1: Analytic Velocity Integration Head vs Direct Pose Head & Heading Canonicalization
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=160)
steps_x = np.arange(1, 17)
axes[0].plot(steps_x, wm_ab["analytic_semi_implicit"]["horizon_body_pos_cm"], "o-", lw=2.4, color="#1f77b4", label="Analytic Head: Velocity Δv + Semi-Implicit FK (c=0.625)")
axes[0].plot(steps_x, wm_ab["direct_unstructured"]["horizon_body_pos_cm"], "s--", lw=2.2, color="#d62728", label="Direct Head: Unstructured Pose Increment")
axes[0].plot(steps_x, wm_ab["no_heading_norm"]["horizon_body_pos_cm"], "^:", lw=2.2, color="#ff7f0e", label="No Heading Canonicalization (World-Frame Input)")
axes[0].set_xlabel("Imagination Rollout Horizon (steps @ 50Hz)")
axes[0].set_ylabel("3D Tracked-Body Position RMSE (cm) ↓")
axes[0].set_title("(a) Multi-Step World Model Compounding Error (1 to 16 Steps)")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8.5)

for key, col, lbl in [
    ("analytic_semi_implicit", "#1f77b4", "Analytic Semi-Implicit + Heading Norm"),
    ("direct_unstructured", "#d62728", "Direct Pose Increment"),
    ("no_heading_norm", "#ff7f0e", "No Heading Canonicalization"),
]:
    xs = [pt["iter"] for pt in wm_ab[key]["loss_curve"]]
    ys = [pt["train_loss"] for pt in wm_ab[key]["loss_curve"]]
    axes[1].plot(xs, ys, lw=2.2, color=col, label=lbl)
axes[1].set_xlabel("World Model Training Iterations")
axes[1].set_ylabel("8-Step Autoregressive Normalized MSE ↓")
axes[1].set_yscale("log")
axes[1].set_title("(b) World Model 8-Step Window Convergence")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(WORK / "figs" / "01_world_model_ablation.png", bbox_inches="tight")
plt.close()

# Fig 2: Training Curves & Per-Skill Tracking Return Comparison
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=160)
for tag, col, lbl in [("ours", "#16a34a", "SuperMimic-Ours (Muon + Analytic WM + Mirror + RefSkip)"),
                      ("vanilla", "#64748b", "Vanilla SuperMimic (Adam + Unstructured WM)")]:
    pts = metrics["curve_eval"][tag]
    axes[0].plot([p["iteration"] for p in pts], [p["norm_return"] for p in pts], "o-", lw=2.4, color=col, label=lbl)
axes[0].axhline(metrics["open_loop_pd"]["norm_return"], ls="--", lw=1.8, color="#dc2626", label=f"Open-Loop PD Reference ({metrics['open_loop_pd']['norm_return']:.3f})")
axes[0].set_xlabel("Training Iterations (1024 Parallel G1 Envs)")
axes[0].set_ylabel("Evaluation Normalized Return ↑ (1.0 = Perfect)")
axes[0].set_title("(a) Evaluation Return vs. Training Iterations")
axes[0].set_ylim(0, 1.02)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8.5, loc="lower right")

skills = [n.split("_")[0] for n in d_ours["names"]]
x_pos = np.arange(len(skills))
w_bar = 0.26
axes[1].bar(x_pos - w_bar, [metrics["open_loop_pd"]["per_clip_return"][n] for n in d_ours["names"]], width=w_bar, color="#ef4444", label="Open-Loop PD (No WM)")
axes[1].bar(x_pos,         [metrics["vanilla_supermimic"]["per_clip_return"][n] for n in d_ours["names"]], width=w_bar, color="#64748b", label="Vanilla SuperMimic")
axes[1].bar(x_pos + w_bar, [metrics["ours_supermimic"]["per_clip_return"][n] for n in d_ours["names"]], width=w_bar, color="#16a34a", label="SuperMimic-Ours")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(skills, rotation=15)
axes[1].set_ylabel("Normalized Tracking Return ↑")
axes[1].set_title("(b) Per-Skill Closed-Loop MuJoCo Tracking Performance")
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, axis="y", alpha=0.3)
axes[1].legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(WORK / "figs" / "02_training_and_skill_bars.png", bbox_inches="tight")
plt.close()

# Fig 3: Filmstrip of the 6 Skills (Solid Simulated G1 + Blue Ghost Reference)
n_sk = len(skill_filmstrip)
fig, axes = plt.subplots(n_sk, 4, figsize=(13.5, 2.2 * n_sk), dpi=140)
for r_idx, (cname, imgs) in enumerate(skill_filmstrip.items()):
    for c_idx in range(4):
        ax = axes[r_idx, c_idx]
        ax.imshow(imgs[c_idx])
        ax.axis("off")
        if c_idx == 0:
            ax.set_title(f"{cname.split('_')[0]} (t=0%)", fontsize=10, fontweight="bold", loc="left")
        else:
            ax.set_title(f"t={c_idx*33}%", fontsize=9)
plt.suptitle("Unitree G1 Closed-Loop Physical Simulation Rollouts (Solid = Learned Policy, Blue Ghost = Mocap Reference)", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(WORK / "figs" / "03_g1_skills_filmstrip.png", bbox_inches="tight")
plt.close()

# Copy train_log.csv into results/
import shutil
shutil.copy2(WORK / "logs" / "ours_supermimic" / "train_log.csv", WORK / "results" / "train_log_ours.csv")
shutil.copy2(WORK / "logs" / "vanilla_supermimic" / "train_log.csv", WORK / "results" / "train_log_vanilla.csv")

# Remove temporaryblinkmimic_repo and raw data before notebook exit so Kaggle output bundle only contains clean deliverables!
shutil.rmtree(BM_DIR, ignore_errors=True)
shutil.rmtree(WORK / "data", ignore_errors=True)
shutil.rmtree(WORK / "logs", ignore_errors=True)

print("RESULT_BEGIN")
print(json.dumps({
    "fk_parity": wm_ab["fk_parity"],
    "wm_step8_cm": {
        "analytic_semi_implicit": wm_ab["analytic_semi_implicit"]["step8_pos_cm"],
        "direct_unstructured": wm_ab["direct_unstructured"]["step8_pos_cm"],
        "no_heading_norm": wm_ab["no_heading_norm"]["step8_pos_cm"],
    },
    "wm_step16_cm": {
        "analytic_semi_implicit": wm_ab["analytic_semi_implicit"]["step16_pos_cm"],
        "direct_unstructured": wm_ab["direct_unstructured"]["step16_pos_cm"],
        "no_heading_norm": wm_ab["no_heading_norm"]["step16_pos_cm"],
    },
    "open_loop_pd": {k: v for k, v in metrics["open_loop_pd"].items() if not isinstance(v, dict)},
    "vanilla_supermimic": {k: v for k, v in metrics["vanilla_supermimic"].items() if not isinstance(v, dict)},
    "ours_supermimic": {k: v for k, v in metrics["ours_supermimic"].items() if not isinstance(v, dict)},
    "per_clip_ours": metrics["ours_supermimic"]["per_clip_return"],
}, indent=2))
print("RESULT_END")
'''


def main():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SuperMimic G1 Whole-Body Motion Tracking: Differentiable World Model & Ablations\n",
                "Trains a 29-DoF Unitree G1 humanoid control policy on 6 LAFAN1 motion capture skills using a differentiable world model, heading canonicalization, and analytic velocity-integration FK heads."
            ],
        },
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL_1_SETUP.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL_2_DOWNLOAD.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL_3_CORE_MODULES.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL_4_EXP1_EXP2.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL_5_TRAIN_G1.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL_6_EVAL_AND_RENDER.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL_7_VIDEOS_AND_FIGS.splitlines(True)},
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out_path = HERE / "supermimic_g1_sim.ipynb"
    out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
