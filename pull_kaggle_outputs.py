"""
pull_kaggle_outputs.py — Waits for `yimingjiang666666/scene40-3dgs` to finish on Kaggle GPU,
then downloads all output artifacts (`results/`, `figs/`, `videos/`, `ckpts/`) and logs into
`supermimic-g1-sim/`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pylibs"))

from kaggle import KaggleApi

KERNEL = "yimingjiang666666/scene40-3dgs"

api = KaggleApi()
api.authenticate()


def get_status() -> tuple[str, str | None]:
    st = api.kernels_status(KERNEL)
    s = str(getattr(st, "status", st)).strip().upper()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s, getattr(st, "failureMessage", None)


def main():
    t0 = time.time()
    while True:
        st, fail = get_status()
        el = (time.time() - t0) / 60.0
        print(f"[{datetime.now():%H:%M:%S}] {KERNEL} -> {st} ({el:.1f} min)" + (f" fail={fail}" if fail else ""), flush=True)
        if st in {"COMPLETE", "ERROR", "FAILED", "CANCELED"}:
            break
        time.sleep(30)

    # Pull logs
    txt = api.kernels_logs(KERNEL)
    if txt:
        parts = []
        if txt.strip().startswith("["):
            try:
                for ev in json.loads(txt):
                    if isinstance(ev, dict) and "data" in ev:
                        parts.append(ev["data"])
            except Exception:
                pass
        if not parts:
            for line in txt.splitlines():
                line = line.strip().rstrip(",")
                if line.startswith("{"):
                    try:
                        ev = json.loads(line)
                        if isinstance(ev, dict) and "data" in ev:
                            parts.append(ev["data"])
                    except Exception:
                        pass
        plain = "".join(parts)
        log_file = HERE / "kaggle_run.log"
        log_file.write_text(plain, encoding="utf-8")
        print(f"Saved log ({len(plain)} chars) -> {log_file}", flush=True)
        print("=" * 72)
        print(plain[-4000:])
        print("=" * 72)

    if st == "COMPLETE":
        print("Downloading kernel output artifacts...", flush=True)
        api.kernels_output(KERNEL, path=str(HERE), force=True, quiet=False)
        print("Downloaded all artifacts to", HERE, flush=True)


if __name__ == "__main__":
    main()
