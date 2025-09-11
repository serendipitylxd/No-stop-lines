# -*- coding: utf-8 -*-
"""
run_all.py — One-click pipeline for 2D→3D→Stopline Check→Visualization→Evaluation

- Step1: 2D inference (conda env "your-mmdetection_trout-env")
- Step2: 3D inference (conda env "your-PCDet_trout-env" + inject PYTHONPATH=/path/to/your-PCDet_trout/tools)
- Step3: Stopline check (conda env "your-PCDet_trout-env")
- Step4: Visualization (conda env "your-PCDet_trout-env")
- Step5: Evaluation (conda env "your-PCDet_trout-env")

Note: Step3/4/5 use the same "your-PCDet_trout-env" environment.
"""

import os
import sys
import shlex
import argparse
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent

MMDET_ENV = "your-mmdetection_trout-env"
PCDET_ENV = "your-PCDet_trout-env"
PCDET_REPO = "/home/luxiaodong/your-PCDet_trout/tools"   # Step2 will be injected into PYTHONPATH


def run(cmd, env=None, cwd=None):
    """Execute a shell command with optional custom environment and cwd."""
    print(f"\n$ {cmd}")
    ret = subprocess.run(cmd, shell=True, env=env, cwd=cwd)
    if ret.returncode != 0:
        raise SystemExit(ret.returncode)

def conda_cmd(env_name, py_file, cfg_path, extra_env=None):
    """Compose a conda run command (base with --cfg, other params handled inside each script)."""
    return (
        f"conda run --no-capture-output -n {shlex.quote(env_name)} python "
        f"{shlex.quote(str(py_file))} --cfg {shlex.quote(str(cfg_path))}"
    ), extra_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to configs/config.yaml")
    ap.add_argument("--no-2d", action="store_true", help="skip 2D inference")
    ap.add_argument("--no-3d", action="store_true", help="skip 3D inference")
    ap.add_argument("--no-check", action="store_true", help="skip stopline check")
    ap.add_argument("--no-viz", action="store_true", help="skip visualization")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluation")  # controls whether to run evaluation
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (PROJ_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        print(f"[ERR] cfg not found: {cfg_path}")
        sys.exit(1)

    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"

    # ========== STEP 1/5: 2D inference ==========
    print("========== STEP 1/5: 2D inference (MMDet) ==========")
    if not args.no_2d:
        cmd, env = conda_cmd(
            MMDET_ENV,
            PROJ_ROOT / "scripts" / "infer_2d_mmdet.py",
            cfg_path,
            env_base,
        )
        run(cmd, env)
    else:
        print("[SKIP] 2D inference")

    # ========== STEP 2/5: 3D inference ==========
    print("========== STEP 2/5: 3D inference (PCDet) ==========")
    if not args.no_3d:
        env_3d = env_base.copy()
        env_3d["PYTHONPATH"] = os.pathsep.join([PCDET_REPO, env_3d.get("PYTHONPATH", "")])
        cmd, _ = conda_cmd(
            PCDET_ENV,
            PROJ_ROOT / "scripts" / "infer_3d_pcdet.py",
            cfg_path,
            env_3d,
        )
        run(cmd, env_3d, cwd= PCDET_REPO)
    else:
        print("[SKIP] 3D inference")

    # ========== STEP 3/5: Stopline check ==========
    print("========== STEP 3/5: Stopline check (PCDet env) ==========")
    if not args.no_check:
        env_3d = env_base.copy()
        env_3d["PYTHONPATH"] = os.pathsep.join([str(PROJ_ROOT), env_3d.get("PYTHONPATH", "")])
        cmd, _ = conda_cmd(
            PCDET_ENV,
            PROJ_ROOT / "scripts" / "stopline_check.py",
            cfg_path,
            env_3d,
        )
        run(cmd, env_3d)
    else:
        print("[SKIP] stopline check")

    # ========== STEP 4/5: Visualization ==========
    print("========== STEP 4/5: Visualization (PCDet env) ==========")
    if not args.no_viz:
        env_3d = env_base.copy()
        env_3d["PYTHONPATH"] = os.pathsep.join([str(PROJ_ROOT), env_3d.get("PYTHONPATH", "")])
        cmd, _ = conda_cmd(
            PCDET_ENV,
            PROJ_ROOT / "scripts" / "visualize_planview.py",
            cfg_path,
            env_3d,
        )
        run(cmd, env_3d)
    else:
        print("[SKIP] visualization")

    # ========== STEP 5/5: Evaluation ==========
    print("========== STEP 5/5: Evaluation (PCDet env) ==========")
    if not args.no_eval:
        env_3d = env_base.copy()
        env_3d["PYTHONPATH"] = os.pathsep.join([str(PROJ_ROOT), env_3d.get("PYTHONPATH", "")])
        cmd, _ = conda_cmd(
            PCDET_ENV,
            PROJ_ROOT / "scripts" / "evaluate_no_stop_violation.py",
            cfg_path,
            env_3d,
        )
        run(cmd, env_3d)
    else:
        print("[SKIP] evaluation")

    print("\n[OK] Pipeline completed.")


if __name__ == "__main__":
    main()
