# -*- coding: utf-8 -*-
"""
No-stop band violation evaluation (frame-level decision)
- Predicted violation boxes: use only alerts_by_frame.json
- Predicted gate presence: use only pred_gate_presence.json (config key: eval.pred_gate_presence)
- Evaluation: one decision per frame (whether any violation exists), no per-box IoU matching
- Frame set to evaluate: prefer all frames covered by det3d (collected from points_dir); if unavailable, fall back to the union of GT/Pred
- Visualization: problem frames only (BEV point cloud + top-right raw image thumbnail, no 2D boxes; keep GT/Pred boxes; do not use “unmatched”)

Command:
  python scripts/evaluate_no_stop_violation.py --cfg configs/config.yaml

Optional:
  --out_dir eval_out
  --iou 0.70
  --require_gate True         # kept for CLI compatibility; no longer affects evaluation
  --viz_dir <dir>             # default <out_dir>/viz_problem
  --viz_max 0                 # 0 = all problem frames; >0 export at most this many
  --xlim 0 102.4 --ylim 0 102.4
  --thumb_rect 82 102 82 102
  --debug --export_lists

Dependencies: numpy, matplotlib, Pillow, pyyaml, tqdm
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image

# -------------------- Basic I/O --------------------

def load_yaml(p: str) -> Dict[str, Any]:
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json(p: str) -> Any:
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

# -------------------- Frame ID normalization --------------------

def infer_width_from_kitti(gt_dir: str) -> int:
    files = list(Path(gt_dir).glob('*.txt'))
    return max((len(p.stem) for p in files), default=6)

def normalize_fid(fid: str, width: int) -> str:
    s = str(fid).split('.')[0].strip()
    return s.zfill(width) if s.isdigit() else s

# -------------------- Geometry / IoU (kept for visualization) --------------------

def rot_corners_xy(x: float, y: float, dx: float, dy: float, yaw: float) -> np.ndarray:
    hx, hy = dx / 2.0, dy / 2.0
    base = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return base @ R.T + np.array([x, y], dtype=np.float32)

def _sat_axes(poly: np.ndarray) -> np.ndarray:
    e = np.roll(poly, -1, axis=0) - poly
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    l = np.linalg.norm(n, axis=1) + 1e-12
    return n / l[:, None]

def _sat_polys_intersect(P: np.ndarray, Q: np.ndarray) -> bool:
    for a in np.vstack((_sat_axes(P), _sat_axes(Q))):
        p = P @ a; q = Q @ a
        if p.max() < q.min() - 1e-8 or q.max() < p.min() - 1e-8:
            return False
    return True

def _poly_area(poly: np.ndarray) -> float:
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _line_intersection(p1, p2, q1, q2):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float32)

def _suth_hodg_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    def inside(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9
    def intersect(a, b, p, q):
        return _line_intersection(a, b, p, q)
    output = subject.copy()
    for i in range(len(clipper)):
        a = clipper[i]; b = clipper[(i + 1) % len(clipper)]
        input_list = output
        if input_list.size == 0:
            break
        output = []
        s = input_list[-1]
        for e in input_list:
            if inside(a, b, e):
                if not inside(a, b, s):
                    ip = intersect(s, e, a, b)
                    if ip is not None:
                        output.append(ip)
                output.append(e)
            elif inside(a, b, s):
                ip = intersect(s, e, a, b)
                if ip is not None:
                    output.append(ip)
            s = e
        output = np.array(output, dtype=np.float32)
    return output

def poly_intersection(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return _suth_hodg_clip(P, Q)

def poly_iou(P: np.ndarray, Q: np.ndarray) -> float:
    inter = poly_intersection(P, Q)
    if inter.size == 0:
        return 0.0
    ai = abs(_poly_area(inter))
    a1 = abs(_poly_area(P))
    a2 = abs(_poly_area(Q))
    return float(ai / (a1 + a2 - ai + 1e-8))

# -------------------- KITTI parsing --------------------

_FLOAT_RE = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"

def parse_kitti_line(line: str):
    ps = line.strip().split()
    if not ps:
        return None
    label = ps[0]
    try:
        h, w, l = map(float, ps[8:11])
        x, y, z = map(float, ps[11:14])
        yaw = float(ps[14])
        return {"label": label, "box7d": [x, y, z, l, w, h, yaw]}
    except Exception:
        nums = re.findall(_FLOAT_RE, line)
        if len(nums) >= 7:
            try:
                h, w, l, x, y, z, yaw = list(map(float, nums[-7:]))
                return {"label": label, "box7d": [x, y, z, l, w, h, yaw]}
            except Exception:
                return None
        return None

def load_kitti_dir(gt_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted(Path(gt_dir).glob('*.txt')):
        fid = p.stem
        items = []
        with open(p, 'r', encoding='utf-8') as f:
            for ln in f:
                rec = parse_kitti_line(ln)
                if rec:
                    items.append(rec)
        out[fid] = items
    return out

# -------------------- GT construction (any-overlap rule) --------------------

def overlaps_no_stop(box7d, chamber_x, no_stop_y) -> bool:
    x, y, z, dx, dy, dz, yaw = box7d
    ship_poly = rot_corners_xy(x, y, dx, dy, yaw)
    band_poly = np.array([
        [chamber_x[0], no_stop_y[0]],
        [chamber_x[1], no_stop_y[0]],
        [chamber_x[1], no_stop_y[1]],
        [chamber_x[0], no_stop_y[1]],
    ], dtype=np.float32)
    return _sat_polys_intersect(ship_poly, band_poly)

def build_gt_violations(kitti: Dict[str, List[Dict[str, Any]]], gate_label: str, ship_labels: set,
                        chamber_x: Tuple[float, float], no_stop_y: Tuple[float, float], require_gate: bool = True):
    gt_viols: Dict[str, List[List[float]]] = {}
    gate_present: Dict[str, bool] = {}
    for fid, items in kitti.items():
        present = any(it.get('label') == gate_label for it in items)
        gate_present[fid] = present
        if require_gate and not present:
            continue
        vec = []
        for it in items:
            if it.get('label') in ship_labels:
                b = it.get('box7d')
                if isinstance(b, (list, tuple)) and len(b) == 7 and overlaps_no_stop(b, chamber_x, no_stop_y):
                    vec.append([float(t) for t in b])
        if vec:
            gt_viols[fid] = vec
    return gt_viols, gate_present

# -------------------- Prediction loading --------------------

def load_pred_from_alerts(alerts: Dict[str, Any], width: int,
                          gate_present: Dict[str, bool], require_gate: bool = True) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    for raw_fid, arr in (alerts or {}).items():
        fid = normalize_fid(raw_fid, width)
        if require_gate and not gate_present.get(fid, False):
            continue
        boxes = []
        for a in (arr or []):
            if isinstance(a, dict) and isinstance(a.get('box7d'), (list, tuple)) and len(a['box7d']) == 7:
                boxes.append([float(t) for t in a['box7d']])
        if boxes:
            out[fid] = boxes
    return out

def load_gate_presence_map(path: str, width: int, all_fids: List[str]) -> Dict[str, bool]:
    """
    Load predicted gate-presence booleans from pred_gate_presence.json.
    Supports two structures:
      1) { "000000": true, "000001": false, ... }
      2) [ {"frame_id": "...", "present": true}, ... ] (compatible)
    Missing entries default to False.
    """
    mapping: Dict[str, bool] = {}
    if path and Path(path).exists():
        raw = load_json(path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                mapping[normalize_fid(k, width)] = bool(v)
        elif isinstance(raw, list):
            for d in raw:
                if isinstance(d, dict) and 'frame_id' in d and 'present' in d:
                    mapping[normalize_fid(str(d['frame_id']), width)] = bool(d['present'])
        else:
            print(f"[WARN] Unknown pred_gate_presence.json format: {type(raw)}; defaulting to False.")
    else:
        print("[WARN] pred_gate_presence.json not provided or path does not exist; Pred Gate defaults to False.")
    return {fid: mapping.get(fid, False) for fid in all_fids}

# -------------------- 2D / point cloud I/O (thumbnail only) --------------------

def load_points_from_any(points_dir: str, fid: str, max_points: int = 200000) -> Optional[np.ndarray]:
    if not points_dir:
        return None
    base = Path(points_dir)
    if not base.exists():
        return None
    for ext in ['.bin', '.npy', '.pcd']:
        p = base / f"{fid}{ext}"
        if p.exists():
            try:
                if ext == '.bin':
                    a = np.fromfile(str(p), dtype=np.float32)
                    if a.size % 4 != 0:
                        return None
                    pts = a.reshape(-1, 4)[:, :3]
                elif ext == '.npy':
                    arr = np.load(str(p))
                    if arr.ndim != 2 or arr.shape[1] < 3:
                        return None
                    pts = arr[:, :3].astype(np.float32)
                else:  # .pcd
                    try:
                        import open3d as o3d
                        pc = o3d.io.read_point_cloud(str(p))
                        pts = np.asarray(pc.points, dtype=np.float32)
                    except Exception:
                        return None
                if pts is None or pts.size == 0:
                    return None
                if max_points > 0 and pts.shape[0] > max_points:
                    idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
                    pts = pts[idx, :]
                return pts
            except Exception:
                return None
    return None

def find_image_path(images_dir: str, fid: str) -> str:
    if images_dir and Path(images_dir).exists():
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            p = Path(images_dir) / f"{fid}{ext}"
            if p.exists():
                return str(p)
    return ''

def collect_frames_from_points_dir(points_dir: str) -> List[str]:
    """
    Collect all frame IDs (file stems) from a point cloud directory; supports .bin/.npy/.pcd
    Do not zero-pad here; call normalize_fid in the main flow.
    """
    if not points_dir or not Path(points_dir).exists():
        return []
    stems = set()
    for ext in ['*.bin', '*.npy', '*.pcd']:
        for p in Path(points_dir).glob(ext):
            stems.add(p.stem)
    return sorted(stems)

# -------------------- Visualization: problem frames --------------------

def viz_problem_frame(fid: str, kitti_items: List[Dict[str, Any]], gate_label: str,
                      chamber_x: Tuple[float, float], gate_y: Tuple[float, float], no_stop_y: Tuple[float, float], stop_line_y: float,
                      g_boxes: List[List[float]], p_boxes: List[List[float]],
                      unmatched_g: List[int], unmatched_p: List[int],
                      gate_gt: bool, gate_pred: bool,
                      images_dir: str, points_dir: str,
                      out_png: str,
                      xlim=(0.0, 102.4), ylim=(0.0, 102.4), dpi=170, font=11,
                      point_size=0.25, point_alpha=0.35,
                      thumb_rect=(82.0, 102.0, 82.0, 102.0)):
    ensure_dir(os.path.dirname(out_png))

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=dpi)

    # Point cloud
    pts = load_points_from_any(points_dir, fid, max_points=200000)
    if pts is not None and pts.size > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c='#000000', alpha=point_alpha, linewidths=0, zorder=0)

    # Geometric background
    ax.add_patch(Rectangle((chamber_x[0], no_stop_y[0]),
                           chamber_x[1]-chamber_x[0], no_stop_y[1]-no_stop_y[0],
                           facecolor='red', alpha=0.08, edgecolor='none', zorder=1))
    if gate_y[0] < gate_y[1]:
        ax.add_patch(Rectangle((chamber_x[0], gate_y[0]),
                               chamber_x[1]-chamber_x[0], gate_y[1]-gate_y[0],
                               facecolor='#ff7f0e', alpha=0.06, edgecolor='none', zorder=1))
    ax.vlines([chamber_x[0], chamber_x[1]], ylim[0], ylim[1], colors='gray', linestyles=':', linewidth=1.6, zorder=2)
    if stop_line_y is not None:
        ax.hlines(stop_line_y, chamber_x[0], chamber_x[1], colors='red', linestyles='--', linewidth=2.0, zorder=2)

    # Gate boxes (orange, GT)
    for it in (kitti_items or []):
        if it.get('label') == gate_label and isinstance(it.get('box7d'), (list, tuple)) and len(it['box7d']) == 7:
            b = it['box7d']
            poly = rot_corners_xy(b[0], b[1], b[3], b[4], b[6])
            ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor='#ff7f0e', linewidth=3.0, zorder=3))

    # GT (green)
    for i, b in enumerate(g_boxes):
        poly = rot_corners_xy(b[0], b[1], b[3], b[4], b[6])
        lw = 3.0 if i in unmatched_g else 2.0
        ls = '--' if i in unmatched_g else '-'
        ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor='#2ca02c', linewidth=lw, linestyle=ls, zorder=4))
        ax.plot([b[0]], [b[1]], marker='o', ms=3, color='#2ca02c', zorder=4)

    # Pred (red)
    for j, b in enumerate(p_boxes):
        poly = rot_corners_xy(b[0], b[1], b[3], b[4], b[6])
        lw = 3.0 if j in unmatched_p else 2.0
        ls = '--' if j in unmatched_p else '-'
        ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor='#d62728', linewidth=lw, linestyle=ls, zorder=5))
        ax.plot([b[0]], [b[1]], marker='x', ms=3, color='#d62728', zorder=5)

    # Top-right: raw image thumbnail only (no boxes)
    img_path = find_image_path(images_dir, fid)
    if img_path:
        try:
            pil2 = Image.open(img_path).convert('RGB')
            pil2r = pil2.rotate(90, expand=True)  # 90° counter-clockwise
            x0, x1, y0, y1 = thumb_rect
            region_w, region_h = (x1 - x0), (y1 - y0)
            img_w, img_h = pil2r.size
            img_aspect = img_w / img_h
            region_aspect = region_w / region_h
            if img_aspect >= region_aspect:
                draw_h = region_w / img_aspect
                x0p, x1p = x0, x1
                y0p = 0.5 * (y0 + y1) - draw_h / 2.0
                y1p = y0p + draw_h
            else:
                draw_w = region_h * img_aspect
                y0p, y1p = y0, y1
                x0p = 0.5 * (x0 + x1) - draw_w / 2.0
                x1p = x0p + draw_w
            thumb_np = np.asarray(pil2r)
            ax.imshow(thumb_np, extent=[x0p, x1p, y0p, y1p], origin='upper', zorder=20, interpolation='bilinear', aspect='auto')
            ax.add_patch(Rectangle((x0, y0), width=(x1 - x0), height=(y1 - y0), fill=False, edgecolor='black', linewidth=1.2, zorder=21))
        except Exception:
            pass

    # Corner annotation
    corner_text = (
        f"Gate (GT): {'✓' if gate_gt else '✗'}    "
        f"Gate (Pred): {'✓' if gate_pred else '✗'}\n"
        f"GT violations ship: {len(g_boxes)}\n"
        f"Pred violations ship: {len(p_boxes)}"
    )
    ax.text(
        0.01, 0.99, corner_text,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=font, color='black',
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.92),
        zorder=50
    )

    # Legend (keep; “unmatched” not used for decisions)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='#2ca02c', lw=2, linestyle='-', label='GT (matched)'),
        Line2D([0], [0], color='#2ca02c', lw=3, linestyle='--', label='GT (unmatched)'),
        Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='Pred (matched)'),
        Line2D([0], [0], color='#d62728', lw=3, linestyle='--', label='Pred (unmatched)'),
        Line2D([0], [0], color='#ff7f0e', lw=3, linestyle='-', label='Gate (GT boxes)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=font-1, frameon=True)

    title = f"Frame {fid} | (Frame-level decision)  (GT=Green, Pred=Red)"
    ax.set_title(title, fontsize=font+1)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)

    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)

# -------------------- Evaluation (frame-level binary classification) + viz --------------------

def evaluate_and_viz(kitti: Dict[str, List[Dict[str, Any]]], gate_label: str,
                     gt_viols: Dict[str, List[List[float]]], pred_viols: Dict[str, List[List[float]]],
                     gate_present: Dict[str, bool], gate_pred_map: Dict[str, bool], iou_thr: float, out_dir: str,
                     chamber_x: Tuple[float, float], gate_y: Tuple[float, float],
                     no_stop_y: Tuple[float, float], stop_line_y: float,
                     data_images_dir: str, data_points_dir: str,
                     xlim: Tuple[float, float], ylim: Tuple[float, float], thumb_rect: Tuple[float, float, float, float],
                     require_gate: bool = False, export_lists: bool = False, debug: bool = False,
                     viz_dir: str = None, viz_max: int = 0,
                     eval_frames: Optional[List[str]] = None):
    """
    Frame-level evaluation: one decision per frame
      - GT_has  = (whether the frame has any GT violation boxes)
      - Pred_has = (whether the frame has any Pred violation boxes)
      - TP: GT_has=True & Pred_has=True
      - FP: GT_has=False & Pred_has=True
      - FN: GT_has=True & Pred_has=False
      - TN: remaining cases (not counted in metrics)
    """
    ensure_dir(out_dir)
    if viz_dir is None:
        viz_dir = os.path.join(out_dir, 'viz_problem')
    ensure_dir(viz_dir)

    # Frame set to evaluate: if not provided, use the union of all keys
    if eval_frames is None:
        eval_frames = sorted(set(list(kitti.keys()) + list(gt_viols.keys()) + list(pred_viols.keys())))

    total_tp = total_fp = total_fn = 0
    per_frame = []
    details = {}
    fp_frames, fn_frames = [], []
    to_viz = []

    for fid in tqdm(eval_frames, desc='Evaluating'):
        g_boxes = gt_viols.get(fid, [])
        p_boxes = pred_viols.get(fid, [])
        GT_has = len(g_boxes) > 0
        Pred_has = len(p_boxes) > 0

        # Frame-level counting
        if GT_has and Pred_has:
            total_tp += 1
            pf, ff = 0, 0
        elif (not GT_has) and Pred_has:
            total_fp += 1
            fp_frames.append(fid); pf, ff = 1, 0
        elif GT_has and (not Pred_has):
            total_fn += 1
            fn_frames.append(fid); pf, ff = 0, 1
        else:
            pf, ff = 0, 0  # TN, not counted

        per_frame.append({
            'frame': fid,
            'GT_has': int(GT_has),
            'Pred_has': int(Pred_has),
            'TP': 1 if (GT_has and Pred_has) else 0,
            'FP': pf,
            'FN': ff,
            'GT_viol_cnt': len(g_boxes),
            'Pred_viol_cnt': len(p_boxes)
        })

        # Whether to dump as a “problem frame”
        gate_pred = gate_pred_map.get(fid, False)
        gate_gt = gate_present.get(fid, False)
        gate_mismatch = (gate_pred != gate_gt)
        if pf or ff or gate_mismatch:
            to_viz.append((fid, g_boxes, p_boxes, [], [], gate_mismatch))

    # Save per-frame metrics (CSV)
    import csv
    csv_path = os.path.join(out_dir, 'per_frame_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'GT_has', 'Pred_has', 'TP', 'FP', 'FN', 'GT_viol_cnt', 'Pred_viol_cnt'])
        w.writeheader(); w.writerows(per_frame)

    # Save empty matching details (for compatibility)
    json_path = os.path.join(out_dir, 'matching_details.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({fid: {'matches': [], 'gt_unmatched': [], 'pred_unmatched': []} for fid in eval_frames},
                  f, ensure_ascii=False, indent=2)

    if export_lists:
        with open(os.path.join(out_dir, 'fp_frames.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(fp_frames))
        with open(os.path.join(out_dir, 'fn_frames.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(fn_frames))

    # Frame-level metrics (rates computed on positive-related frames)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    miss_rate = total_fn / (total_tp + total_fn + 1e-8)
    far       = total_fp / (total_tp + total_fp + 1e-8)

    print("\n===== Frame-level (one decision per frame) =====")
    print("TP: {}  FP: {}  FN: {}".format(total_tp, total_fp, total_fn))
    print("Precision: {:.4f}".format(precision))
    print("Recall   : {:.4f}".format(recall))
    print("F1-score : {:.4f}".format(f1))
    print("Miss Rate: {:.4f}".format(miss_rate))
    print("False Alarm Rate: {:.4f}".format(far))
    print("Per-frame CSV  :", csv_path)
    print("Matching detail:", json_path)

    if debug:
        gate_frames = sum(1 for v in gate_present.values() if v)
        n_gt_frames = sum(1 for _fid in eval_frames if len(gt_viols.get(_fid, [])) > 0)
        n_pred_frames = sum(1 for _fid in eval_frames if len(pred_viols.get(_fid, [])) > 0)
        print("\n[debug] frames(total eval):", len(eval_frames))
        print("[debug] frames(GT gate):", gate_frames)
        print("[debug] frames(GT has viol):", n_gt_frames)
        print("[debug] frames(Pred has viol):", n_pred_frames)
        print(f"[debug] problem frames to viz: {len(to_viz)} (limit: {viz_max or 'all'})")

    # Render problem frames (do not use unmatched; for display only)
    if to_viz:
        if viz_max > 0:
            to_viz = to_viz[:viz_max]
        for fid, g, p, _unG, _unP, _ in tqdm(to_viz, desc='Rendering problem frames'):
            items = kitti.get(fid, [])
            out_png = os.path.join(viz_dir, f'{fid}.png')
            viz_problem_frame(
                fid=fid,
                kitti_items=items,
                gate_label=gate_label,
                chamber_x=chamber_x,
                gate_y=gate_y,
                no_stop_y=no_stop_y,
                stop_line_y=stop_line_y,
                g_boxes=g,
                p_boxes=p,
                unmatched_g=[],  # frame-level eval, not used
                unmatched_p=[],
                gate_gt=gate_present.get(fid, False),
                gate_pred=gate_pred_map.get(fid, False),
                images_dir=data_images_dir,
                points_dir=data_points_dir,
                out_png=out_png,
                xlim=xlim, ylim=ylim, dpi=170, font=11,
                point_size=0.25, point_alpha=0.35,
                thumb_rect=thumb_rect
            )

def dump_json_safe(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[save] {path}")

# -------------------- Main --------------------

def str2bool(s: str) -> bool:
    return str(s).lower() in ['1', 'true', 'yes', 'y', 't']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='configs/config.yaml')
    ap.add_argument('--out_dir', default='outputs/eval_out', help='Output directory')
    ap.add_argument('--iou', type=float, default=0.70, help='BEV IoU threshold (not used in frame-level eval; kept for CLI compatibility)')
    ap.add_argument('--require_gate', type=str2bool, default=True, help='Compatibility flag: no longer used in evaluation')
    ap.add_argument('--debug', action='store_true', help='Print debug stats')
    ap.add_argument('--export_lists', action='store_true', help='Export FP/FN frame lists')
    # Visualization params
    ap.add_argument('--viz_dir', default=None, help='Directory for problem-frame visualizations (default <out_dir>/viz_problem)')
    ap.add_argument('--viz_max', type=int, default=0, help='Max number of problem frames to export (0=all)')
    ap.add_argument('--xlim', type=float, nargs=2, default=[0.0, 102.4])
    ap.add_argument('--ylim', type=float, nargs=2, default=[0.0, 102.4])
    ap.add_argument('--thumb_rect', type=float, nargs=4, default=[82.0, 102.0, 82.0, 102.0], help='Region to place 2D thumbnail: x0 x1 y0 y1')

    ap.add_argument('--export_gt_json', default='outputs/gt/violation_gt.json',
                    help='Export GT-derived violation annotations to JSON (frame->boxes or bool); set empty to disable')
    ap.add_argument('--gt_json_mode', default='boxes', choices=['boxes', 'bool'],
                    help='Export mode: boxes=per-frame list of box7d; bool=per-frame existence (True/False)')
    ap.add_argument('--export_gate_gt', default='outputs/gt/gate_present_gt.json',
                    help='Export GT gate-presence map to JSON; set empty to disable')
    ap.add_argument('--gt_require_gate', type=str2bool, default=True,
                    help='Whether GT violation requires gate present (default True)')

    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    labels = cfg['labels']
    gate_label = labels['lock_gate']
    ship_labels = set(labels['ships'])

    geom = cfg['geometry']
    chamber_x = tuple(map(float, geom['chamber_x_range']))
    no_stop_y = tuple(map(float, geom['no_stop_y_range']))
    gate_y = tuple(map(float, geom.get('gate_y_range', [0, 0])))
    stop_line_y = float(geom.get('stop_line_y', no_stop_y[1] if no_stop_y else 0.0))

    data_conf = cfg.get('data', {}) or {}
    data_images_dir = str(data_conf.get('images_dir', ''))
    data_points_dir = str(data_conf.get('points_dir', ''))

    eval_io = cfg['eval']
    gt_dir = eval_io['kitti_gt_dir']
    alerts_path = eval_io['pred_alerts']
    pred_gate_presence_path = str(eval_io.get('pred_gate_presence', ''))

    # 1) GT (KITTI -> violations) — first do NOT filter by gate to obtain full gate_present
    kitti_raw = load_kitti_dir(gt_dir)
    target_width = infer_width_from_kitti(gt_dir)
    # Normalize kitti keys
    kitti = {normalize_fid(fid, target_width): items for fid, items in kitti_raw.items()}

    gt_viols_raw, gate_present_raw = build_gt_violations(
        kitti_raw, gate_label, ship_labels, chamber_x, no_stop_y,
        require_gate=False
    )
    gt_viols = {normalize_fid(fid, target_width): v for fid, v in gt_viols_raw.items()}
    gate_present = {normalize_fid(fid, target_width): v for fid, v in gate_present_raw.items()}

    # ★ Important fix: GT violations must require gate presence (default True; controlled by --gt_require_gate)
    if args.gt_require_gate:
        gt_viols = {fid: boxes for fid, boxes in gt_viols.items() if gate_present.get(fid, False)}
        # Note: frames not in gt_viols are naturally treated as GT_has=False during evaluation

    # === Export GT violations JSON (optional) ===
    if args.export_gt_json:
        if args.gt_json_mode == 'boxes':
            dump_json_safe(gt_viols, args.export_gt_json)
        else:
            gt_bool = {fid: (len(boxes) > 0) for fid, boxes in gt_viols.items()}
            dump_json_safe(gt_bool, args.export_gt_json)

    # === Export GT gate presence JSON (optional) ===
    if args.export_gate_gt:
        dump_json_safe(gate_present, args.export_gate_gt)

    # 2) Predicted violation boxes (alerts)
    alerts = load_json(alerts_path)
    if not isinstance(alerts, dict):
        print("[WARN] alerts_by_frame.json is expected to be a dict {frame_id: [...]}; trying to adapt.")
        tmp = {}
        if isinstance(alerts, list):
            for d in alerts:
                if isinstance(d, dict) and 'frame_id' in d and 'box7d' in d:
                    fid = normalize_fid(str(d['frame_id']), target_width)
                    tmp.setdefault(fid, []).append({'box7d': d['box7d']})
        alerts = tmp
    # Do NOT filter by gate
    pred_viols_all = load_pred_from_alerts(alerts, width=target_width, gate_present=gate_present, require_gate=False)

    # 3) Frame set from det3d coverage (points_dir)
    points_frames_raw = collect_frames_from_points_dir(data_points_dir)
    if points_frames_raw:
        eval_frames_norm = [normalize_fid(fid, target_width) for fid in points_frames_raw]
    else:
        # Fallback: if points_dir unavailable, use union of GT + Pred
        eval_frames_norm = sorted(list(set(list(kitti.keys()) + list(gt_viols.keys()) + list(pred_viols_all.keys()))))

    # Keep only predictions on frames covered by det3d
    pred_viols = {fid: boxes for fid, boxes in pred_viols_all.items() if fid in set(eval_frames_norm)}

    # 4) Predicted gate presence (from pred_gate_presence only) — restrict to eval_frames_norm
    kitti_keys_norm = list(kitti.keys())
    gate_pred_map_full = load_gate_presence_map(pred_gate_presence_path, target_width, kitti_keys_norm)
    gate_pred_map = {fid: gate_pred_map_full.get(fid, False) for fid in eval_frames_norm}

    if args.debug:
        print('[info] frames(total kitti):', len(kitti))
        print('[info] frames(eval via points_dir):', len(eval_frames_norm))
        print('[info] frames(GT viol):', sum(1 for fid in eval_frames_norm if len(gt_viols.get(fid, [])) > 0))
        print('[info] frames(Pred viol):', sum(1 for fid in eval_frames_norm if len(pred_viols.get(fid, [])) > 0))
        print('[info] frames(Pred gate=True):', sum(1 for fid in eval_frames_norm if gate_pred_map.get(fid, False)))

    # 5) Evaluate + visualize problem frames (frame-level)
    out_dir = args.out_dir
    evaluate_and_viz(
        kitti=kitti, gate_label=gate_label,
        gt_viols=gt_viols, pred_viols=pred_viols,
        gate_present=gate_present, gate_pred_map=gate_pred_map,
        iou_thr=args.iou, out_dir=out_dir,
        chamber_x=chamber_x, gate_y=gate_y, no_stop_y=no_stop_y, stop_line_y=stop_line_y,
        data_images_dir=data_images_dir, data_points_dir=data_points_dir,
        xlim=tuple(args.xlim), ylim=tuple(args.ylim), thumb_rect=tuple(args.thumb_rect),
        require_gate=False, export_lists=args.export_lists, debug=args.debug,
        viz_dir=args.viz_dir, viz_max=args.viz_max,
        eval_frames=eval_frames_norm
    )

if __name__ == '__main__':
    main()

