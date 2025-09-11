# -*- coding: utf-8 -*-
"""
visualize_planview.py
Dependencies: numpy, matplotlib, Pillow, pyyaml, tqdm
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# -------------------- Basic I/O --------------------
def load_yaml(p: str) -> Dict[str, Any]:
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json(p: str) -> Dict[str, Any]:
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def load_all_jsons(dir_path: str, suffix: str) -> List[Dict[str, Any]]:
    if not dir_path:
        return []
    p = Path(dir_path)
    if not p.exists():
        return []
    out = []
    for f in sorted(p.glob(f'*{suffix}')):
        try:
            out.append(load_json(str(f)))
        except Exception as e:
            print(f"[WARN] cannot read {f}: {e}")
    return out


# -------------------- Point cloud loading --------------------
def load_points(points_path: str, max_points: int = 200000) -> Optional[np.ndarray]:
    """
    Return Nx3 (x,y,z); supports .bin/.npy; .pcd requires open3d (if not installed, skip).
    """
    if not points_path:
        return None
    p = Path(points_path)
    if not p.exists():
        return None
    ext = p.suffix.lower()
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
        elif ext == '.pcd':
            try:
                import open3d as o3d
                pc = o3d.io.read_point_cloud(str(p))
                pts = np.asarray(pc.points, dtype=np.float32)
            except Exception:
                return None
        else:
            return None
    except Exception:
        return None

    if pts is None or pts.size == 0:
        return None

    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx, :]
    return pts


# -------------------- Geometry --------------------
def rot_corners_xy(x: float, y: float, dx: float, dy: float, yaw: float) -> np.ndarray:
    """Generate (4,2) top-view corners (clockwise) from x,y,dx,dy,yaw."""
    hx, hy = dx / 2.0, dy / 2.0
    base = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return base @ R.T + np.array([x, y], dtype=np.float32)


# -------------------- Polygon / clipping / IoU --------------------
def _polygon_area(poly: np.ndarray) -> float:
    """Polygon area (works for CW/CCW). poly: (N,2)"""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _suth_hodg_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """
    Sutherland–Hodgman polygon clipping: return subject ∩ clipper
    (clipper must be convex).
    """
    def is_inside(p, a, b):
        # left side of a->b is inside
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-12

    def intersect(p1, p2, a, b):
        s10 = p2 - p1
        s32 = b - a
        denom = s10[0]*s32[1] - s10[1]*s32[0]
        if abs(denom) < 1e-12:
            return p2  # nearly parallel: return endpoint for robustness
        t = ((a[0]-p1[0])*s32[1] - (a[1]-p1[1])*s32[0]) / denom
        return p1 + t*s10

    output = subject.copy()
    if output.shape[0] == 0:
        return output
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i+1) % len(clipper)]
        input_list = output
        output = []
        if len(input_list) == 0:
            break
        S = input_list[-1]
        for E in input_list:
            if is_inside(E, a, b):
                if not is_inside(S, a, b):
                    output.append(intersect(S, E, a, b))
                output.append(E)
            elif is_inside(S, a, b):
                output.append(intersect(S, E, a, b))
            S = E
        output = np.array(output, dtype=np.float32)
    if output is None or len(output) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return output

def _poly_iou(polyA: np.ndarray, polyB: np.ndarray) -> float:
    """Polygon IoU (robust: try both clipping orders)."""
    inter = _suth_hodg_clip(polyA, polyB)
    if inter.shape[0] == 0:
        inter = _suth_hodg_clip(polyB, polyA)
    if inter.shape[0] == 0:
        return 0.0
    aA = abs(_polygon_area(polyA))
    aB = abs(_polygon_area(polyB))
    aI = abs(_polygon_area(inter))
    denom = aA + aB - aI + 1e-12
    return float(aI / denom)

def fraction_inside_chamber(poly: np.ndarray, x_range: Tuple[float,float], y_range: Tuple[float,float]) -> float:
    """Return the fraction of polygon area inside the chamber rectangle [0,1]."""
    x_min, x_max = float(min(x_range)), float(max(x_range))
    y_min, y_max = float(min(y_range)), float(max(y_range))
    chamber_rect = np.array([[x_min, y_min],
                             [x_max, y_min],
                             [x_max, y_max],
                             [x_min, y_max]], dtype=np.float32)
    inter = _suth_hodg_clip(poly, chamber_rect)
    a_poly = abs(_polygon_area(poly))
    if a_poly < 1e-9:
        return 0.0
    a_in = abs(_polygon_area(inter)) if inter.shape[0] > 0 else 0.0
    return float(a_in / a_poly)

def rotated_iou_bev(b1: List[float], b2: List[float]) -> float:
    """b=[x,y,z,dx,dy,dz,yaw] → BEV rotated IoU based on polygon clipping."""
    if not (isinstance(b1, (list, tuple)) and isinstance(b2, (list, tuple)) and len(b1) == 7 and len(b2) == 7):
        return 0.0
    x1, y1, _, dx1, dy1, _, yaw1 = map(float, b1)
    x2, y2, _, dx2, dy2, _, yaw2 = map(float, b2)
    poly1 = rot_corners_xy(x1, y1, dx1, dy1, yaw1).astype(np.float32)
    poly2 = rot_corners_xy(x2, y2, dx2, dy2, yaw2).astype(np.float32)
    return _poly_iou(poly1, poly2)


# -------------------- Per-box area-fraction filter (ships only) & merge --------------------
def _box_has_valid7d(d):
    b = d.get('box7d')
    return isinstance(b, (list, tuple)) and len(b) == 7

def merge_3d_with_per_box_area_filter(
    det3d_list: List[Dict[str, Any]],
    ship_labels: set,
    gate_label: str,
    chamber_x_range: Tuple[float, float],
    chamber_y_range: Optional[Tuple[float, float]],
    outside_frac_thr: float = 0.10
) -> Dict[str, Any]:
    """
    Merge multiple det3d.json files and apply a per-box area-fraction filter (ships only):
    - For each ship detection, compute the fraction of its rotated box area inside the chamber;
      if (1 - fraction_inside) > outside_frac_thr, drop the box.
    - Keep gate/other categories (no area filter).
    - points_path is taken as the first non-empty one.
    Note: if chamber_y_range is missing, skip filtering (merge only).
    """
    out: Dict[str, Any] = {}
    do_filter = chamber_y_range is not None

    for src in det3d_list:
        for fid, rec in src.items():
            tgt = out.setdefault(fid, {'points_path': rec.get('points_path', ''), 'detections': []})
            if not tgt.get('points_path') and rec.get('points_path'):
                tgt['points_path'] = rec['points_path']

            dets = rec.get('detections', []) or []
            for d in dets:
                lbl = d.get('label', '')
                if not _box_has_valid7d(d):
                    # No valid box7d — keep (or drop, if you prefer)
                    tgt['detections'].append(d)
                    continue

                if do_filter and (lbl in ship_labels):
                    x, y, _, dx, dy, _, yaw = [float(t) for t in d['box7d']]
                    poly = rot_corners_xy(x, y, dx, dy, yaw)
                    fin = fraction_inside_chamber(poly, chamber_x_range, chamber_y_range)
                    outside_frac = 1.0 - fin
                    if outside_frac > outside_frac_thr:
                        # Drop this ship box
                        continue

                # Gate/others or filtering disabled → keep
                tgt['detections'].append(d)

    return out



# -------------------- Class-wise de-dup (rotated IoU NMS per label) --------------------
def nms_rotated_by_label(
    dets3: List[Dict[str, Any]],
    iou_thresh: float = 0.30
) -> List[Dict[str, Any]]:
    """
    Perform de-dup across all classes: group by label and run rotated-quadrilateral IoU NMS,
    keeping the highest-score set per class.
    """
    # Group by label
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets3:
        lbl = d.get('label', '__UNK__')
        by_label.setdefault(lbl, []).append(d)

    kept_all: List[Dict[str, Any]] = []

    for lbl, lst in by_label.items():
        enriched = []
        for d in lst:
            b = d.get('box7d')
            if not (isinstance(b, (list, tuple)) and len(b) == 7):
                continue
            poly = rot_corners_xy(float(b[0]), float(b[1]), float(b[3]), float(b[4]), float(b[6])).astype(np.float32)
            sc = float(d.get('score', 0.0))
            enriched.append((d, poly, sc))

        # Sort by score descending
        enriched.sort(key=lambda x: x[2], reverse=True)

        kept = []
        while enriched:
            best = enriched.pop(0)
            kept.append(best)
            remain = []
            for item in enriched:
                iou = _poly_iou(best[1], item[1])
                if iou <= iou_thresh:
                    remain.append(item)
            enriched = remain

        kept_all.extend([k[0] for k in kept])

    return kept_all


# --------- Violation matching (idx + box7d nearest-neighbor) ---------
def collect_violation_indices(
    dets: List[Dict[str, Any]],
    alerts_list: List[Dict[str, Any]],
    idx_set: set,
    center_tol: float = 1.5,   # center distance tolerance (meters)
    size_tol: float = 3.0,     # L1 size tolerance (meters)
    yaw_tol: float = 0.35      # yaw tolerance (radians, ~20°)
) -> set:
    """
    Return a set of detection indices to highlight:
    - Start with idx already present in alerts
    - For each alert carrying box7d, match against dets via nearest neighbor
      (center/size/yaw). Add the matched index if within tolerances.
    """
    viol = set(idx_set) if idx_set else set()
    if not alerts_list:
        return viol

    # Pre-extract det params
    det_params = []
    for i, d in enumerate(dets):
        b = d.get('box7d')
        if not (isinstance(b, (list, tuple)) and len(b) == 7):
            det_params.append(None)
            continue
        x, y, z, dx, dy, dz, yaw = map(float, b)
        det_params.append((i, x, y, dx, dy, yaw))

    for a in alerts_list:
        if 'idx' in a:
            try:
                viol.add(int(a['idx']))
            except Exception:
                pass

        b2 = a.get('box7d')
        if not (isinstance(b2, (list, tuple)) and len(b2) == 7):
            continue
        ax_, ay_, _, adx_, ady_, _, ayaw_ = map(float, b2)

        best_i, best_score = None, 1e9
        for p in det_params:
            if p is None:
                continue
            i, x, y, dx, dy, yaw = p

            # Center distance
            d_center = ((x - ax_)**2 + (y - ay_)**2) ** 0.5
            if d_center > center_tol:
                continue

            # Size similarity (L1)
            d_size = abs(dx - adx_) + abs(dy - ady_)
            if d_size > size_tol:
                continue

            # Yaw similarity (2π periodic)
            dyaw = abs((yaw - ayaw_ + math.pi) % (2*math.pi) - math.pi)
            if dyaw > yaw_tol:
                continue

            score = d_center + 0.2 * d_size + 0.5 * dyaw
            if score < best_score:
                best_score = score
                best_i = i

        if best_i is not None:
            viol.add(best_i)

    return viol


# -------------------- 2D image annotation --------------------
def _load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf"
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_2d_annotated(image_path: str,
                      dets: List[Dict[str,Any]],
                      ship_labels: set,
                      gate_label: str,
                      font_size: int = 16,
                      lw_ship: int = 3,
                      lw_gate: int = 3,
                      lw_other: int = 2) -> Optional[Image.Image]:
    """Overlay boxes on the raw image (ships blue, gate orange, others gray; do not mark red violations). Returns PIL.Image."""
    if not image_path or not Path(image_path).exists():
        return None
    try:
        im = Image.open(image_path).convert('RGB')
    except Exception:
        return None

    draw = ImageDraw.Draw(im)
    font = _load_font(font_size)

    col_ship = (31, 119, 180)   # blue
    col_gate = (255, 127, 14)   # orange
    col_other = (127, 127, 127) # gray

    for d in dets:
        bb = d.get('bbox_xyxy'); lbl = d.get('label',''); sc = float(d.get('score', 0.0))
        if not (isinstance(bb,(list,tuple)) and len(bb)==4):
            continue
        x1,y1,x2,y2 = [float(t) for t in bb]
        if lbl in ship_labels:
            color = col_ship; w = lw_ship
        elif lbl == gate_label:
            color = col_gate; w = lw_gate
        else:
            color = col_other; w = lw_other

        draw.rectangle([x1,y1,x2,y2], outline=color, width=int(w))
        txt = f"{lbl} {sc:.2f}"

        # Text size (compatible with different Pillow versions)
        try:
            bbox = font.getbbox(txt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = font.getsize(txt)
            except Exception:
                tw, th = len(txt) * font.size, font.size

        # Text background & text
        bg = [x1 + 2, y1 + 2, x1 + 2 + tw + 4, y1 + 2 + th + 2]
        draw.rectangle(bg, fill=(255, 255, 255))
        draw.text((x1 + 4, y1 + 3), txt, fill=color, font=font)

    return im


# -------------------- Merge/index (2D + violation index) --------------------
def merge_2d(det2d_list: List[Dict[str,Any]]) -> Dict[str, Any]:
    """Merge multiple det2d.json: concatenate detections; lock_gate_present is OR-reduced; image_path prefers the first."""
    out: Dict[str, Any] = {}
    for d in det2d_list:
        for fid, rec in d.items():
            tgt = out.setdefault(fid, {'image_path': rec.get('image_path',''),
                                       'detections': [],
                                       'lock_gate_present': False})
            if not tgt.get('image_path') and rec.get('image_path'):
                tgt['image_path'] = rec['image_path']
            dets = rec.get('detections', []) or []
            tgt['detections'].extend(dets)
            if rec.get('lock_gate_present') is True:
                tgt['lock_gate_present'] = True
    return out

def build_violation_index(alerts: Dict[str, Any]) -> Dict[str, set]:
    """alerts_by_frame.json -> { frame_id: set(idx) }"""
    out = {}
    if not alerts:
        return out
    for fid, items in alerts.items():
        s = set()
        for it in items:
            if 'idx' in it:
                try:
                    s.add(int(it['idx']))
                except Exception:
                    pass
        out[fid] = s
    return out


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--det3d_dir', default='outputs/det3d', help='directory of *.det3d.json')
    ap.add_argument('--det2d_dir', default='outputs/det2d', help='directory of *.det2d.json')
    ap.add_argument('--alerts',   default='outputs/alerts/alerts_by_frame.json',
                    help='alerts_by_frame.json; optional, ignored if missing')

    ap.add_argument('--frames', default='', help='comma-separated frame_ids to render only')
    ap.add_argument('--max_frames', type=int, default=0, help='limit number of frames (0=all)')

    ap.add_argument('--out_dir', default='outputs/viz_plan', help='output directory for rendered images')
    ap.add_argument('--dpi', type=int, default=150)

    # Plan-view coordinate range (default 0~102.4; can be overridden)
    ap.add_argument('--xlim', type=float, nargs=2, default=[0.0, 102.4])
    ap.add_argument('--ylim', type=float, nargs=2, default=[0.0, 102.4])

    # Point cloud rendering
    ap.add_argument('--max_points', type=int, default=200000)
    ap.add_argument('--point_size', type=float, default=0.25)
    ap.add_argument('--point_alpha', type=float, default=0.35)

    # Style
    ap.add_argument('--font_size', type=int, default=16)
    ap.add_argument('--lw_ship', type=float, default=3.0)
    ap.add_argument('--lw_gate', type=float, default=3.5)
    ap.add_argument('--lw_other', type=float, default=2.0)
    ap.add_argument('--lw_violate', type=float, default=4.0)

    # 2D thumbnail placement in 3D coordinates (data coordinate system)
    ap.add_argument('--thumb_rect', type=float, nargs=4, default=[82.0, 102.0, 82.0, 102.0],
                    help='x0 x1 y0 y1 for 2D thumbnail placement in data coordinates')

    # Whether to draw the gate y-band (light orange)
    ap.add_argument('--draw_gate_band', action='store_false', default=True,
                    help='disable gate band drawing by --no-draw_gate_band')

    # Filtering / de-dup thresholds
    ap.add_argument('--outside_frac', type=float, default=0.10,
                    help='per-box outside-area ratio threshold (> value means too much area outside chamber)')
    ap.add_argument('--lowq_source_ratio_thr', type=float, default=0.10,
                    help='source-level low-quality threshold (not used in this script; kept for compatibility)')
    ap.add_argument('--nms_iou', type=float, default=0.30,
                    help='rotated IoU NMS threshold (class-wise de-dup)')

    args = ap.parse_args()

    # Labels / geometry
    cfg = load_yaml(args.cfg)
    labels = cfg['labels']
    gate_label = labels['lock_gate']
    ship_labels = set(labels['ships'])

    geom = cfg.get('geometry', {}) or {}
    chamber_x = geom.get('chamber_x_range', [39.7, 62.7])
    x_min_ch, x_max_ch = float(chamber_x[0]), float(chamber_x[1])
    gate_y_range = geom.get('gate_y_range', [19.078, 23.002])
    gate_y0, gate_y1 = float(gate_y_range[0]), float(gate_y_range[1])
    stop_y = float(geom.get('stop_line_y', 48.002))
    no_band = geom.get('no_stop_y_range', [23.002, 48.002])
    yb0, yb1 = float(no_band[0]), float(no_band[1])

    # Area-based filtering needs the chamber y-range
    chamber_y_range = tuple(geom.get('chamber_y_range')) if geom.get('chamber_y_range', None) is not None else None
    if chamber_y_range is None:
        print("[WARN] geometry.chamber_y_range not set; skip 'area-fraction' low-quality filtering.")

    # Load & merge (with area-fraction filtering for ships only)
    det3d_all = load_all_jsons(args.det3d_dir, '.det3d.json')
    det2d_all = load_all_jsons(args.det2d_dir, '.det2d.json')
    if not det3d_all:
        raise RuntimeError(f"No det3d jsons in {args.det3d_dir}")
    if not det2d_all:
        print(f"[WARN] No det2d jsons in {args.det2d_dir}")

    det3d = merge_3d_with_per_box_area_filter(
        det3d_all, ship_labels, gate_label,
        chamber_x_range=(x_min_ch, x_max_ch),
        chamber_y_range=chamber_y_range,
        outside_frac_thr=args.outside_frac
    )

    det2d = merge_2d(det2d_all)

    # Alerts (optional)
    alerts = load_json(args.alerts) if (args.alerts and Path(args.alerts).exists()) else {}
    viol_idx_map = build_violation_index(alerts)

    # Frame list
    fids = sorted(det3d.keys())
    if args.frames:
        wanted = {s.strip() for s in args.frames.split(',') if s.strip()}
        fids = [fid for fid in fids if fid in wanted]
    if args.max_frames > 0:
        fids = fids[:args.max_frames]

    ensure_dir(args.out_dir)

    # Colors
    col_ship = '#1f77b4'
    col_ship_violate = '#d62728'
    col_gate = '#ff7f0e'
    col_other = '#7f7f7f'
    pt_color = '#000000'

    # 2D thumbnail region (clamped to axes limits)
    thumb_x0, thumb_x1, thumb_y0, thumb_y1 = args.thumb_rect
    thumb_x0 = max(args.xlim[0], min(args.xlim[1], thumb_x0))
    thumb_x1 = max(args.xlim[0], min(args.xlim[1], thumb_x1))
    thumb_y0 = max(args.ylim[0], min(args.ylim[1], thumb_y0))
    thumb_y1 = max(args.ylim[0], min(args.ylim[1], thumb_y1))

    for fid in tqdm(fids, desc="Rendering frames"):
        rec3 = det3d.get(fid, {})
        dets3 = rec3.get('detections', []) or []
        pts_path = rec3.get('points_path', '')

        rec2 = det2d.get(fid, {})
        dets2 = rec2.get('detections', []) or []
        img_path = rec2.get('image_path', '')
        lgp = bool(rec2.get('lock_gate_present', False))

        # — Class-wise de-dup (rotated IoU NMS per label) —
        dets3 = nms_rotated_by_label(dets3, iou_thresh=args.nms_iou)

        # --------- 1) 3D plan view ---------
        fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)  # square canvas for visual 1:1

        # Point cloud
        pts = load_points(pts_path, max_points=args.max_points)
        if pts is not None and pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=args.point_size, c=pt_color, alpha=args.point_alpha, linewidths=0)

        # Geometry elements (restricted to chamber x-range)
        # No-stop rectangular band
        band_rect = Rectangle(
            (x_min_ch, yb0),
            width=(x_max_ch - x_min_ch),
            height=(yb1 - yb0),
            facecolor='red', alpha=0.08, edgecolor='none', zorder=1
        )
        ax.add_patch(band_rect)

        # Optional: gate y-band (light orange)
        if args.draw_gate_band:
            gate_rect = Rectangle(
                (x_min_ch, gate_y0),
                width=(x_max_ch - x_min_ch),
                height=(gate_y1 - gate_y0),
                facecolor='#ff7f0e', alpha=0.05, edgecolor='none', zorder=1
            )
            ax.add_patch(gate_rect)

        # Stop line (within chamber)
        ax.hlines(stop_y, x_min_ch, x_max_ch, colors='red', linestyles='--', linewidth=2.2, zorder=2)
        # Chamber x-range vertical lines
        ax.vlines([x_min_ch, x_max_ch], args.ylim[0], args.ylim[1],
                  colors='gray', linestyles=':', linewidth=1.8, zorder=2)

        # Violation indices (combine idx + box7d NN matching)
        idx_set = set(viol_idx_map.get(fid, set()))
        viol_set = collect_violation_indices(
            dets3,
            alerts.get(fid, []) if isinstance(alerts, dict) else [],
            idx_set,
            center_tol=1.5,
            size_tol=3.0,
            yaw_tol=0.35
        )

        # Draw 3D boxes
        for i, d in enumerate(dets3):
            b = d.get('box7d'); lbl = d.get('label','')
            if not (isinstance(b, (list, tuple)) and len(b) == 7):
                continue
            poly = rot_corners_xy(float(b[0]), float(b[1]), float(b[3]), float(b[4]), float(b[6]))

            if lbl in ship_labels:
                ec = col_ship_violate if i in viol_set else col_ship
                lw = args.lw_violate if i in viol_set else args.lw_ship
            elif lbl == gate_label:
                ec = col_gate
                lw = args.lw_gate
            else:
                ec = col_other
                lw = args.lw_other

            patch = Polygon(poly, closed=True, fill=False, edgecolor=ec, linewidth=lw)
            ax.add_patch(patch)
            ax.plot([b[0]], [b[1]], marker='o', ms=3, color=ec)
            ax.text(b[0], b[1], lbl, fontsize=args.font_size, color=ec)

        ax.set_title(f"Plan View | frame {fid} | lock_gate_present(2D): {lgp}", fontsize=args.font_size+1)
        ax.set_xlabel('X (m)', fontsize=args.font_size)
        ax.set_ylabel('Y (m)', fontsize=args.font_size)
        ax.tick_params(labelsize=max(8, args.font_size-2))
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)
        ax.set_aspect('equal')       # 1:1 data aspect ratio
        try:
            ax.set_box_aspect(1)     # square axes (Matplotlib>=3.3)
        except Exception:
            pass
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)

        # --------- 2) 2D thumbnail overlay (in data coordinates) ---------
        pil2 = draw_2d_annotated(
            img_path, dets2, ship_labels, gate_label,
            font_size=args.font_size,
            lw_ship=int(round(args.lw_ship)),
            lw_gate=int(round(args.lw_gate)),
            lw_other=int(round(args.lw_other))
        )
        if pil2 is not None:
            pil2r = pil2.rotate(90, expand=True)  # rotate CCW 90°
            region_w = thumb_x1 - thumb_x0
            region_h = thumb_y1 - thumb_y0
            img_w, img_h = pil2r.size
            img_aspect = img_w / img_h
            region_aspect = region_w / region_h

            if img_aspect >= region_aspect:
                draw_w = region_w
                draw_h = region_w / img_aspect
                x0, x1 = thumb_x0, thumb_x1
                y_center = 0.5 * (thumb_y0 + thumb_y1)
                y0, y1 = y_center - draw_h / 2.0, y_center + draw_h / 2.0
            else:
                draw_h = region_h
                draw_w = region_h * img_aspect
                y0, y1 = thumb_y0, thumb_y1
                x_center = 0.5 * (thumb_x0 + thumb_x1)
                x0, x1 = x_center - draw_w / 2.0, x_center + draw_w / 2.0

            thumb_np = np.asarray(pil2r)
            ax.imshow(thumb_np, extent=[x0, x1, y0, y1], origin='upper',
                      zorder=20, interpolation='bilinear', aspect='auto')

            # Thumbnail border
            ax.add_patch(Rectangle((thumb_x0, thumb_y0),
                                   width=(thumb_x1 - thumb_x0), height=(thumb_y1 - thumb_y0),
                                   fill=False, edgecolor='black', linewidth=1.2, zorder=21))

        # Save
        out_path = os.path.join(args.out_dir, f'{fid}.png')
        ensure_dir(args.out_dir)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)


if __name__ == '__main__':
    main()

