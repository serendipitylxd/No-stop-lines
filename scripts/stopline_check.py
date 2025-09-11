# -*- coding: utf-8 -*-
"""
stopline_check.py — No-stop line/band check after fusing 2D & 3D results
(includes 3D low-quality box filtering + duplicate removal)

Inputs:
- outputs/det2d/*.det2d.json (from infer_2d_mmdet.py)
- outputs/det3d/*.det3d.json (from infer_3d_pcdet.py; detections only)

New logic:
- For 3D detections (ships/gate), first apply "low-quality box filtering":
  if the area of the rotated box lying outside the chamber rectangle > 10%
  (i.e., inside fraction < 90%), drop it (requires chamber_y_range in config).
- For 3D detections, perform "duplicate removal (same object)": rotated
  quadrilateral IoU NMS in score-desc order (default threshold 0.30);
  keep only the highest-scoring box.

Fusion logic (gate presence):
- Prefer 2D lock_gate_present; if 2D is missing, fall back to
  "existence of filtered 3D gate boxes".

Violation criteria:
- For each frame and each 3D ship box (three ship classes), if:
  a) the ship box intersects with the segment y = stop_line_y (x in chamber_x_range)
     → violation "line_cross"
  b) the ship box overlaps the no-stop band
     (x in chamber_x_range and y in no_stop_y_range)
     → violation "zone_overlap"
  then add an alert.

Outputs:
- outputs/alerts/gate_presence.json
- outputs/alerts/alerts_by_frame.json
- outputs/alerts/alerts_summary.csv
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import pandas as pd

from utils.io_utils import ensure_dir, save_json, list_files_multi


def load_cfg(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def frame_id_from_path(p: str, rgx: str) -> str:
    m = re.match(rgx, Path(p).name)
    if m:
        return m.group(1)
    return Path(p).stem


# ----------------- Geometry utilities -----------------
def rot2d(cos_t: float, sin_t: float, x: float, y: float) -> Tuple[float, float]:
    return cos_t * x - sin_t * y, sin_t * x + cos_t * y

def box2d_corners_xy(x: float, y: float, dx: float, dy: float, yaw: float) -> np.ndarray:
    """
    Generate XY corners (clockwise order) from box7d (x,y,dx,dy,yaw).
    Returns shape = (4, 2).
    """
    hx, hy = dx / 2.0, dy / 2.0
    base = np.array([[-hx, -hy],
                     [ hx, -hy],
                     [ hx,  hy],
                     [-hx,  hy]], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = base @ R.T + np.array([x, y], dtype=np.float32)
    return out  # (4,2)

def segment_intersect(p1, p2, q1, q2) -> bool:
    """Whether segments p1-p2 and q1-q2 intersect."""
    def cross(a, b): return a[0]*b[1] - a[1]*b[0]
    def sub(a, b): return (a[0]-b[0], a[1]-b[1])
    r, s = sub(p2, p1), sub(q2, q1)
    rxs = cross(r, s)
    q_p = sub(q1, p1)
    qpxr = cross(q_p, r)
    if abs(rxs) < 1e-9 and abs(qpxr) < 1e-9:
        # Collinear: check overlap
        def dot(a, b): return a[0]*b[0] + a[1]*b[1]
        rr = dot(r, r)
        t0 = dot(q_p, r) / rr
        t1 = t0 + dot(s, r) / rr
        lo, hi = min(t0, t1), max(t0, t1)
        return hi >= 0 and lo <= 1
    if abs(rxs) < 1e-9 and abs(qpxr) >= 1e-9:
        return False
    t = cross(q_p, s) / rxs
    u = cross(q_p, r) / rxs
    return (0 <= t <= 1) and (0 <= u <= 1)

def poly_intersect_stop_line(poly: np.ndarray, stop_y: float, x_range: Tuple[float, float]) -> bool:
    """
    Whether polygon poly (4,2) intersects the line segment y=stop_y with x∈x_range.
    """
    x1, x2 = float(min(x_range)), float(max(x_range))
    L1 = (x1, stop_y); L2 = (x2, stop_y)
    # polygon edges
    for i in range(len(poly)):
        P1 = (float(poly[i][0]), float(poly[i][1]))
        P2 = (float(poly[(i+1) % len(poly)][0]), float(poly[(i+1) % len(poly)][1]))
        if segment_intersect(P1, P2, L1, L2):
            return True
    return False

def poly_overlaps_band(poly: np.ndarray, x_range: Tuple[float,float], y_range: Tuple[float,float]) -> bool:
    """
    Whether polygon overlaps the rectangle band [x_min,x_max]×[y_min,y_max]
    (coarse test: AABB early-out + corner-inclusion + edge intersection).
    """
    x_min, x_max = float(min(x_range)), float(max(x_range))
    y_min, y_max = float(min(y_range)), float(max(y_range))

    # Early reject: AABBs do not overlap
    poly_xmin, poly_ymin = float(np.min(poly[:,0])), float(np.min(poly[:,1]))
    poly_xmax, poly_ymax = float(np.max(poly[:,0])), float(np.max(poly[:,1]))
    if (poly_xmax < x_min) or (poly_xmin > x_max) or (poly_ymax < y_min) or (poly_ymin > y_max):
        return False

    # Corner tests
    rect_corners = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.float32)

    def point_in_poly(pt, poly_):
        x, y = pt
        inside = False
        n = len(poly_)
        for i in range(n):
            x1,y1 = poly_[i]
            x2,y2 = poly_[(i+1)%n]
            inter = ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12)+x1)
            if inter: inside = not inside
        return inside

    # Any rect corner inside polygon or any polygon corner inside rect → overlap
    if any(point_in_poly(tuple(rc), poly) for rc in rect_corners):
        return True
    if any((x_min <= px <= x_max) and (y_min <= py <= y_max) for px,py in poly):
        return True

    # Edge intersection (4 rect edges vs 4 polygon edges)
    rect_edges = [(tuple(rect_corners[i]), tuple(rect_corners[(i+1)%4])) for i in range(4)]
    for i in range(4):
        p1 = (float(poly[i][0]), float(poly[i][1]))
        p2 = (float(poly[(i+1)%4][0]), float(poly[(i+1)%4][1]))
        for e1,e2 in rect_edges:
            if segment_intersect(p1, p2, e1, e2):
                return True

    return False


# ----------------- Polygon tools (area / clipping / IoU) -----------------
def polygon_area(poly: np.ndarray) -> float:
    """Polygon area (works for CW/CCW), poly shape=(N,2)."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def suth_hodg_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """
    Sutherland–Hodgman polygon clipping: keep subject ∩ clipper.
    Requires convex clipper (both chamber rectangle and rotated boxes are convex).
    Returns (M,2); if disjoint, returns empty array shape=(0,2).
    """
    def is_inside(p, a, b):
        # Left side of edge a->b is considered inside
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-12

    def compute_intersection(p1, p2, a, b):
        # Intersection of segment p1->p2 with infinite line a->b
        s10 = p2 - p1
        s32 = b - a
        denom = s10[0] * s32[1] - s10[1] * s32[0]
        if abs(denom) < 1e-12:
            # Parallel/overlapping; approximate by returning the current endpoint
            return p2
        t = ((a[0] - p1[0]) * s32[1] - (a[1] - p1[1]) * s32[0]) / denom
        return p1 + t * s10

    output = subject.copy()
    if output.shape[0] == 0:
        return output
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        input_list = output
        output = []
        if len(input_list) == 0:
            break
        S = input_list[-1]
        for E in input_list:
            if is_inside(E, a, b):
                if not is_inside(S, a, b):
                    output.append(compute_intersection(S, E, a, b))
                output.append(E)
            elif is_inside(S, a, b):
                output.append(compute_intersection(S, E, a, b))
            S = E
        output = np.array(output, dtype=np.float32)
    return output if output is not None and len(output) > 0 else np.zeros((0,2), dtype=np.float32)

def poly_iou(polyA: np.ndarray, polyB: np.ndarray) -> float:
    inter = suth_hodg_clip(polyA, polyB)
    if inter.shape[0] == 0:
        # Try swapping clipper for numerical robustness
        inter = suth_hodg_clip(polyB, polyA)
    if inter.shape[0] == 0:
        return 0.0
    aA = abs(polygon_area(polyA))
    aB = abs(polygon_area(polyB))
    aI = abs(polygon_area(inter))
    denom = aA + aB - aI + 1e-12
    return float(aI / denom)

def fraction_inside_chamber(poly: np.ndarray, x_range: Tuple[float,float], y_range: Tuple[float,float]) -> float:
    """Fraction (0–1) of polygon area inside the chamber rectangle."""
    x_min, x_max = float(min(x_range)), float(max(x_range))
    y_min, y_max = float(min(y_range)), float(max(y_range))
    chamber_rect = np.array([[x_min, y_min],
                             [x_max, y_min],
                             [x_max, y_max],
                             [x_min, y_max]], dtype=np.float32)
    inter = suth_hodg_clip(poly, chamber_rect)
    a_poly = abs(polygon_area(poly))
    if a_poly < 1e-9:
        return 0.0
    a_in = abs(polygon_area(inter)) if inter.shape[0] > 0 else 0.0
    return float(a_in / a_poly)


# ----------------- I/O & fusion (load directory JSONs) -----------------
def load_jsons_in_dir(d: str, suffix: str) -> List[Dict[str,Any]]:
    """Load all JSON files under directory d whose names end with suffix."""
    files = sorted([str(p) for p in Path(d).glob(f'*{suffix}')])
    out = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as h:
                out.append(json.load(h))
        except Exception as e:
            print(f"[WARN] failed to load {f}: {e}")
    return out


# ----------------- 2D/3D fusion (gate presence: prefer 2D) -----------------
def merge_gate_presence(det2d_dicts: List[Dict[str,Any]],
                        det3d_dicts: List[Dict[str,Any]],
                        lock_gate_label: str) -> Dict[str, bool]:
    """
    gate_present[frame_id] = True/False
    Prefer 2D lock_gate_present; otherwise check whether detections contain label == lock_gate_label
    (In this project, main() passes an empty det3d_dicts to avoid noisy 3D affecting presence;
     we keep the function generic for broader use.)
    """
    pres: Dict[str,bool] = {}

    def scan_and_update(d, from_which):
        for fid, rec in d.items():
            # If lock_gate_present already exists, use it
            lgp = rec.get('lock_gate_present', None)
            if isinstance(lgp, bool):
                pres[fid] = bool(lgp)
                continue
            # Otherwise infer from detections
            dets = rec.get('detections', []) or []
            has = any(isinstance(x, dict) and x.get('label') == lock_gate_label for x in dets)
            if fid not in pres:
                pres[fid] = has
            else:
                pres[fid] = pres[fid] or has

    for d2 in det2d_dicts:
        scan_and_update(d2, '2d')
    for d3 in det3d_dicts:
        scan_and_update(d3, '3d')
    return pres


def process_3d_detections(det3d_dicts: List[Dict[str,Any]],
                          ship_labels: List[str],
                          lock_gate_label: str,
                          chamber_x_range: Tuple[float,float],
                          chamber_y_range: Tuple[float,float] = None,
                          outside_frac_thresh: float = 0.10,
                          iou_thresh: float = 0.30) -> Tuple[Dict[str, List[Dict[str,Any]]], Dict[str, bool]]:
    """
    Returns:
      ships_3d[fid] = filtered & deduplicated ship detections (only labels in ship_labels)
      gate_present_by_3d[fid] = whether filtered 3D gate detections exist

    Notes:
      1) If chamber_y_range is provided, apply "outside fraction" filtering to ship rotated boxes only
         (do NOT filter gate boxes).
      2) Duplicate removal uses rotated-quadrilateral IoU NMS (keep highest score).
    """
    ship_set = set(ship_labels)
    per_frame_all: Dict[str, List[Dict[str,Any]]] = {}

    # 1) Merge all 3D files
    for d3 in det3d_dicts:
        for fid, rec in d3.items():
            dets = rec.get('detections', []) or []
            if not dets:
                continue
            per_frame_all.setdefault(fid, []).extend(dets)

    # 2) Filtering (low-quality box removal for ships only)
    per_frame_filtered: Dict[str, List[Dict[str,Any]]] = {}
    for fid, dets in per_frame_all.items():
        keeps = []
        for d in dets:
            if not isinstance(d, dict): 
                continue
            label = d.get('label', '')
            b = d.get('box7d')
            if not (isinstance(b, (list,tuple)) and len(b) == 7):
                continue
            x,y,_,dx,dy,_,yaw = [float(t) for t in b]
            poly = box2d_corners_xy(x,y,dx,dy,yaw)

            # 🚩 apply low-quality filtering only to ships
            if chamber_y_range is not None and (label in ship_set):
                fin = fraction_inside_chamber(poly, chamber_x_range, chamber_y_range)
                if (1.0 - fin) > outside_frac_thresh:
                    continue

            keeps.append(d)
        if keeps:
            per_frame_filtered[fid] = keeps

    # 3) Deduplicate (same object merge: keep highest score)
    ships_3d: Dict[str, List[Dict[str,Any]]] = {}
    gate_present_by_3d: Dict[str, bool] = {}

    for fid, dets in per_frame_filtered.items():
        ships = [d for d in dets if d.get('label') in ship_set]
        gates = [d for d in dets if d.get('label') == lock_gate_label]

        def nms_rotated(dets_list: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
            if not dets_list:
                return []
            enriched = []
            for d in dets_list:
                b = d['box7d']
                x,y,_,dx,dy,_,yaw = [float(t) for t in b]
                poly = box2d_corners_xy(x,y,dx,dy,yaw).astype(np.float32)
                enriched.append((d, poly, float(d.get('score', 0.0))))
            enriched.sort(key=lambda x: x[2], reverse=True)

            kept = []
            while enriched:
                best = enriched.pop(0)
                kept.append(best)
                rest = []
                for item in enriched:
                    iou = poly_iou(best[1], item[1])
                    if iou <= iou_thresh:
                        rest.append(item)
                enriched = rest
            return [k[0] for k in kept]

        ships_nms = nms_rotated(ships)
        gates_nms = nms_rotated(gates)

        if ships_nms:
            ships_3d[fid] = ships_nms
        gate_present_by_3d[fid] = len(gates_nms) > 0

    return ships_3d, gate_present_by_3d



# ----------------- Main pipeline -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='Path to configs/config.yaml')
    ap.add_argument('--det2d_dir', default='outputs/det2d', help='dir containing *.det2d.json')
    ap.add_argument('--det3d_dir', default='outputs/det3d', help='dir containing *.det3d.json')
    ap.add_argument('--out_dir',  default='outputs/alerts', help='output dir for alerts')
    ap.add_argument('--outside_frac', type=float, default=0.10, help='Outside-area fraction threshold (> value → drop); default 0.10')
    ap.add_argument('--nms_iou', type=float, default=0.30, help='Rotated IoU NMS threshold; default 0.30')
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    labels = cfg['labels']
    lock_gate_label = labels['lock_gate']
    ship_labels = list(labels['ships'])

    geom = cfg.get('geometry', {}) or {}
    chamber_x_range = tuple(geom.get('chamber_x_range', [39.7, 62.7]))
    stop_line_y = float(geom.get('stop_line_y', 23.002))
    no_stop_y_range = tuple(geom.get('no_stop_y_range', [23.002, 43.002]))

    # New: chamber y-range (for outside-fraction filtering)
    chamber_y_range = tuple(geom.get('chamber_y_range')) if geom.get('chamber_y_range', None) is not None else None
    if chamber_y_range is None:
        print("[WARN] geometry.chamber_y_range not set; will skip 'outside-fraction' low-quality filtering.")

    ensure_dir(args.out_dir)

    # 1) Load 2D/3D results
    det2d_all = load_jsons_in_dir(args.det2d_dir, '.det2d.json')
    det3d_all = load_jsons_in_dir(args.det3d_dir, '.det3d.json')
    if not det3d_all:
        raise RuntimeError(f"No 3D json in {args.det3d_dir}")
    if not det2d_all:
        print(f"[WARN] No 2D json in {args.det2d_dir}; will rely on filtered 3D gate boxes to infer presence (more conservative).")

    # 2) Apply "low-quality filtering + duplicate removal" to 3D detections
    ships_3d, gate_present_3d = process_3d_detections(
        det3d_all,
        ship_labels=ship_labels,
        lock_gate_label=lock_gate_label,
        chamber_x_range=chamber_x_range,
        chamber_y_range=chamber_y_range,   # can be None (skip low-quality filtering)
        outside_frac_thresh=float(args.outside_frac),
        iou_thresh=float(args.nms_iou)
    )

    # 3) Fuse gate presence (prefer 2D; if 2D missing, use "filtered 3D")
    gate_present = merge_gate_presence(det2d_all, [], lock_gate_label)  # 2D only first
    for fid, v in gate_present_3d.items():
        if fid not in gate_present:
            gate_present[fid] = bool(v)
        else:
            gate_present[fid] = gate_present[fid] or bool(v)
    save_json(gate_present, os.path.join(args.out_dir, 'gate_presence.json'))

    # 4) Check no-stop line/band
    rows = []
    alerts_by_frame: Dict[str, List[Dict[str,Any]]] = {}

    for fid, dets in ships_3d.items():
        if not gate_present.get(fid, False):
            # No gate present → skip checks
            continue

        cur_alerts = []
        for idx, d in enumerate(dets):
            b = d.get('box7d')
            if not (isinstance(b, (list, tuple)) and len(b) == 7):
                continue
            x,y,z,dx,dy,dz,yaw = [float(t) for t in b]
            poly = box2d_corners_xy(x,y,dx,dy,yaw)

            cross = poly_intersect_stop_line(poly, stop_line_y, chamber_x_range)
            overlap = poly_overlaps_band(poly, chamber_x_range, no_stop_y_range)

            if cross or overlap:
                rec = {
                    'frame_id': fid,
                    'idx': idx,
                    'label': d.get('label', ''),
                    'score': float(d.get('score', 0.0)),
                    'box7d': [float(t) for t in b],
                    'violation': 'line_cross' if cross else 'zone_overlap'
                }
                rows.append(rec)
                cur_alerts.append(rec)

        if cur_alerts:
            alerts_by_frame[fid] = cur_alerts

    # 5) Outputs
    save_json(alerts_by_frame, os.path.join(args.out_dir, 'alerts_by_frame.json'))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out_dir, 'alerts_summary.csv'), index=False, encoding='utf-8-sig')
        print(f"[ALERTS] violations: {len(rows)} -> {os.path.join(args.out_dir, 'alerts_summary.csv')}")
    else:
        print("[ALERTS] No violations.")

if __name__ == '__main__':
    main()

