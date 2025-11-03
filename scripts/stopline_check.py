# -*- coding: utf-8 -*-
"""
stopline_check.py — No-stop line/band check after fusing 2D & 3D results
(includes 3D low-quality box filtering + duplicate removal + gate FSM presence)

Inputs:
- outputs/det2d/*.det2d.json (from infer_2d_mmdet.py)  [kept for compatibility, not used for gate presence]
- outputs/det3d/*.det3d.json (from infer_3d_pcdet.py; detections only)
- add_info_testing_by_timestamp.txt (first two columns: frame_id, timestamp; optional 4th column: segment id)

New gate presence logic (your rules):
- Two-state FSM: PRESENT / ABSENT (no TRANSITION state)
- Birth (ABSENT->PRESENT):
    * If it's the first frame of a segment: ANY detected gate width (any dy) immediately PRESENT.
    * Otherwise: first detected frame must have w <= w_small (default 1.0 m) to PRESENT immediately.
- Disappear (PRESENT->ABSENT):
    * While PRESENT, accumulate dwell time for w <= w_small; once >= small_dwell (default 2.0s),
      the NEXT missing frame (no gate detected) immediately flips to ABSENT.
    * Missing before dwell is reached does NOT cause disappearance.
- Global very first frame: if 'skip_first_frame_judgment' is on, do not flip state (treated as ABSENT).

Ship detections:
- Low-quality filtering (outside-area fraction) for ships only
- Rotated quadrilateral IoU NMS for deduplication

Violation criteria (only when gate_present == True):
- line_cross: ship polygon intersects the segment y = stop_line_y (x in chamber_x_range)
- zone_overlap: ship polygon overlaps the no-stop band (x in chamber_x_range, y in no_stop_y_range)

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
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from utils.io_utils import ensure_dir, save_json, list_files_multi


def load_cfg(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def frame_id_from_path(p: str, rgx: str) -> str:
    """
    Extract frame_id from a file path using a regex on the basename.
    Fallback to stem if regex does not match.
    """
    m = re.match(rgx, Path(p).name)
    if m:
        return m.group(1)
    return Path(p).stem


# ----------------- Geometry utilities -----------------
def rot2d(cos_t: float, sin_t: float, x: float, y: float) -> Tuple[float, float]:
    """Rotate (x,y) by a given cosine/sine of the angle."""
    return cos_t * x - sin_t * y, sin_t * x + cos_t * y

def box2d_corners_xy(x: float, y: float, dx: float, dy: float, yaw: float) -> np.ndarray:
    """
    Generate XY corners (clockwise order) from a 2D box parameterized as (x,y,dx,dy,yaw).
    Returns:
        np.ndarray of shape (4, 2).
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
    """
    Return True if line segments p1-p2 and q1-q2 intersect.
    Uses cross product tests; handles collinear overlaps.
    """
    def cross(a, b): return a[0]*b[1] - a[1]*b[0]
    def sub(a, b): return (a[0]-b[0], a[1]-b[1])
    r, s = sub(p2, p1), sub(q2, q1)
    rxs = cross(r, s)
    q_p = sub(q1, p1)
    qpxr = cross(q_p, r)
    if abs(rxs) < 1e-9 and abs(qpxr) < 1e-9:
        # Collinear: check overlap using projection parameters
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
    Check whether polygon 'poly' (shape (4,2)) intersects the horizontal segment y=stop_y,
    where x is restricted to x_range.
    """
    x1, x2 = float(min(x_range)), float(max(x_range))
    L1 = (x1, stop_y); L2 = (x2, stop_y)
    # Test each polygon edge against the line segment
    for i in range(len(poly)):
        P1 = (float(poly[i][0]), float(poly[i][1]))
        P2 = (float(poly[(i+1) % len(poly)][0]), float(poly[(i+1) % len(poly)][1]))
        if segment_intersect(P1, P2, L1, L2):
            return True
    return False

def poly_overlaps_band(poly: np.ndarray, x_range: Tuple[float,float], y_range: Tuple[float,float]) -> bool:
    """
    Coarse but robust test whether polygon overlaps the rectangle band
    [x_min,x_max]×[y_min,y_max]:
      1) AABB early-out
      2) Corner-inclusion tests (both ways)
      3) Edge intersection tests
    """
    x_min, x_max = float(min(x_range)), float(max(x_range))
    y_min, y_max = float(min(y_range)), float(max(y_range))

    # Early reject: AABBs do not overlap
    poly_xmin, poly_ymin = float(np.min(poly[:,0])), float(np.min(poly[:,1]))
    poly_xmax, poly_ymax = float(np.max(poly[:,0])), float(np.max(poly[:,1]))
    if (poly_xmax < x_min) or (poly_xmin > x_max) or (poly_ymax < y_min) or (poly_ymin > y_max):
        return False

    # Rectangle corners
    rect_corners = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.float32)

    def point_in_poly(pt, poly_):
        """Ray casting test for point-in-polygon."""
        x, y = pt
        inside = False
        n = len(poly_)
        for i in range(n):
            x1,y1 = poly_[i]
            x2,y2 = poly_[(i+1)%n]
            inter = ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12)+x1)
            if inter: inside = not inside
        return inside

    # Any rect corner inside polygon OR any polygon corner inside rect → overlap
    if any(point_in_poly(tuple(rc), poly) for rc in rect_corners):
        return True
    if any((x_min <= px <= x_max) and (y_min <= py <= y_max) for px,py in poly):
        return True

    # Edge intersection (4 rect edges vs polygon edges)
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
    """Return signed polygon area (CW/CCW works). poly shape=(N,2)."""
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
        # Keep left side of edge a->b
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
    """Compute IoU between two convex polygons via Sutherland–Hodgman clipping."""
    inter = suth_hodg_clip(polyA, polyB)
    if inter.shape[0] == 0:
        # Try swapped roles for numerical robustness
        inter = suth_hodg_clip(polyB, polyA)
    if inter.shape[0] == 0:
        return 0.0
    aA = abs(polygon_area(polyA))
    aB = abs(polygon_area(polyB))
    aI = abs(polygon_area(inter))
    denom = aA + aB - aI + 1e-12
    return float(aI / denom)

def fraction_inside_chamber(poly: np.ndarray, x_range: Tuple[float,float], y_range: Tuple[float,float]) -> float:
    """
    Compute the fraction (0–1) of polygon area inside the chamber rectangle.
    Returns 0 for degenerate polygons (area ~ 0).
    """
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


# ----------------- I/O helpers -----------------
def load_jsons_in_dir(d: str, suffix: str) -> List[Dict[str,Any]]:
    """Load all JSON files under directory 'd' whose names end with 'suffix'."""
    files = sorted([str(p) for p in Path(d).glob(f'*{suffix}')])
    out = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as h:
                out.append(json.load(h))
        except Exception as e:
            print(f"[WARN] failed to load {f}: {e}")
    return out


# ----------------- Gate width extraction & FSM presence -----------------
def parse_timestamp_str(ts: str) -> datetime:
    """
    Parse 'YYYY_MM_DD_HH_MM_SS_micro' to datetime.
    Example: 2024_07_17_17_47_42_196221
    Non-digit chars are normalized to underscores before parsing.
    """
    s = ''.join(ch if ch.isdigit() else '_' for ch in ts)
    return datetime.strptime(s, '%Y_%m_%d_%H_%M_%S_%f')

def load_frame_timestamps(ts_file: str) -> Tuple[Dict[str, datetime], Dict[str, str]]:
    """
    Read mappings from a timestamp file:
      - frame_id -> datetime (first two whitespace-separated columns)
      - frame_id -> segment_id (optional: 4th column if present)

    Returns:
      ts_map, seg_map
    """
    ts_map: Dict[str, datetime] = {}
    seg_map: Dict[str, str] = {}
    with open(ts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            fid, ts = parts[0], parts[1]
            try:
                ts_map[fid] = parse_timestamp_str(ts)
            except Exception:
                continue
            # Optional: 4th column is the segment id
            if len(parts) >= 4:
                seg_map[fid] = parts[3]
    return ts_map, seg_map

def extract_gate_best_width(det3d_dicts: List[Dict[str,Any]],
                            lock_gate_label: str,
                            iou_thresh: float = 0.30) -> Dict[str, float]:
    """
    For each frame, take 3D detections with label==lock_gate_label, apply rotated NMS,
    then keep the highest-scoring gate box; return its dy as width (meters).

    Returns:
        gate_w[fid] = dy   (no entry means 'missing/no gate')
    """
    # Gather gate detections per frame
    per_frame_gates: Dict[str, List[Dict[str,Any]]] = {}
    for d3 in det3d_dicts:
        for fid, rec in d3.items():
            dets = rec.get('detections', []) or []
            gs = [d for d in dets if isinstance(d, dict) and d.get('label') == lock_gate_label and isinstance(d.get('box7d'), (list,tuple)) and len(d.get('box7d'))==7]
            if gs:
                per_frame_gates.setdefault(fid, []).extend(gs)

    # Rotated NMS for gates using polygon IoU
    def nms_rotated(dets_list: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        if not dets_list:
            return []
        enriched = []
        for d in dets_list:
            x,y,_,dx,dy,_,yaw = [float(t) for t in d['box7d']]
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

    gate_w: Dict[str, float] = {}
    for fid, gs in per_frame_gates.items():
        nms = nms_rotated(gs)
        if not nms:
            continue
        # Pick highest score after NMS and take dy (width)
        top = max(nms, key=lambda d: float(d.get('score', 0.0)))
        dy = float(top['box7d'][4])  # box7d = [x,y,z,dx,dy,dz,yaw]
        gate_w[fid] = dy
    return gate_w

def compute_gate_presence_fsm_2state(
    gate_width: Dict[str, float],
    ts_map: Dict[str, datetime],
    seg_map: Dict[str, str] = None,
    w_small: float = 1.0,
    small_dwell: float = 2.0,
    skip_first_frame_judgment: bool = True,
    relax_birth_on_segment_first: bool = True,
    raw_present_passthrough: Optional[Dict[str, bool]] = None,
    skip_segment_first_use_raw: bool = False
) -> Dict[str, bool]:
    """
    Two-state FSM for gate presence (PRESENT/ABSENT) with raw passthrough options.

    Behavior:
      A) Global first frame passthrough (if skip_first_frame_judgment=True):
         - Do NOT run FSM.
         - Use raw_present_passthrough[fid] if provided; otherwise fallback to 3D existence (fid in gate_width).
         - Seed the internal state with this boolean to ensure temporal continuity.

      B) Segment-first passthrough (if skip_segment_first_use_raw=True):
         - For the first frame of a segment, same logic as A) but only at segment starts.

      C) Standard FSM (other frames):
         - Birth (ABSENT→PRESENT):
             * If relax_birth_on_segment_first and current frame is the first of its segment:
               any detected gate (any width) gives PRESENT immediately.
             * Otherwise, require 'has_gate' AND width <= w_small to set PRESENT.
         - Disappear (PRESENT→ABSENT):
             * While PRESENT, accumulate a "small width dwell clock" when width <= w_small.
             * Upon the NEXT missing frame with small_clock >= small_dwell, flip to ABSENT.
             * Missing frames before dwell is reached do NOT cause disappearance.

    Args:
        gate_width: {frame_id -> width_meters} for frames where a gate was detected after NMS.
        ts_map:     {frame_id -> datetime} for temporal ordering.
        seg_map:    {frame_id -> segment_id} (optional, used for segment-first logic).
        w_small:    Threshold (meters) for "small width".
        small_dwell:Seconds of accumulated "small width" time before a missing frame can flip to ABSENT.
        skip_first_frame_judgment: If True, use passthrough on the very first frame in the entire sequence.
        relax_birth_on_segment_first: If True, allow birth on segment-first frames regardless of width.
        raw_present_passthrough: Optional {frame_id -> bool} prior/raw presence map to use in passthrough.
        skip_segment_first_use_raw: If True, also apply passthrough on the first frame of each segment.

    Returns:
        presence: {frame_id -> bool} indicating PRESENT (True) or ABSENT (False).
    """
    frames = [fid for fid in ts_map.keys()]
    frames.sort(key=lambda k: ts_map[k])

    # Compute the first frame of each segment for segment-first logic
    first_fid_of_seg: Dict[str, str] = {}
    if seg_map:
        seg_to_first: Dict[str, Tuple[str, datetime]] = {}
        for fid in frames:
            seg_id = seg_map.get(fid, None)
            if seg_id is None: 
                continue
            t = ts_map[fid]
            if seg_id not in seg_to_first or t < seg_to_first[seg_id][1]:
                seg_to_first[seg_id] = (fid, t)
        first_fid_of_seg = {seg: pair[0] for seg, pair in seg_to_first.items()}

    PRESENT, ABSENT = True, False
    state = ABSENT
    small_clock = 0.0
    last_ts = None
    out: Dict[str, bool] = {}

    for idx, fid in enumerate(frames):
        cur_ts = ts_map[fid]
        dt = 0.0 if last_ts is None else max(0.0, (cur_ts - last_ts).total_seconds())
        last_ts = cur_ts

        has_gate = fid in gate_width
        w = gate_width.get(fid, None)

        # Is this the first frame of its segment?
        is_segment_first = False
        if seg_map and fid in seg_map and first_fid_of_seg:
            seg_id = seg_map.get(fid, None)
            if seg_id is not None and first_fid_of_seg.get(seg_id, None) == fid:
                is_segment_first = True

        # ---------- A) Global first frame passthrough ----------
        if idx == 0 and skip_first_frame_judgment:
            raw_val = None
            if raw_present_passthrough is not None:
                raw_val = raw_present_passthrough.get(fid, None)
            if raw_val is None:
                raw_val = bool(has_gate)  # fallback: 3D existence
            out[fid] = bool(raw_val)
            # Seed internal state for continuity
            state = PRESENT if out[fid] else ABSENT
            small_clock = 0.0
            continue

        # ---------- B) Segment-first passthrough (optional) ----------
        if skip_segment_first_use_raw and is_segment_first:
            raw_val = None
            if raw_present_passthrough is not None:
                raw_val = raw_present_passthrough.get(fid, None)
            if raw_val is None:
                raw_val = bool(has_gate)
            out[fid] = bool(raw_val)
            state = PRESENT if out[fid] else ABSENT
            small_clock = 0.0
            continue

        # ---------- C) Standard FSM ----------
        if state == ABSENT:
            if relax_birth_on_segment_first and is_segment_first:
                if has_gate:
                    state = PRESENT
                    small_clock = 0.0
            else:
                if has_gate and (w is not None) and (w <= w_small):
                    state = PRESENT
                    small_clock = 0.0
        else:  # PRESENT
            if has_gate and (w is not None):
                if w <= w_small:
                    small_clock += dt
                else:
                    small_clock = 0.0
            else:
                if small_clock >= small_dwell:
                    state = ABSENT
                    small_clock = 0.0

        out[fid] = (state == PRESENT)

    return out


# ----------------- 2D/3D fusion (legacy helper: kept but unused for presence) -----------------
def merge_gate_presence(det2d_dicts: List[Dict[str,Any]],
                        det3d_dicts: List[Dict[str,Any]],
                        lock_gate_label: str) -> Dict[str, bool]:
    """
    Legacy helper (kept for compatibility). Not used in final presence decision.
    It scans 2D/3D results and ORs presence flags if a 'lock_gate_present' field exists,
    or infers presence from the existence of detections with the lock_gate_label.
    """
    pres: Dict[str,bool] = {}

    def scan_and_update(d, from_which):
        for fid, rec in d.items():
            lgp = rec.get('lock_gate_present', None)
            if isinstance(lgp, bool):
                pres[fid] = bool(lgp)
                continue
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


# ----------------- 3D ships processing (filter + rotated NMS) -----------------
def process_3d_detections(det3d_dicts: List[Dict[str,Any]],
                          ship_labels: List[str],
                          lock_gate_label: str,
                          chamber_x_range: Tuple[float,float],
                          chamber_y_range: Tuple[float,float] = None,
                          outside_frac_thresh: float = 0.10,
                          iou_thresh: float = 0.30) -> Tuple[Dict[str, List[Dict[str,Any]]], Dict[str, bool]]:
    """
    Post-process raw 3D detections:
      1) Merge all frames.
      2) Filter out low-quality ship boxes by "outside-area fraction", if chamber_y_range is provided.
         (Note: this filter is applied to SHIP boxes only; NEVER to gate boxes.)
      3) Remove duplicates by rotated polygon IoU NMS (keep highest score).

    Args:
        det3d_dicts: List of per-file dicts mapping frame_id -> record with 'detections'.
        ship_labels: List of labels considered as ships.
        lock_gate_label: Label used for the lock gate.
        chamber_x_range: (xmin, xmax) in meters.
        chamber_y_range: Optional (ymin, ymax) in meters; if None, step (2) is skipped.
        outside_frac_thresh: If (1 - fraction_inside) > threshold, drop ship box.
        iou_thresh: IoU threshold for rotated NMS.

    Returns:
        ships_3d: {frame_id -> [filtered & deduplicated ship detections]}
        gate_present_by_3d: {frame_id -> bool} legacy presence by 3D after NMS (not used later)
    """
    ship_set = set(ship_labels)
    per_frame_all: Dict[str, List[Dict[str,Any]]] = {}

    # 1) Merge all 3D files into a per-frame list
    for d3 in det3d_dicts:
        for fid, rec in d3.items():
            dets = rec.get('detections', []) or []
            if not dets:
                continue
            per_frame_all.setdefault(fid, []).extend(dets)

    # 2) Low-quality filtering for SHIP boxes only (by inside fraction of chamber)
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

            # Apply low-quality filtering ONLY to ships
            if chamber_y_range is not None and (label in ship_set):
                fin = fraction_inside_chamber(poly, chamber_x_range, chamber_y_range)
                if (1.0 - fin) > outside_frac_thresh:
                    continue

            keeps.append(d)
        if keeps:
            per_frame_filtered[fid] = keeps

    # 3) Rotate-NMS per frame for ships and gates separately
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
    ap.add_argument('--det2d_dir', default='outputs/det2d', help='Dir containing *.det2d.json')
    ap.add_argument('--det3d_dir', default='outputs/det3d', help='Dir containing *.det3d.json')
    ap.add_argument('--out_dir',  default='outputs/alerts', help='Output directory for alert artifacts')
    ap.add_argument('--outside_frac', type=float, default=0.10, help='Outside-area fraction threshold; if (1 - inside_fraction) > this value → drop (ships only). Default 0.10')
    ap.add_argument('--nms_iou', type=float, default=0.30, help='Rotated IoU NMS threshold. Default 0.30')

    # Time-series & FSM parameters
    ap.add_argument('--ts_file', required=False, help='Timestamp file (can also be set in config.data.ts_file)')
    ap.add_argument('--w_small', type=float, default=1.0, help='Width threshold (m) for "small" gate width. Default 1.0')
    ap.add_argument('--small_dwell', type=float, default=2.0, help='Dwell time (s) that w<=w_small must hold before the first following MISSING can flip to ABSENT. Default 2.0')
    ap.add_argument('--skip_first_frame_judgment', action='store_true', default=True,
                    help='If set, the very first frame in time uses passthrough and does not run FSM.')
    ap.add_argument('--no_skip_first_frame_judgment', dest='skip_first_frame_judgment',
                    action='store_false')
    ap.add_argument('--relax_birth_on_segment_first', action='store_true', default=True,
                    help='If set, the first frame of a segment allows birth with any gate width.')
    ap.add_argument('--no_relax_birth_on_segment_first', dest='relax_birth_on_segment_first',
                    action='store_false')

    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    labels = cfg['labels']
    lock_gate_label = labels['lock_gate']
    ship_labels = list(labels['ships'])

    geom = cfg.get('geometry', {}) or {}
    chamber_x_range = tuple(geom.get('chamber_x_range', [39.7, 62.7]))
    stop_line_y = float(geom.get('stop_line_y', 23.002))
    no_stop_y_range = tuple(geom.get('no_stop_y_range', [23.002, 43.002]))

    # Chamber y-range (used for outside-fraction filtering on ships)
    chamber_y_range = tuple(geom.get('chamber_y_range')) if geom.get('chamber_y_range', None) is not None else None
    if chamber_y_range is None:
        print("[WARN] geometry.chamber_y_range not set; will skip 'outside-fraction' low-quality filtering for ships.")

    ensure_dir(args.out_dir)

    # 1) Load 2D/3D results (2D kept for compatibility, not used for gate presence)
    det2d_all = load_jsons_in_dir(args.det2d_dir, '.det2d.json')
    det3d_all = load_jsons_in_dir(args.det3d_dir, '.det3d.json')
    if not det3d_all:
        raise RuntimeError(f"No 3D json in {args.det3d_dir}")
    if not det2d_all:
        print(f"[INFO] No 2D json in {args.det2d_dir}; proceeding with 3D-only gate width FSM presence.")

    # 2) Ships processing (filter + rotated NMS)
    ships_3d, _gate_present_3d_legacy = process_3d_detections(
        det3d_all,
        ship_labels=ship_labels,
        lock_gate_label=lock_gate_label,
        chamber_x_range=chamber_x_range,
        chamber_y_range=chamber_y_range,   # can be None (skip low-quality filtering)
        outside_frac_thresh=float(args.outside_frac),
        iou_thresh=float(args.nms_iou)
    )

    # 3) Gate presence from 3D gate width with two-state FSM (your rule set)
    ts_file_path = args.ts_file if args.ts_file else cfg.get('data', {}).get('ts_file', None)
    if not ts_file_path or not os.path.exists(ts_file_path):
       raise RuntimeError(f"Timestamp file not found. Please set --ts_file or data.ts_file in config.yaml. Got: {ts_file_path}")

    ts_map, seg_map = load_frame_timestamps(ts_file_path)

    # Compute per-frame gate width (meters) using best NMSed gate box
    gate_w = extract_gate_best_width(det3d_all, lock_gate_label, iou_thresh=float(args.nms_iou))

    # Raw passthrough prior: if a frame has a measured width (3D gate exists), mark True
    raw_present = {fid: True for fid in gate_w.keys()}
    # Also honor any 2D records that explicitly claim 'lock_gate_present'
    for d2 in det2d_all:
        for fid, rec in d2.items():
            if bool(rec.get("lock_gate_present", False)):
                raw_present[fid] = True

    gate_present = compute_gate_presence_fsm_2state(
        gate_width=gate_w,
        ts_map=ts_map,
        seg_map=seg_map,  # may be empty if your ts file has no 4th column
        w_small=float(args.w_small),
        small_dwell=float(args.small_dwell),
        skip_first_frame_judgment=bool(args.skip_first_frame_judgment),
        relax_birth_on_segment_first=bool(args.relax_birth_on_segment_first), 
        raw_present_passthrough=raw_present,             # passthrough for first/global or segment-first
        skip_segment_first_use_raw=True                  # apply passthrough at segment starts
    )
    save_json(gate_present, os.path.join(args.out_dir, 'gate_presence.json'))

    # 4) Check no-stop line/band (only when gate_present == True)
    rows = []
    alerts_by_frame: Dict[str, List[Dict[str,Any]]] = {}

    for fid in sorted(ts_map.keys(), key=lambda k: ts_map[k]):
        dets = ships_3d.get(fid, [])
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
