# K-CVD-Hybrid-Splat-Reconstruction
 (Real-Time 3D Reconstruction on Mobile) For Android - Pydroid 3 
# K-CVD HYBRID ACCUM SPLAT V3.1
# ------------------------------------------------------
# Added / upgraded:
# - CAMERA CALIBRATION + UNDISTORTION
# - MOTION QUALITY GATE (rejects weak / rotation-heavy captures)
# - FORWARD/BACKWARD LK FILTERING
# - MULTI-KEYFRAME FUSION
# - REPROJECTION-ERROR PRUNING
# - LIGHT PLANE SNAP CLEANUP FOR INDOOR SCANS
# - EDGE-GUIDED DENSIFICATION
# - ANDROID-FRIENDLY UI LAYOUT (buttons separated into touchable rows)
#
# NEW IN THIS VERSION:
# - REAL CALIBRATED INTRINSICS USED FOR POSE SOLVE WHEN AVAILABLE
# - TRIANGULATION-ANGLE FILTERING + SCORING
# - WEIGHTED VOXEL MERGE FOR POSITION/COLOR ACCUMULATION
#
# Saves accumulated model to:
# /storage/emulated/0/Python Projects/scans/last_splat_cloud.json
#
# Saves camera calibration to:
# /storage/emulated/0/Python Projects/scans/camera_calibration.json
#
# HOW TO CALIBRATE:
# 1) Print / show a 9x6 inner-corner chessboard pattern.
# 2) Point camera at it from multiple angles/distances.
# 3) Press "CAP CAL" several times (10-20 good captures).
# 4) Press "SOLVE CAL".
# 5) Toggle "UNDIST: ON".
#
# Notes:
# - Calibration uses 9x6 INNER corners.
# - Works without external services or cloud.
# - Plane snap is intentionally light so it does not over-flatten detail.

import os
import time
import math
import json

import cv2
import numpy as np

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy.graphics.texture import Texture

try:
    from android.permissions import request_permissions, Permission
    ANDROID_PERMS_AVAILABLE = True
except Exception:
    ANDROID_PERMS_AVAILABLE = False


# ================== CONFIG ==================

BASE_DIR = "/storage/emulated/0/Python Projects"
SCANS_DIR = os.path.join(BASE_DIR, "scans")
LAST_CLOUD_PATH = os.path.join(SCANS_DIR, "last_splat_cloud.json")
CALIB_PATH = os.path.join(SCANS_DIR, "camera_calibration.json")

CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS_TARGET = 24.0

MAX_FEATURE_POINTS_DEFAULT = 1000
MAX_EDGE_POINTS_DEFAULT = 900
CALIB_FRAMES = 20
CALIB_MIN_POINTS = 40

# rolling keyframe positions during a scan burst
KEYFRAME_STEPS = (5, 10, 15, 20)

SCREEN_W = 640
SCREEN_H = 480
FOCAL_LEN = 1.25

DEFAULT_SPLAT_RADIUS = 3.0
SPLAT_MAX_RADIUS = 12

CLOUD_RADIUS_CLAMP = 3.0

MIN_MOTION_PX = 1.15
ESSENTIAL_RANSAC_THRESH = 1.0

MIN_TRACKED_BEFORE_RESEED = 140
RESEED_COOLDOWN_FRAMES = 3
LK_WIN = (21, 21)
LK_LEVELS = 2

FB_ERR_THRESH = 1.5
LK_ERR_THRESH = 24.0

CAM_TRIES = 8
CAM_RETRY_DELAY = 0.25
CAM_INDEX_TRY_LIST = (0, 1, 2, 3, 4, 5, 6)

HUD_TINT_ALPHA = 0.35
HUD_COLOR = (0, 0, 255)  # red BGR

LIVE_OVERLAY_ENABLED_DEFAULT = True
LIVE_OVERLAY_RADIUS = 4
LIVE_OVERLAY_ALPHA = 0.42
COLOR_SAMPLE_PATCH = 3
LIVE_GHOST_MAX = 2400

VOXEL_SIZE = 0.020
ACC_MAX_POINTS = 18000

DEFAULT_SHAPE_MODE = "TRI"

EDGE_CANNY_T1 = 70
EDGE_CANNY_T2 = 155
EDGE_GRID_STRIDE = 6
EDGE_GRAD_MIN = 36.0

CONF_MIN_KEEP = 0.22
MERGE_COLOR_CLAMP = True

# Motion quality gate
MOTION_GATE_MIN_MEAN_PX = 1.7
MOTION_GATE_MIN_PARALLAX_RESID = 0.55
MOTION_GATE_MIN_GOOD_RATIO = 0.33

# Reprojection pruning
REPROJ_ERR_THRESH = 2.75

# Camera calibration
CHESSBOARD_COLS = 9   # inner corners
CHESSBOARD_ROWS = 6   # inner corners
CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

# Plane snapping
PLANE_SNAP_ENABLED_DEFAULT = True
PLANE_RANSAC_ITERS = 60
PLANE_DIST_THRESH = 0.028
PLANE_MAX_COUNT = 3
PLANE_MIN_INLIERS = 120
PLANE_SNAP_BLEND = 0.45

# NEW: triangulation-angle filtering / scoring
TRI_ANGLE_MIN_DEG = 0.75
TRI_ANGLE_SOFT_MAX_DEG = 6.0


# ================== FILE HELPERS ==================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def ensure_scans_dir():
    ensure_dir(SCANS_DIR)

def save_cloud(points_3d, colors_bgr, weights=None, meta=None):
    ensure_scans_dir()
    payload = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "points": points_3d.tolist() if isinstance(points_3d, np.ndarray) else [],
        "colors_bgr": colors_bgr.tolist() if isinstance(colors_bgr, np.ndarray) else [],
        "weights": weights.tolist() if isinstance(weights, np.ndarray) else [],
        "meta": meta or {}
    }
    with open(LAST_CLOUD_PATH, "w") as f:
        json.dump(payload, f, indent=2)

def load_cloud():
    if not os.path.isfile(LAST_CLOUD_PATH):
        return None, None, None, None
    try:
        with open(LAST_CLOUD_PATH, "r") as f:
            d = json.load(f)
        pts = np.array(d.get("points", []), dtype=np.float32)
        cols = np.array(d.get("colors_bgr", []), dtype=np.float32)
        wts = np.array(d.get("weights", []), dtype=np.float32)
        meta = d.get("meta", {})
        if pts.size == 0 or cols.size == 0:
            return None, None, None, meta

        n = min(len(pts), len(cols))
        pts = pts[:n]
        cols = cols[:n]

        if wts.size == 0:
            wts = np.ones((n,), dtype=np.float32)
        else:
            wts = wts.reshape(-1).astype(np.float32)
            wts = wts[:n]
            if len(wts) < n:
                pad = np.ones((n - len(wts),), dtype=np.float32)
                wts = np.concatenate([wts, pad], axis=0)

        return pts, cols, wts, meta
    except Exception as e:
        print("Failed to load cloud:", e)
        return None, None, None, None

def save_camera_calibration(camera_matrix, dist_coeffs, image_size):
    ensure_scans_dir()
    payload = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": list(image_size),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(CALIB_PATH, "w") as f:
        json.dump(payload, f, indent=2)

def load_camera_calibration():
    if not os.path.isfile(CALIB_PATH):
        return None, None, None
    try:
        with open(CALIB_PATH, "r") as f:
            d = json.load(f)
        K = np.array(d.get("camera_matrix", []), dtype=np.float64)
        D = np.array(d.get("dist_coeffs", []), dtype=np.float64)
        sz = d.get("image_size", None)
        if K.shape != (3, 3):
            return None, None, None
        if D.ndim == 1:
            D = D.reshape(1, -1)
        return K, D, tuple(sz) if sz is not None else None
    except Exception as e:
        print("Failed to load calibration:", e)
        return None, None, None


# ================== IMAGE HELPERS ==================

def _good_features(gray_blur, max_pts=900):
    pts = cv2.goodFeaturesToTrack(
        gray_blur,
        maxCorners=int(max_pts),
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        useHarrisDetector=False
    )
    if pts is None:
        return None
    return pts.astype(np.float32)

def _sample_colors_bgr_patch(img_bgr, pts_xy, patch_r=3):
    if img_bgr is None or pts_xy is None or len(pts_xy) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    h, w = img_bgr.shape[:2]
    pr = int(max(0, patch_r))
    cols = np.zeros((len(pts_xy), 3), dtype=np.float32)

    for i, (x, y) in enumerate(pts_xy):
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        x0 = max(0, xi - pr)
        x1 = min(w - 1, xi + pr)
        y0 = max(0, yi - pr)
        y1 = min(h - 1, yi + pr)
        patch = img_bgr[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        if patch.size == 0:
            cols[i] = img_bgr[yi, xi].astype(np.float32)
        else:
            cols[i] = patch.reshape(-1, 3).mean(axis=0)
    return cols

def _gradient_mag(gray_blur):
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def _sample_gradient_at_points(grad_mag, pts_xy):
    if grad_mag is None or pts_xy is None or len(pts_xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = grad_mag.shape[:2]
    out = np.zeros((len(pts_xy),), dtype=np.float32)
    for i, (x, y) in enumerate(pts_xy):
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        out[i] = float(grad_mag[yi, xi])
    return out

def _sample_edge_points(gray_blur, max_pts=800, grid_stride=6):
    edges = cv2.Canny(gray_blur, int(EDGE_CANNY_T1), int(EDGE_CANNY_T2))
    grad = _gradient_mag(gray_blur)

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return None

    pts = []
    taken = set()

    for x, y in zip(xs.tolist(), ys.tolist()):
        gx = int(x // max(1, grid_stride))
        gy = int(y // max(1, grid_stride))
        k = (gx, gy)
        if k in taken:
            continue
        if float(grad[y, x]) < float(EDGE_GRAD_MIN):
            continue
        taken.add(k)
        pts.append((float(x), float(y)))

    if len(pts) == 0:
        return None

    pts = np.array(pts, dtype=np.float32)
    if len(pts) > int(max_pts):
        idx = np.random.choice(len(pts), int(max_pts), replace=False)
        pts = pts[idx]

    return pts.reshape(-1, 1, 2).astype(np.float32)

def _alpha_blend(bgr_a, bgr_b, alpha):
    return cv2.addWeighted(bgr_b, float(alpha), bgr_a, 1.0 - float(alpha), 0.0)

def _draw_live_splats(preview_bgr, pts_xy, cols_bgr, radius=5, alpha=0.45):
    if pts_xy is None or cols_bgr is None or len(pts_xy) == 0:
        return preview_bgr

    overlay = preview_bgr.copy()
    pts = pts_xy.reshape(-1, 2)
    cols = cols_bgr.reshape(-1, 3)

    n = len(pts)
    if n > LIVE_GHOST_MAX:
        idx = np.random.choice(n, LIVE_GHOST_MAX, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    for (x, y), c in zip(pts, cols):
        cv2.circle(
            overlay,
            (int(x), int(y)),
            int(radius),
            (int(c[0]), int(c[1]), int(c[2])),
            -1,
            lineType=cv2.LINE_AA
        )
    return _alpha_blend(preview_bgr, overlay, alpha)


# ================== CALIBRATION HELPERS ==================

def _make_chessboard_object_points(cols, rows):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp

def _find_chessboard(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pattern = (int(CHESSBOARD_COLS), int(CHESSBOARD_ROWS))
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=CHESSBOARD_FLAGS)
    if not ok or corners is None:
        return False, None, gray
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return True, corners, gray

def _solve_camera_calibration(objpoints, imgpoints, image_size):
    if len(objpoints) < 4 or len(imgpoints) < 4:
        return False, None, None, None

    flags = 0
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        flags=flags
    )
    if not ret:
        return False, None, None, None
    return True, K, D, float(ret)

def _undistort_if_needed(frame_bgr, K, D, use_undistort):
    if not use_undistort or K is None or D is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.9, (w, h))
    return cv2.undistort(frame_bgr, K, D, None, newK)


# ================== TRACKING ==================

def _lk_forward_backward(gray0, gray1, pts0):
    if pts0 is None or len(pts0) == 0:
        return None, None, None, None

    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
        gray0, gray1, pts0, None,
        winSize=LK_WIN,
        maxLevel=LK_LEVELS,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.03)
    )
    if p1 is None or st1 is None:
        return None, None, None, None

    p0b, st2, err2 = cv2.calcOpticalFlowPyrLK(
        gray1, gray0, p1, None,
        winSize=LK_WIN,
        maxLevel=LK_LEVELS,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.03)
    )
    if p0b is None or st2 is None:
        return None, None, None, None

    pts0f = pts0.reshape(-1, 2).astype(np.float32)
    p1f = p1.reshape(-1, 2).astype(np.float32)
    p0bf = p0b.reshape(-1, 2).astype(np.float32)

    st1 = st1.reshape(-1).astype(bool)
    st2 = st2.reshape(-1).astype(bool)
    fb = np.linalg.norm(pts0f - p0bf, axis=1)
    lkerr = np.zeros((len(pts0f),), dtype=np.float32)
    if err1 is not None:
        lkerr = err1.reshape(-1).astype(np.float32)

    keep = st1 & st2 & (fb <= float(FB_ERR_THRESH)) & (lkerr <= float(LK_ERR_THRESH))
    if np.sum(keep) == 0:
        return None, None, None, None

    fb_kept = fb[keep]
    lk_kept = lkerr[keep]

    conf_fb = 1.0 - np.clip(fb_kept / max(1e-6, float(FB_ERR_THRESH)), 0.0, 1.0)
    conf_lk = 1.0 - np.clip(lk_kept / max(1e-6, float(LK_ERR_THRESH)), 0.0, 1.0)
    conf = 0.65 * conf_fb + 0.35 * conf_lk
    conf = np.clip(conf.astype(np.float32), 0.0, 1.0)

    return (
        pts0f[keep].reshape(-1, 1, 2).astype(np.float32),
        p1f[keep].reshape(-1, 1, 2).astype(np.float32),
        conf,
        fb_kept.astype(np.float32)
    )


# ================== MOTION QUALITY GATE ==================

def _motion_quality_gate(pts0, pts1):
    """
    Returns:
      ok, reason, score_dict
    Gate tries to reject:
    - very low motion
    - motion explained almost entirely by flat affine rotation/scale
    """
    if pts0 is None or pts1 is None:
        return False, "NO TRACKS", {}

    p0 = pts0.reshape(-1, 2).astype(np.float32)
    p1 = pts1.reshape(-1, 2).astype(np.float32)

    if len(p0) < CALIB_MIN_POINTS:
        return False, "TOO FEW TRACKS", {}

    flow = p1 - p0
    flow_mag = np.linalg.norm(flow, axis=1)
    mean_motion = float(np.mean(flow_mag))

    good_motion = flow_mag > float(MIN_MOTION_PX)
    good_ratio = float(np.mean(good_motion.astype(np.float32))) if len(good_motion) else 0.0

    # Affine fit: if nearly all motion is explained by flat affine, depth evidence is weaker
    resid_mean = 0.0
    try:
        A, inliers = cv2.estimateAffinePartial2D(
            p0, p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=500,
            confidence=0.99,
            refineIters=10
        )
        if A is not None:
            ones = np.ones((len(p0), 1), dtype=np.float32)
            p0h = np.concatenate([p0, ones], axis=1)
            pred = (A @ p0h.T).T
            resid = np.linalg.norm(pred - p1, axis=1)
            resid_mean = float(np.mean(resid))
    except Exception:
        resid_mean = 0.0

    ok = (
        mean_motion >= float(MOTION_GATE_MIN_MEAN_PX) and
        resid_mean >= float(MOTION_GATE_MIN_PARALLAX_RESID) and
        good_ratio >= float(MOTION_GATE_MIN_GOOD_RATIO)
    )

    reason = "GOOD"
    if not ok:
        if mean_motion < float(MOTION_GATE_MIN_MEAN_PX):
            reason = "MOVE MORE / LOW PARALLAX"
        elif resid_mean < float(MOTION_GATE_MIN_PARALLAX_RESID):
            reason = "TOO MUCH ROTATION / FLAT MOTION"
        else:
            reason = "WEAK TRACK GEOMETRY"

    return ok, reason, {
        "mean_motion": mean_motion,
        "resid_mean": resid_mean,
        "good_ratio": good_ratio,
    }


# ================== CLOUD HELPERS ==================

def _normalize_cloud(pts3d):
    if pts3d is None or pts3d.shape[0] < 5:
        return None, None
    center = np.mean(pts3d, axis=0)
    pts_centered = pts3d - center
    max_range = float(np.max(np.linalg.norm(pts_centered, axis=1)))
    if max_range < 1e-6:
        return None, None
    pts_norm = (pts_centered / max_range).astype(np.float32)
    m = np.linalg.norm(pts_norm, axis=1) < float(CLOUD_RADIUS_CLAMP)
    pts_norm = pts_norm[m]
    return pts_norm, m

def _color_var_patch(img_bgr, pts_xy, patch_r=2):
    if img_bgr is None or pts_xy is None or len(pts_xy) == 0:
        return np.zeros((0,), dtype=np.float32)

    h, w = img_bgr.shape[:2]
    pr = int(max(1, patch_r))
    out = np.zeros((len(pts_xy),), dtype=np.float32)

    for i, (x, y) in enumerate(pts_xy):
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        x0 = max(0, xi - pr)
        x1 = min(w - 1, xi + pr)
        y0 = max(0, yi - pr)
        y1 = min(h - 1, yi + pr)
        patch = img_bgr[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        if patch.size == 0:
            out[i] = 0.0
        else:
            out[i] = float(np.var(patch.reshape(-1, 3), axis=0).mean())
    return out

def _score_parallax(pts0, pts1):
    if pts0 is None or pts1 is None or len(pts0) == 0:
        return np.zeros((0,), dtype=np.float32)
    d = np.linalg.norm(pts1.reshape(-1, 2) - pts0.reshape(-1, 2), axis=1)
    score = np.clip((d - MIN_MOTION_PX) / 8.0, 0.0, 1.0)
    return score.astype(np.float32)

def _triangulate_points(P1, P2, pts0, pts1):
    pts4d = cv2.triangulatePoints(
        P1, P2,
        pts0.reshape(-1, 2).T.astype(np.float64),
        pts1.reshape(-1, 2).T.astype(np.float64)
    )
    w4 = pts4d[3]
    w4 = np.where(np.abs(w4) < 1e-9, 1e-9, w4)
    return (pts4d[:3] / w4).T.astype(np.float32)

def _project_points(P, pts3d):
    X = np.concatenate([pts3d.astype(np.float64), np.ones((len(pts3d), 1), dtype=np.float64)], axis=1)
    x = (P @ X.T).T
    z = np.where(np.abs(x[:, 2]) < 1e-9, 1e-9, x[:, 2])
    uv = x[:, :2] / z[:, None]
    return uv.astype(np.float32)

def _reprojection_prune(P1, P2, pts3d, pts0, pts1, thresh=2.75):
    if pts3d is None or len(pts3d) == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)
    uv0 = _project_points(P1, pts3d)
    uv1 = _project_points(P2, pts3d)
    e0 = np.linalg.norm(uv0 - pts0.reshape(-1, 2), axis=1)
    e1 = np.linalg.norm(uv1 - pts1.reshape(-1, 2), axis=1)
    e = 0.5 * (e0 + e1)
    keep = e <= float(thresh)
    return keep, e.astype(np.float32)

def _triangulation_angles_deg(pts3d, R, t):
    """
    pts3d are in camera-1/world coordinates.
    Camera-1 center is at origin.
    Camera-2 center in camera-1/world coords is C2 = -R^T t
    """
    if pts3d is None or len(pts3d) == 0:
        return np.zeros((0,), dtype=np.float32)

    C1 = np.zeros((3,), dtype=np.float32)
    C2 = (-R.T @ t).reshape(3).astype(np.float32)

    v1 = pts3d.astype(np.float32) - C1.reshape(1, 3)
    v2 = pts3d.astype(np.float32) - C2.reshape(1, 3)

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = np.maximum(n1 * n2, 1e-9)

    c = np.sum(v1 * v2, axis=1) / denom
    c = np.clip(c, -1.0, 1.0)
    ang = np.degrees(np.arccos(c)).astype(np.float32)
    return ang

def _score_triangulation_angle(ang_deg):
    if ang_deg is None or len(ang_deg) == 0:
        return np.zeros((0,), dtype=np.float32)
    denom = max(1e-6, float(TRI_ANGLE_SOFT_MAX_DEG - TRI_ANGLE_MIN_DEG))
    s = (ang_deg.astype(np.float32) - float(TRI_ANGLE_MIN_DEG)) / denom
    return np.clip(s, 0.0, 1.0).astype(np.float32)

def _solve_pose_from_sparse(pts0, pts1, w, h, camera_matrix=None):
    if pts0 is None or pts1 is None:
        return None

    p0 = pts0.reshape(-1, 2).astype(np.float32)
    p1 = pts1.reshape(-1, 2).astype(np.float32)

    if len(p0) < CALIB_MIN_POINTS:
        return None

    # NEW: use real calibrated intrinsics if available
    K = None
    if isinstance(camera_matrix, np.ndarray) and camera_matrix.shape == (3, 3):
        K = camera_matrix.astype(np.float64)
    else:
        f = float(max(w, h))
        K = np.array([[f, 0, w / 2.0],
                      [0, f, h / 2.0],
                      [0, 0, 1]], dtype=np.float64)

    E, maskE = cv2.findEssentialMat(
        p0, p1, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=float(ESSENTIAL_RANSAC_THRESH)
    )
    if E is None or maskE is None:
        return None

    maskE = maskE.ravel().astype(bool)
    p0_in = p0[maskE]
    p1_in = p1[maskE]
    if len(p0_in) < CALIB_MIN_POINTS:
        return None

    _, R, t, maskPose = cv2.recoverPose(E, p0_in, p1_in, K)
    if maskPose is None:
        return None

    maskPose = maskPose.ravel().astype(bool)
    p0_fin = p0_in[maskPose]
    p1_fin = p1_in[maskPose]
    if len(p0_fin) < CALIB_MIN_POINTS:
        return None

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    return {
        "K": K,
        "R": R,
        "t": t,
        "P1": P1,
        "P2": P2,
        "pts0": p0_fin.reshape(-1, 1, 2).astype(np.float32),
        "pts1": p1_fin.reshape(-1, 1, 2).astype(np.float32)
    }

def _fit_plane_from_3(pts):
    a, b, c = pts[0], pts[1], pts[2]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-8:
        return None
    n = n / nn
    d = -float(np.dot(n, a))
    return n.astype(np.float32), float(d)

def _plane_distance(pts, n, d):
    return np.abs(np.dot(pts, n) + d)

def _apply_plane_snap(points, enabled=True):
    if not enabled or points is None or len(points) < PLANE_MIN_INLIERS:
        return points

    pts = points.copy().astype(np.float32)
    remaining = np.arange(len(pts))
    snapped = pts.copy()

    for _ in range(int(PLANE_MAX_COUNT)):
        if len(remaining) < int(PLANE_MIN_INLIERS):
            break

        best_inliers = None
        best_model = None
        rem_pts = pts[remaining]

        for _it in range(int(PLANE_RANSAC_ITERS)):
            if len(rem_pts) < 3:
                break
            idx = np.random.choice(len(rem_pts), 3, replace=False)
            model = _fit_plane_from_3(rem_pts[idx])
            if model is None:
                continue
            n, d = model
            dist = _plane_distance(rem_pts, n, d)
            inliers = dist < float(PLANE_DIST_THRESH)
            count = int(np.sum(inliers))
            if best_inliers is None or count > int(np.sum(best_inliers)):
                best_inliers = inliers
                best_model = (n, d)

        if best_inliers is None or best_model is None:
            break

        count = int(np.sum(best_inliers))
        if count < int(PLANE_MIN_INLIERS):
            break

        n, d = best_model
        rem_idx = remaining[best_inliers]
        p = snapped[rem_idx]
        signed = (np.dot(p, n) + d).reshape(-1, 1)
        proj = p - signed * n.reshape(1, 3)
        snapped[rem_idx] = p * (1.0 - float(PLANE_SNAP_BLEND)) + proj * float(PLANE_SNAP_BLEND)

        remaining = remaining[~best_inliers]

    return snapped.astype(np.float32)

def _build_cloud_from_pose(
    pose_info,
    sparse0, sparse1, sparse_conf,
    dense0, dense1, dense_conf,
    img0_bgr,
    grad0,
    plane_snap_enabled=True
):
    if pose_info is None:
        return None, None, None, "NONE (pose failed)"

    P1 = pose_info["P1"]
    P2 = pose_info["P2"]
    R = pose_info["R"]
    t = pose_info["t"]

    clouds = []
    colors = []
    confs = []

    # Sparse set
    if sparse0 is not None and sparse1 is not None and len(sparse0) >= CALIB_MIN_POINTS:
        pts3d_s = _triangulate_points(P1, P2, sparse0, sparse1)
        z = pts3d_s[:, 2]
        front = z > 0
        pts3d_s = pts3d_s[front]
        sparse0f = sparse0.reshape(-1, 2)[front]
        sparse1f = sparse1.reshape(-1, 2)[front]
        conf_s = sparse_conf[front] if sparse_conf is not None else np.ones((len(pts3d_s),), np.float32)

        keep_reproj, reproj_err = _reprojection_prune(P1, P2, pts3d_s, sparse0f, sparse1f, thresh=float(REPROJ_ERR_THRESH))
        pts3d_s = pts3d_s[keep_reproj]
        sparse0f = sparse0f[keep_reproj]
        sparse1f = sparse1f[keep_reproj]
        conf_s = conf_s[keep_reproj]
        reproj_err = reproj_err[keep_reproj]

        # NEW: triangulation-angle filter
        tri_ang = _triangulation_angles_deg(pts3d_s, R, t)
        keep_ang = tri_ang >= float(TRI_ANGLE_MIN_DEG)
        pts3d_s = pts3d_s[keep_ang]
        sparse0f = sparse0f[keep_ang]
        sparse1f = sparse1f[keep_ang]
        conf_s = conf_s[keep_ang]
        reproj_err = reproj_err[keep_ang]
        tri_ang = tri_ang[keep_ang]

        cols_s = _sample_colors_bgr_patch(img0_bgr, sparse0f, patch_r=int(COLOR_SAMPLE_PATCH))
        g_s = _sample_gradient_at_points(grad0, sparse0f)
        pv_s = _color_var_patch(img0_bgr, sparse0f, patch_r=2)

        g_score = np.clip(g_s / 120.0, 0.0, 1.0)
        pv_score = np.clip(pv_s / 160.0, 0.0, 1.0)
        prx_score = _score_parallax(sparse0f, sparse1f)
        rp_score = 1.0 - np.clip(reproj_err / max(1e-6, float(REPROJ_ERR_THRESH)), 0.0, 1.0)
        ang_score = _score_triangulation_angle(tri_ang)

        cscore = (
            0.28 * conf_s +
            0.18 * g_score +
            0.14 * prx_score +
            0.08 * pv_score +
            0.16 * rp_score +
            0.16 * ang_score
        )
        keep = cscore >= float(CONF_MIN_KEEP)

        if np.any(keep):
            clouds.append(pts3d_s[keep])
            colors.append(cols_s[keep])
            confs.append(cscore[keep].astype(np.float32))

    # Dense set
    if dense0 is not None and dense1 is not None and len(dense0) > 0:
        pts3d_d = _triangulate_points(P1, P2, dense0, dense1)
        z = pts3d_d[:, 2]
        front = z > 0
        pts3d_d = pts3d_d[front]
        dense0f = dense0.reshape(-1, 2)[front]
        dense1f = dense1.reshape(-1, 2)[front]
        conf_d = dense_conf[front] if dense_conf is not None else np.ones((len(pts3d_d),), np.float32)

        keep_reproj, reproj_err = _reprojection_prune(P1, P2, pts3d_d, dense0f, dense1f, thresh=float(REPROJ_ERR_THRESH * 1.15))
        pts3d_d = pts3d_d[keep_reproj]
        dense0f = dense0f[keep_reproj]
        dense1f = dense1f[keep_reproj]
        conf_d = conf_d[keep_reproj]
        reproj_err = reproj_err[keep_reproj]

        # NEW: triangulation-angle filter
        tri_ang = _triangulation_angles_deg(pts3d_d, R, t)
        keep_ang = tri_ang >= float(TRI_ANGLE_MIN_DEG)
        pts3d_d = pts3d_d[keep_ang]
        dense0f = dense0f[keep_ang]
        dense1f = dense1f[keep_ang]
        conf_d = conf_d[keep_ang]
        reproj_err = reproj_err[keep_ang]
        tri_ang = tri_ang[keep_ang]

        cols_d = _sample_colors_bgr_patch(img0_bgr, dense0f, patch_r=int(COLOR_SAMPLE_PATCH))
        g_d = _sample_gradient_at_points(grad0, dense0f)
        pv_d = _color_var_patch(img0_bgr, dense0f, patch_r=2)

        g_score = np.clip(g_d / 120.0, 0.0, 1.0)
        pv_score = np.clip(pv_d / 180.0, 0.0, 1.0)
        prx_score = _score_parallax(dense0f, dense1f)
        rp_score = 1.0 - np.clip(reproj_err / max(1e-6, float(REPROJ_ERR_THRESH * 1.15)), 0.0, 1.0)
        ang_score = _score_triangulation_angle(tri_ang)

        cscore = (
            0.22 * conf_d +
            0.23 * g_score +
            0.14 * prx_score +
            0.08 * pv_score +
            0.17 * rp_score +
            0.16 * ang_score
        )
        keep = cscore >= float(CONF_MIN_KEEP)

        if np.any(keep):
            clouds.append(pts3d_d[keep])
            colors.append(cols_d[keep])
            confs.append(cscore[keep].astype(np.float32))

    if len(clouds) == 0:
        return None, None, None, "WEAK (no valid triangulated points)"

    pts3d = np.concatenate(clouds, axis=0).astype(np.float32)
    cols3d = np.concatenate(colors, axis=0).astype(np.float32)
    c_all = np.concatenate(confs, axis=0).astype(np.float32)

    pts3d = _apply_plane_snap(pts3d, enabled=bool(plane_snap_enabled))

    pts_norm, keep_mask = _normalize_cloud(pts3d)
    if pts_norm is None:
        return None, None, None, "WEAK (normalize failed)"

    cols_norm = cols3d[keep_mask]
    c_norm = c_all[keep_mask]

    # visual slight confidence boost to color
    c_vis = np.clip(c_norm.reshape(-1, 1), 0.0, 1.0)
    cols_norm = cols_norm * (0.75 + 0.25 * c_vis)

    if MERGE_COLOR_CLAMP:
        cols_norm = np.clip(cols_norm, 0.0, 255.0)

    n = min(len(pts_norm), len(cols_norm), len(c_norm))
    pts_norm = pts_norm[:n]
    cols_norm = cols_norm[:n]
    c_norm = c_norm[:n]

    if n < 120:
        q = "WEAK"
    elif n < 260:
        q = "OK"
    else:
        q = "RICH"

    return pts_norm.astype(np.float32), cols_norm.astype(np.float32), c_norm.astype(np.float32), f"{q} HYBRID ({n} pts)"

def build_hybrid_cloud_from_pair(
    sparse0, sparse1, sparse_conf,
    dense0, dense1, dense_conf,
    img0_bgr, gray0_blur, w, h,
    plane_snap_enabled=True,
    camera_matrix=None
):
    if sparse0 is None or sparse1 is None:
        return None, None, None, "NONE (missing sparse points)"

    ok_motion, motion_reason, motion_info = _motion_quality_gate(sparse0, sparse1)
    if not ok_motion:
        return None, None, None, f"REJECTED ({motion_reason})"

    d_sparse = np.linalg.norm(sparse1.reshape(-1, 2) - sparse0.reshape(-1, 2), axis=1)
    ms = d_sparse > float(MIN_MOTION_PX)
    sparse0 = sparse0[ms]
    sparse1 = sparse1[ms]
    sparse_conf = sparse_conf[ms] if sparse_conf is not None and len(sparse_conf) == len(ms) else None

    if sparse0 is None or len(sparse0) < CALIB_MIN_POINTS:
        return None, None, None, "WEAK (not enough sparse motion)"

    if dense0 is not None and dense1 is not None and len(dense0) > 0:
        d_dense = np.linalg.norm(dense1.reshape(-1, 2) - dense0.reshape(-1, 2), axis=1)
        md = d_dense > float(MIN_MOTION_PX)
        if np.any(md):
            dense0 = dense0[md]
            dense1 = dense1[md]
            dense_conf = dense_conf[md] if dense_conf is not None and len(dense_conf) == len(md) else None
        else:
            dense0, dense1, dense_conf = None, None, None

    pose_info = _solve_pose_from_sparse(sparse0, sparse1, w, h, camera_matrix=camera_matrix)
    grad0 = _gradient_mag(gray0_blur)

    return _build_cloud_from_pose(
        pose_info,
        sparse0, sparse1, sparse_conf,
        dense0, dense1, dense_conf,
        img0_bgr,
        grad0,
        plane_snap_enabled=plane_snap_enabled
    )


# ================== ACCUMULATION / MERGE ==================

def voxel_merge(points, colors, voxel_size=0.02, max_points=15000, weights=None):
    if points is None or colors is None or len(points) == 0:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0,), np.float32)
        )

    n = min(len(points), len(colors))
    pts = points[:n].astype(np.float32)
    cols = colors[:n].astype(np.float32)

    if weights is None or len(weights) == 0:
        wts = np.ones((n,), dtype=np.float32)
    else:
        wts = np.array(weights[:n], dtype=np.float32).reshape(-1)
        if len(wts) < n:
            pad = np.ones((n - len(wts),), dtype=np.float32)
            wts = np.concatenate([wts, pad], axis=0)

    wts = np.maximum(wts, 1e-6).astype(np.float32)

    vs = float(max(1e-6, voxel_size))
    key = np.floor(pts / vs).astype(np.int32)

    kx = key[:, 0].astype(np.int64)
    ky = key[:, 1].astype(np.int64)
    kz = key[:, 2].astype(np.int64)
    h = (kx * 73856093) ^ (ky * 19349663) ^ (kz * 83492791)

    order = np.argsort(h)
    h = h[order]
    pts = pts[order]
    cols = cols[order]
    wts = wts[order]

    uniq_h, idx_start = np.unique(h, return_index=True)
    idx_start = idx_start.tolist() + [len(h)]

    out_pts = []
    out_cols = []
    out_wts = []

    for i in range(len(uniq_h)):
        a = idx_start[i]
        b = idx_start[i + 1]
        p = pts[a:b]
        c = cols[a:b]
        w = wts[a:b].reshape(-1, 1)

        ws = float(np.sum(w))
        if ws <= 1e-9:
            continue

        p_mean = np.sum(p * w, axis=0) / ws
        c_mean = np.sum(c * w, axis=0) / ws

        out_pts.append(p_mean)
        out_cols.append(c_mean)
        out_wts.append(ws)

    out_pts = np.array(out_pts, dtype=np.float32)
    out_cols = np.array(out_cols, dtype=np.float32)
    out_wts = np.array(out_wts, dtype=np.float32)

    if len(out_pts) > int(max_points):
        # prefer keeping stronger weighted voxels
        order_keep = np.argsort(out_wts)[::-1]
        keep = order_keep[:int(max_points)]
        out_pts = out_pts[keep]
        out_cols = out_cols[keep]
        out_wts = out_wts[keep]

    return out_pts, np.clip(out_cols, 0.0, 255.0).astype(np.float32), out_wts.astype(np.float32)


# ================== VIEWER ==================

def _gaussian_alpha(dx, dy, sigma):
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

def _triangle_mask(hh, ww):
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cx = ww // 2
    cy = hh // 2
    r = min(cx, cy)
    pts = np.array([
        [cx, cy - r],
        [cx - r, cy + r],
        [cx + r, cy + r],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return (mask.astype(np.float32) / 255.0)

def _draw_gaussian_shape(img, cx, cy, radius, color_bgr, intensity=1.0, shape="TRI"):
    h, w = img.shape[:2]
    r = int(max(1, min(SPLAT_MAX_RADIUS, radius)))

    x0 = max(0, cx - r)
    x1 = min(w - 1, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h - 1, cy + r)
    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32) - float(cx)
    ys = np.arange(y0, y1 + 1, dtype=np.float32) - float(cy)
    xx, yy = np.meshgrid(xs, ys)

    sigma = max(1e-6, r * 0.55)
    a = _gaussian_alpha(xx, yy, sigma) * float(np.clip(intensity, 0.0, 1.0))

    hh = (y1 - y0 + 1)
    ww = (x1 - x0 + 1)
    if str(shape).upper() == "CUBE":
        m = np.ones((hh, ww), dtype=np.float32)
    else:
        m = _triangle_mask(hh, ww)

    a = np.clip(a * m, 0.0, 1.0)
    patch = img[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    c = np.array(color_bgr, dtype=np.float32)[None, None, :]
    patch = patch * (1.0 - a[..., None]) + c * (a[..., None])
    img[y0:y1 + 1, x0:x1 + 1] = patch.astype(np.uint8)

def _render_room_map(points_3d, W, H):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if points_3d is None or len(points_3d) == 0:
        cv2.putText(img, "NO CLOUD", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img

    pts = points_3d.astype(np.float32)
    XZ = pts[:, [0, 2]]
    mn = XZ.min(axis=0)
    mx = XZ.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    norm = (XZ - mn) / span

    margin = 0.08
    norm = norm * (1 - 2 * margin) + margin
    px = (norm[:, 0] * (W - 1)).astype(np.int32)
    py = ((1.0 - norm[:, 1]) * (H - 1)).astype(np.int32)

    n = len(px)
    if n > 1800:
        idx = np.random.choice(n, 1800, replace=False)
        px, py, norm = px[idx], py[idx], norm[idx]
        n = len(px)

    for i in range(n):
        cv2.circle(img, (int(px[i]), int(py[i])), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    for i in range(n):
        d = norm - norm[i]
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        order = np.argsort(dist)
        links = 0
        for j in order[1:]:
            if dist[j] > 0.08:
                break
            cv2.line(img, (int(px[i]), int(py[i])), (int(px[j]), int(py[j])), (0, 0, 255), 1, cv2.LINE_AA)
            links += 1
            if links >= 2:
                break

    cv2.putText(img, "ROOM MAP", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return img


class SplatViewer(Image):
    def __init__(self, points_3d, colors_bgr, quality_text="MODEL", **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (1, 1)
        try:
            self.fit_mode = "contain"
        except Exception:
            pass

        self.points = points_3d.astype(np.float32) if isinstance(points_3d, np.ndarray) else np.zeros((0, 3), np.float32)
        self.colors = colors_bgr.astype(np.float32) if isinstance(colors_bgr, np.ndarray) else np.zeros((0, 3), np.float32)

        n = min(len(self.points), len(self.colors))
        self.points = self.points[:n]
        self.colors = self.colors[:n]

        self.yaw = 0.0
        self.pitch = 0.0
        self.dragging = False
        self.last_touch = None

        self.base_splat_radius = float(DEFAULT_SPLAT_RADIUS)
        self.quality_text = quality_text
        self.mode = "SPLAT"

        self.shape_mode = str(DEFAULT_SHAPE_MODE).upper().strip()
        if self.shape_mode not in ("TRI", "CUBE"):
            self.shape_mode = "TRI"

        self.texture = Texture.create(size=(SCREEN_W, SCREEN_H), colorfmt="rgb")
        self._event = Clock.schedule_interval(self._tick, 1 / 30.0)

    def set_splat_radius(self, r):
        self.base_splat_radius = float(max(1.0, min(12.0, r)))

    def set_mode(self, m):
        m = str(m).upper().strip()
        if m in ("SPLAT", "ROOM"):
            self.mode = m

    def set_shape_mode(self, m):
        m = str(m).upper().strip()
        if m in ("TRI", "CUBE"):
            self.shape_mode = m

    def stop(self):
        if self._event:
            self._event.cancel()
            self._event = None

    def on_touch_down(self, touch):
        if self.mode != "SPLAT":
            return True
        self.dragging = True
        self.last_touch = (touch.x, touch.y)
        return True

    def on_touch_move(self, touch):
        if self.mode != "SPLAT":
            return True
        if not self.dragging or self.last_touch is None:
            return True
        x, y = touch.x, touch.y
        lx, ly = self.last_touch
        dx = (x - lx) / 200.0
        dy = (y - ly) / 200.0
        self.yaw += dx
        self.pitch += dy
        self.pitch = max(-1.2, min(1.2, self.pitch))
        self.last_touch = (x, y)
        return True

    def on_touch_up(self, touch):
        self.dragging = False
        self.last_touch = None
        return True

    def _tick(self, dt):
        frame = self.render_frame()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 0)
        if self.texture is None or self.texture.size != (SCREEN_W, SCREEN_H):
            self.texture = Texture.create(size=(SCREEN_W, SCREEN_H), colorfmt="rgb")
        self.texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.canvas.ask_update()

    def render_frame(self):
        if self.mode == "ROOM":
            img = _render_room_map(self.points, SCREEN_W, SCREEN_H)
            cv2.putText(img, self.quality_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            return img

        img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        if len(self.points) == 0:
            cv2.putText(img, "NO CLOUD", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return img

        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)

        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        xr = x * cy + z * sy
        zr = -x * sy + z * cy

        yr = y * cp - zr * sp
        zr2 = y * sp + zr * cp
        zr2 = zr2 + float(FOCAL_LEN)

        sx = xr / zr2
        sy2 = yr / zr2

        px = (sx * (SCREEN_W / 2.0) + (SCREEN_W / 2.0)).astype(np.int32)
        py = (sy2 * (SCREEN_H / 2.0) + (SCREEN_H / 2.0)).astype(np.int32)

        zc = np.clip(zr2, 0.35, 4.0)
        size = self.base_splat_radius * (1.35 / zc)

        order = np.argsort(zc)[::-1]
        for i in order:
            xpi = int(px[i])
            ypi = int(py[i])
            if xpi < 0 or xpi >= SCREEN_W or ypi < 0 or ypi >= SCREEN_H:
                continue
            inten = float(np.clip((1.8 / zc[i]) * 0.6, 0.12, 1.0))
            _draw_gaussian_shape(
                img,
                xpi, ypi,
                float(size[i]),
                self.colors[i],
                intensity=inten,
                shape=self.shape_mode
            )

        cv2.putText(img, "K-CVD HYBRID GAUSSIAN VIEWER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"{self.quality_text}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(img, f"SHAPE: {self.shape_mode}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return img


# ================== CAMERA ==================

class CameraFeed(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (1, 1)
        try:
            self.fit_mode = "contain"
        except Exception:
            pass

        self.cap = None
        self._open_any_camera()
        self.texture = None

        self.calibrating = False
        self.calib_frames_left = 0
        self.reseed_cooldown = 0

        self.gray_prev = None
        self.img0_bgr = None
        self.gray0_blur = None

        self.pts0_sparse = None
        self.pts_prev_sparse = None
        self.conf_sparse = None

        self.pts0_dense = None
        self.pts_prev_dense = None
        self.conf_dense = None

        self.last_quality = "MODEL: (none)"
        self.last_gate_reason = ""

        self.live_overlay = bool(LIVE_OVERLAY_ENABLED_DEFAULT)
        self.live_overlay_radius = int(LIVE_OVERLAY_RADIUS)

        self.max_feature_points = int(MAX_FEATURE_POINTS_DEFAULT)
        self.max_edge_points = int(MAX_EDGE_POINTS_DEFAULT)

        self.keyframes = []

        # calibration store
        self.cal_objpoints = []
        self.cal_imgpoints = []
        self.cal_last_detect_ok = False

        # loaded camera intrinsics
        self.camera_matrix, self.dist_coeffs, self.cal_image_size = load_camera_calibration()
        self.use_undistort = self.camera_matrix is not None and self.dist_coeffs is not None

        self.plane_snap_enabled = bool(PLANE_SNAP_ENABLED_DEFAULT)

        Clock.schedule_interval(self.update, 1.0 / FPS_TARGET)

    def _try_open_index(self, index):
        cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _open_any_camera(self):
        for idx in CAM_INDEX_TRY_LIST:
            for t in range(int(CAM_TRIES)):
                cap = self._try_open_index(idx)
                if cap is not None:
                    self.cap = cap
                    print(f"Camera opened index {idx} (try {t+1}/{CAM_TRIES})")
                    return
                time.sleep(float(CAM_RETRY_DELAY))
        self.cap = None
        print("ERROR: Could not open any camera index.")

    def set_live_overlay(self, enabled: bool):
        self.live_overlay = bool(enabled)

    def set_density(self, max_points: int):
        self.max_feature_points = int(max(200, min(1800, max_points)))
        self.max_edge_points = int(max(180, min(1600, int(max_points * 0.9))))

    def set_use_undistort(self, enabled: bool):
        if enabled and (self.camera_matrix is None or self.dist_coeffs is None):
            self.use_undistort = False
        else:
            self.use_undistort = bool(enabled)

    def set_plane_snap(self, enabled: bool):
        self.plane_snap_enabled = bool(enabled)

    def capture_calibration_frame(self):
        if self.cap is None:
            self.last_quality = "CAL: NO CAMERA"
            return False

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.last_quality = "CAL: NO FRAME"
            return False

        # detect on raw frame, not undistorted
        ok, corners, gray = _find_chessboard(frame)
        self.cal_last_detect_ok = bool(ok)

        if not ok or corners is None:
            self.last_quality = "CAL: BOARD NOT FOUND"
            return False

        objp = _make_chessboard_object_points(CHESSBOARD_COLS, CHESSBOARD_ROWS)
        self.cal_objpoints.append(objp.astype(np.float32))
        self.cal_imgpoints.append(corners.astype(np.float32))

        self.last_quality = f"CAL: CAPTURED {len(self.cal_objpoints)}"
        return True

    def solve_camera_calibration(self):
        if self.cap is None:
            self.last_quality = "CAL: NO CAMERA"
            return False

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.last_quality = "CAL: NO FRAME"
            return False

        h, w = frame.shape[:2]
        ok, K, D, reproj = _solve_camera_calibration(
            self.cal_objpoints,
            self.cal_imgpoints,
            (w, h)
        )
        if not ok or K is None or D is None:
            self.last_quality = "CAL: SOLVE FAILED"
            return False

        self.camera_matrix = K
        self.dist_coeffs = D
        self.cal_image_size = (w, h)
        self.use_undistort = True

        save_camera_calibration(K, D, (w, h))
        self.last_quality = f"CAL: SOLVED reproj={reproj:.3f}"
        return True

    def clear_calibration_samples(self):
        self.cal_objpoints = []
        self.cal_imgpoints = []
        self.cal_last_detect_ok = False
        self.last_quality = "CAL: CLEARED"

    def start_calib_splat(self):
        if self.cap is None:
            self.last_quality = "MODEL: NONE (no camera)"
            return False

        ret, frame_raw = self.cap.read()
        if not ret or frame_raw is None:
            self.last_quality = "MODEL: NONE (no frame)"
            return False

        frame = _undistort_if_needed(frame_raw, self.camera_matrix, self.dist_coeffs, self.use_undistort)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 1)

        pts_sparse = _good_features(gray_blur, self.max_feature_points)
        pts_dense = _sample_edge_points(gray_blur, self.max_edge_points, grid_stride=int(EDGE_GRID_STRIDE))

        if pts_sparse is None or len(pts_sparse) < CALIB_MIN_POINTS:
            self.last_quality = "MODEL: WEAK (no sparse features)"
            return False

        self.calibrating = True
        self.calib_frames_left = int(CALIB_FRAMES)
        self.reseed_cooldown = 0

        self.gray_prev = gray_blur.copy()
        self.gray0_blur = gray_blur.copy()
        self.img0_bgr = frame.copy()

        self.pts0_sparse = pts_sparse.copy()
        self.pts_prev_sparse = pts_sparse.copy()
        self.conf_sparse = np.ones((len(pts_sparse),), dtype=np.float32)

        self.pts0_dense = pts_dense.copy() if pts_dense is not None else None
        self.pts_prev_dense = pts_dense.copy() if pts_dense is not None else None
        self.conf_dense = np.ones((len(pts_dense),), dtype=np.float32) if pts_dense is not None else None

        self.keyframes = []
        self.last_gate_reason = ""
        self.last_quality = "MODEL: CAPTURING..."
        return True

    def _reseed_features(self, gray_blur):
        pts_sparse = _good_features(gray_blur, self.max_feature_points)
        if pts_sparse is None or len(pts_sparse) < CALIB_MIN_POINTS:
            return False

        self.pts_prev_sparse = pts_sparse.copy()
        self.pts0_sparse = pts_sparse.copy()
        self.conf_sparse = np.ones((len(pts_sparse),), dtype=np.float32)

        pts_dense = _sample_edge_points(gray_blur, self.max_edge_points, grid_stride=int(EDGE_GRID_STRIDE))
        self.pts_prev_dense = pts_dense.copy() if pts_dense is not None else None
        self.pts0_dense = pts_dense.copy() if pts_dense is not None else None
        self.conf_dense = np.ones((len(pts_dense),), dtype=np.float32) if pts_dense is not None else None
        return True

    def _capture_keyframe(self):
        if self.img0_bgr is None or self.gray0_blur is None:
            return

        item = {
            "img0_bgr": self.img0_bgr.copy(),
            "gray0_blur": self.gray0_blur.copy(),
            "sparse0": self.pts0_sparse.copy() if self.pts0_sparse is not None else None,
            "sparse1": self.pts_prev_sparse.copy() if self.pts_prev_sparse is not None else None,
            "conf_sparse": self.conf_sparse.copy() if self.conf_sparse is not None else None,
            "dense0": self.pts0_dense.copy() if self.pts0_dense is not None else None,
            "dense1": self.pts_prev_dense.copy() if self.pts_prev_dense is not None else None,
            "conf_dense": self.conf_dense.copy() if self.conf_dense is not None else None,
        }
        self.keyframes.append(item)

    def finish_splat(self, frame_for_dims):
        clouds = []
        cols_all = []
        wts_all = []

        if frame_for_dims is None:
            frame_for_dims = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        h, w = frame_for_dims.shape[:2]

        if len(self.keyframes) == 0 and self.img0_bgr is not None and self.pts_prev_sparse is not None:
            self._capture_keyframe()

        accepted = 0
        rejected = 0

        for kf in self.keyframes:
            cloud, cols, weights, quality = build_hybrid_cloud_from_pair(
                kf["sparse0"], kf["sparse1"], kf["conf_sparse"],
                kf["dense0"], kf["dense1"], kf["conf_dense"],
                kf["img0_bgr"], kf["gray0_blur"], w, h,
                plane_snap_enabled=bool(self.plane_snap_enabled),
                camera_matrix=self.camera_matrix
            )
            if cloud is not None and cols is not None and weights is not None and len(cloud) > 0:
                clouds.append(cloud.astype(np.float32))
                cols_all.append(cols.astype(np.float32))
                wts_all.append(weights.astype(np.float32))
                accepted += 1
            else:
                rejected += 1

        self.calibrating = False
        self.calib_frames_left = 0
        self.reseed_cooldown = 0
        self.gray_prev = None
        self.img0_bgr = None
        self.gray0_blur = None
        self.pts0_sparse = None
        self.pts_prev_sparse = None
        self.conf_sparse = None
        self.pts0_dense = None
        self.pts_prev_dense = None
        self.conf_dense = None
        self.keyframes = []

        if len(clouds) == 0:
            return None, None, None, "NONE (no usable hybrid keyframes)"

        pts = np.concatenate(clouds, axis=0).astype(np.float32)
        cols = np.concatenate(cols_all, axis=0).astype(np.float32)
        wts = np.concatenate(wts_all, axis=0).astype(np.float32)

        pts, cols, wts = voxel_merge(
            pts, cols,
            voxel_size=float(VOXEL_SIZE * 0.85),
            max_points=int(max(2500, ACC_MAX_POINTS // 2)),
            weights=wts
        )

        n = len(pts)
        if n < 120:
            quality = f"WEAK HYBRID FUSED ({n} pts, acc:{accepted} rej:{rejected})"
        elif n < 350:
            quality = f"OK HYBRID FUSED ({n} pts, acc:{accepted} rej:{rejected})"
        else:
            quality = f"RICH HYBRID FUSED ({n} pts, acc:{accepted} rej:{rejected})"

        return pts.astype(np.float32), cols.astype(np.float32), wts.astype(np.float32), quality

    def update(self, dt):
        if self.cap is None:
            return

        ret, frame_raw = self.cap.read()
        if not ret or frame_raw is None:
            return

        frame = _undistort_if_needed(frame_raw, self.camera_matrix, self.dist_coeffs, self.use_undistort)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 1)

        live_pts = None
        live_cols = None

        if self.calibrating and self.gray_prev is not None:
            if self.reseed_cooldown > 0:
                self.reseed_cooldown -= 1

            # Sparse tracking
            if self.pts_prev_sparse is not None and len(self.pts_prev_sparse) > 0:
                p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
                    self.gray_prev, gray_blur, self.pts_prev_sparse, None,
                    winSize=LK_WIN, maxLevel=LK_LEVELS,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.03)
                )
                p0b, st2, err2 = cv2.calcOpticalFlowPyrLK(
                    gray_blur, self.gray_prev, p1, None,
                    winSize=LK_WIN, maxLevel=LK_LEVELS,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.03)
                )

                prevf = self.pts_prev_sparse.reshape(-1, 2).astype(np.float32)
                basef = self.pts0_sparse.reshape(-1, 2).astype(np.float32)
                p1f = p1.reshape(-1, 2).astype(np.float32)
                p0bf = p0b.reshape(-1, 2).astype(np.float32)
                st1b = st1.reshape(-1).astype(bool)
                st2b = st2.reshape(-1).astype(bool)
                errv = err1.reshape(-1).astype(np.float32) if err1 is not None else np.zeros((len(prevf),), np.float32)
                fb = np.linalg.norm(prevf - p0bf, axis=1)
                keep = st1b & st2b & (fb <= float(FB_ERR_THRESH)) & (errv <= float(LK_ERR_THRESH))

                if np.any(keep):
                    self.pts_prev_sparse = p1f[keep].reshape(-1, 1, 2).astype(np.float32)
                    self.pts0_sparse = basef[keep].reshape(-1, 1, 2).astype(np.float32)

                    conf_fb = 1.0 - np.clip(fb[keep] / max(1e-6, float(FB_ERR_THRESH)), 0.0, 1.0)
                    conf_lk = 1.0 - np.clip(errv[keep] / max(1e-6, float(LK_ERR_THRESH)), 0.0, 1.0)
                    self.conf_sparse = np.clip(0.65 * conf_fb + 0.35 * conf_lk, 0.0, 1.0).astype(np.float32)
                else:
                    self.pts_prev_sparse = None
                    self.pts0_sparse = None
                    self.conf_sparse = None

            # Dense tracking
            if self.pts_prev_dense is not None and len(self.pts_prev_dense) > 0:
                p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
                    self.gray_prev, gray_blur, self.pts_prev_dense, None,
                    winSize=LK_WIN, maxLevel=LK_LEVELS,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.03)
                )
                if p1 is not None and st1 is not None:
                    p0b, st2, err2 = cv2.calcOpticalFlowPyrLK(
                        gray_blur, self.gray_prev, p1, None,
                        winSize=LK_WIN, maxLevel=LK_LEVELS,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.03)
                    )

                    prevf = self.pts_prev_dense.reshape(-1, 2).astype(np.float32)
                    basef = self.pts0_dense.reshape(-1, 2).astype(np.float32)
                    p1f = p1.reshape(-1, 2).astype(np.float32)
                    p0bf = p0b.reshape(-1, 2).astype(np.float32)
                    st1b = st1.reshape(-1).astype(bool)
                    st2b = st2.reshape(-1).astype(bool)
                    errv = err1.reshape(-1).astype(np.float32) if err1 is not None else np.zeros((len(prevf),), np.float32)

                    fb = np.linalg.norm(prevf - p0bf, axis=1)
                    grad_now = _gradient_mag(gray_blur)
                    grad_score = np.clip(_sample_gradient_at_points(grad_now, p1f) / 120.0, 0.0, 1.0)

                    keep = st1b & st2b & (fb <= (float(FB_ERR_THRESH) * 1.25)) & (errv <= (float(LK_ERR_THRESH) * 1.15))
                    if np.any(keep):
                        self.pts_prev_dense = p1f[keep].reshape(-1, 1, 2).astype(np.float32)
                        self.pts0_dense = basef[keep].reshape(-1, 1, 2).astype(np.float32)

                        conf_fb = 1.0 - np.clip(fb[keep] / max(1e-6, float(FB_ERR_THRESH) * 1.25), 0.0, 1.0)
                        conf_lk = 1.0 - np.clip(errv[keep] / max(1e-6, float(LK_ERR_THRESH) * 1.15), 0.0, 1.0)
                        conf = 0.50 * conf_fb + 0.20 * conf_lk + 0.30 * grad_score[keep]
                        self.conf_dense = np.clip(conf, 0.0, 1.0).astype(np.float32)
                    else:
                        self.pts_prev_dense = None
                        self.pts0_dense = None
                        self.conf_dense = None

            self.gray_prev = gray_blur.copy()

            # live gate hint
            if self.pts0_sparse is not None and self.pts_prev_sparse is not None and len(self.pts0_sparse) >= CALIB_MIN_POINTS:
                ok_gate, reason, info = _motion_quality_gate(self.pts0_sparse, self.pts_prev_sparse)
                self.last_gate_reason = reason
            else:
                self.last_gate_reason = "TRACKING..."

            tracked_sparse = 0 if self.pts_prev_sparse is None else len(self.pts_prev_sparse)
            if tracked_sparse < MIN_TRACKED_BEFORE_RESEED and self.reseed_cooldown == 0:
                if self._reseed_features(gray_blur):
                    self.reseed_cooldown = int(RESEED_COOLDOWN_FRAMES)

            step_done = CALIB_FRAMES - self.calib_frames_left + 1
            if step_done in KEYFRAME_STEPS:
                self._capture_keyframe()

            self.calib_frames_left -= 1

            live_sparse = self.pts_prev_sparse.reshape(-1, 2) if self.pts_prev_sparse is not None else np.zeros((0, 2), np.float32)
            live_dense = self.pts_prev_dense.reshape(-1, 2) if self.pts_prev_dense is not None else np.zeros((0, 2), np.float32)

            if len(live_sparse) > 0 and len(live_dense) > 0:
                live_pts = np.concatenate([live_sparse, live_dense], axis=0)
            elif len(live_sparse) > 0:
                live_pts = live_sparse
            elif len(live_dense) > 0:
                live_pts = live_dense
            else:
                live_pts = None

            if live_pts is not None and len(live_pts) > 0:
                live_cols = _sample_colors_bgr_patch(frame, live_pts, patch_r=int(COLOR_SAMPLE_PATCH))

            if self.calib_frames_left <= 0:
                cloud, cols, weights, quality = self.finish_splat(frame)
                self.last_quality = "MODEL: " + str(quality)
                app = App.get_running_app()
                if app is not None and hasattr(app, "_add_cloud"):
                    app._add_cloud(cloud, cols, weights, quality)

        preview = cv2.convertScaleAbs(frame, alpha=float(HUD_TINT_ALPHA))
        cv2.putText(preview, "K-CVD HYBRID ACCUM SPLAT V3.1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, HUD_COLOR, 2)
        cv2.putText(preview, self.last_quality, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.54, HUD_COLOR, 2)

        cal_state = "UNDIST: ON" if self.use_undistort else "UNDIST: OFF"
        cal_have = "CAL READY" if (self.camera_matrix is not None and self.dist_coeffs is not None) else "NO CAL"
        cv2.putText(preview, f"{cal_have} | {cal_state} | CAL SAMPLES:{len(self.cal_objpoints)}", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, HUD_COLOR, 2)

        if self.calibrating:
            ps = 0 if self.pts_prev_sparse is None else len(self.pts_prev_sparse)
            pd = 0 if self.pts_prev_dense is None else len(self.pts_prev_dense)
            cv2.putText(
                preview,
                f"CAPTURE: {self.calib_frames_left}  sparse:{ps} dense:{pd} keys:{len(self.keyframes)}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                HUD_COLOR,
                2
            )
            cv2.putText(
                preview,
                f"GATE: {self.last_gate_reason}",
                (10, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                HUD_COLOR,
                2
            )
            if self.live_overlay and live_pts is not None and live_cols is not None:
                preview = _draw_live_splats(
                    preview,
                    live_pts,
                    live_cols,
                    radius=int(self.live_overlay_radius),
                    alpha=float(LIVE_OVERLAY_ALPHA)
                )
        else:
            # draw chessboard hint overlay when visible
            ok, corners, _ = _find_chessboard(frame_raw)
            if ok and corners is not None:
                cv2.drawChessboardCorners(preview, (CHESSBOARD_COLS, CHESSBOARD_ROWS), corners, True)
                cv2.putText(preview, "CHESSBOARD FOUND", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLOR, 2)

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 0)
        hh, ww = rgb.shape[:2]

        if self.texture is None or self.texture.size != (ww, hh):
            self.texture = Texture.create(size=(ww, hh), colorfmt="rgb")

        self.texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.canvas.ask_update()

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


# ================== APP ==================

class SplatApp(App):
    def build(self):
        ensure_scans_dir()
        root = FloatLayout()

        if ANDROID_PERMS_AVAILABLE:
            try:
                request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
            except Exception as e:
                print("Permission request failed:", e)

        self.cam = CameraFeed()
        root.add_widget(self.cam)

        self.acc_points = np.zeros((0, 3), dtype=np.float32)
        self.acc_colors = np.zeros((0, 3), dtype=np.float32)
        self.acc_weights = np.zeros((0,), dtype=np.float32)

        pts, cols, wts, meta = load_cloud()
        if pts is not None and cols is not None:
            self.acc_points = pts
            self.acc_colors = cols
            self.acc_weights = wts if wts is not None else np.ones((len(pts),), dtype=np.float32)
            self.cam.last_quality = "MODEL: (loaded accumulated)"

        # -----------------------------
        # ROW 1: scan toggles
        # -----------------------------
        top1 = BoxLayout(
            orientation="horizontal",
            size_hint=(1, 0.085),
            pos_hint={"x": 0, "top": 1.0},
            padding=8,
            spacing=8
        )

        self.tog_overlay = ToggleButton(
            text="LIVE: ON",
            state="down",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1),
            size_hint=(0.34, 1)
        )
        self.tog_overlay.bind(on_release=self.on_overlay_toggle)

        self.tog_undist = ToggleButton(
            text="UNDIST: ON" if self.cam.use_undistort else "UNDIST: OFF",
            state="down" if self.cam.use_undistort else "normal",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1),
            size_hint=(0.33, 1)
        )
        self.tog_undist.bind(on_release=self.on_undist_toggle)

        self.tog_plane = ToggleButton(
            text="PLANE: ON" if self.cam.plane_snap_enabled else "PLANE: OFF",
            state="down" if self.cam.plane_snap_enabled else "normal",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1),
            size_hint=(0.33, 1)
        )
        self.tog_plane.bind(on_release=self.on_plane_toggle)

        top1.add_widget(self.tog_overlay)
        top1.add_widget(self.tog_undist)
        top1.add_widget(self.tog_plane)
        root.add_widget(top1)

        # -----------------------------
        # ROW 2: density slider only
        # -----------------------------
        top2 = BoxLayout(
            orientation="horizontal",
            size_hint=(1, 0.085),
            pos_hint={"x": 0, "top": 0.91},
            padding=8,
            spacing=8
        )

        self.lbl_density = Label(
            text=f"POINTS: {MAX_FEATURE_POINTS_DEFAULT}",
            color=(1, 0, 0, 1),
            size_hint=(0.28, 1)
        )

        self.sld_density = Slider(
            min=300,
            max=1800,
            value=MAX_FEATURE_POINTS_DEFAULT,
            size_hint=(0.72, 1)
        )
        self.sld_density.bind(value=self.on_density_change)

        top2.add_widget(self.lbl_density)
        top2.add_widget(self.sld_density)
        root.add_widget(top2)

        # -----------------------------
        # ROW 3: camera calibration buttons
        # -----------------------------
        top3 = BoxLayout(
            orientation="horizontal",
            size_hint=(1, 0.085),
            pos_hint={"x": 0, "top": 0.82},
            padding=8,
            spacing=8
        )

        btn_cap_cal = Button(
            text="CAP CAL",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1),
            size_hint=(0.25, 1)
        )
        btn_cap_cal.bind(on_release=self.on_capture_cal)

        btn_solve_cal = Button(
            text="SOLVE CAL",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1),
            size_hint=(0.25, 1)
        )
        btn_solve_cal.bind(on_release=self.on_solve_cal)

        btn_clear_cal = Button(
            text="CLEAR CAL",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1),
            size_hint=(0.25, 1)
        )
        btn_clear_cal.bind(on_release=self.on_clear_cal)

        btn_clear_cloud = Button(
            text="CLEAR CLOUD",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1),
            size_hint=(0.25, 1)
        )
        btn_clear_cloud.bind(on_release=self.on_clear)

        top3.add_widget(btn_cap_cal)
        top3.add_widget(btn_solve_cal)
        top3.add_widget(btn_clear_cal)
        top3.add_widget(btn_clear_cloud)
        root.add_widget(top3)

        # -----------------------------
        # BOTTOM BAR
        # -----------------------------
        bar = BoxLayout(
            orientation="horizontal",
            size_hint=(1, 0.12),
            pos_hint={"x": 0, "y": 0},
            padding=8,
            spacing=8
        )

        self.btn_calib = Button(
            text="CALIB ADD",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1)
        )
        self.btn_calib.bind(on_release=self.on_calib)

        self.btn_view = Button(
            text="VIEW MODEL",
            background_normal="",
            background_color=(0, 0, 0, 0.72),
            color=(1, 0, 0, 1)
        )
        self.btn_view.bind(on_release=self.on_view)

        bar.add_widget(self.btn_calib)
        bar.add_widget(self.btn_view)
        root.add_widget(bar)

        self.viewer_overlay = None
        self.viewer_widget = None
        self.splat_slider = None
        self.splat_label = None
        self.mode_toggle = None
        self.shape_toggle = None

        self.root = root
        return root

    def on_overlay_toggle(self, *_):
        enabled = (self.tog_overlay.state == "down")
        self.tog_overlay.text = "LIVE: ON" if enabled else "LIVE: OFF"
        self.cam.set_live_overlay(enabled)

    def on_undist_toggle(self, *_):
        enabled = (self.tog_undist.state == "down")
        self.cam.set_use_undistort(enabled)
        self.tog_undist.state = "down" if self.cam.use_undistort else "normal"
        self.tog_undist.text = "UNDIST: ON" if self.cam.use_undistort else "UNDIST: OFF"

    def on_plane_toggle(self, *_):
        enabled = (self.tog_plane.state == "down")
        self.cam.set_plane_snap(enabled)
        self.tog_plane.text = "PLANE: ON" if enabled else "PLANE: OFF"

    def on_density_change(self, _, value):
        v = int(value)
        self.lbl_density.text = f"POINTS: {v}"
        self.cam.set_density(v)

    def on_capture_cal(self, *_):
        self.cam.capture_calibration_frame()

    def on_solve_cal(self, *_):
        self.cam.solve_camera_calibration()
        self.tog_undist.state = "down" if self.cam.use_undistort else "normal"
        self.tog_undist.text = "UNDIST: ON" if self.cam.use_undistort else "UNDIST: OFF"

    def on_clear_cal(self, *_):
        self.cam.clear_calibration_samples()

    def on_clear(self, *_):
        self.acc_points = np.zeros((0, 3), dtype=np.float32)
        self.acc_colors = np.zeros((0, 3), dtype=np.float32)
        self.acc_weights = np.zeros((0,), dtype=np.float32)
        self.cam.last_quality = "MODEL: CLEARED"
        save_cloud(self.acc_points, self.acc_colors, self.acc_weights, meta={"quality": "CLEARED"})
        print("ACCUM MODEL CLEARED")

    def on_calib(self, *_):
        if self.cam.calibrating:
            print("Finishing early...")
            cloud, cols, weights, quality = self.cam.finish_splat(np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8))
            self._add_cloud(cloud, cols, weights, quality)
            return

        ok = self.cam.start_calib_splat()
        if not ok:
            print("CALIB: could not start")

    def _add_cloud(self, cloud, cols, weights, quality):
        if cloud is None or cols is None or weights is None or len(cloud) == 0:
            self.cam.last_quality = "MODEL: " + str(quality)
            return

        pts = cloud.astype(np.float32)
        c = cols.astype(np.float32)
        w = weights.astype(np.float32).reshape(-1)

        n = min(len(pts), len(c), len(w))
        pts = pts[:n]
        c = c[:n]
        w = w[:n]

        self.acc_points = np.concatenate([self.acc_points, pts], axis=0) if len(self.acc_points) else pts
        self.acc_colors = np.concatenate([self.acc_colors, c], axis=0) if len(self.acc_colors) else c
        self.acc_weights = np.concatenate([self.acc_weights, w], axis=0) if len(self.acc_weights) else w

        self.acc_points = _apply_plane_snap(self.acc_points, enabled=bool(self.cam.plane_snap_enabled))

        self.acc_points, self.acc_colors, self.acc_weights = voxel_merge(
            self.acc_points,
            self.acc_colors,
            voxel_size=float(VOXEL_SIZE),
            max_points=int(ACC_MAX_POINTS),
            weights=self.acc_weights
        )

        self.cam.last_quality = f"MODEL: ADDED ({quality})  total:{len(self.acc_points)}"
        print("ACCUM ADD:", quality, "total:", len(self.acc_points))

        save_cloud(
            self.acc_points,
            self.acc_colors,
            self.acc_weights,
            meta={
                "quality": f"ACCUM total:{len(self.acc_points)}",
                "voxel": float(VOXEL_SIZE),
                "max_points": int(ACC_MAX_POINTS),
                "mode": "HYBRID_V3P1_CAL_INTRINSICS_TRIANGLE_WEIGHTED"
            }
        )

    def on_view(self, *_):
        if self.acc_points is None or self.acc_colors is None or len(self.acc_points) == 0:
            print("VIEW: no accumulated model yet. Do CALIB ADD.")
            return
        self.open_viewer(self.acc_points, self.acc_colors, quality_text=f"ACCUM total:{len(self.acc_points)}")

    def open_viewer(self, points_3d, colors_bgr, quality_text="MODEL"):
        self.close_viewer()

        overlay = FloatLayout()

        dim = Button(background_normal="", background_color=(0, 0, 0, 0.85), size_hint=(1, 1))
        dim.bind(on_release=lambda *_: self.close_viewer())
        overlay.add_widget(dim)

        self.viewer_widget = SplatViewer(points_3d=points_3d, colors_bgr=colors_bgr, quality_text=quality_text)
        overlay.add_widget(self.viewer_widget)

        panel = BoxLayout(
            orientation="vertical",
            size_hint=(0.92, 0.28),
            pos_hint={"center_x": 0.5, "y": 0.02},
            padding=8,
            spacing=6
        )

        row1 = BoxLayout(orientation="horizontal", size_hint=(1, 0.42), spacing=8)

        self.mode_toggle = ToggleButton(
            text="MODE: SPLAT",
            state="normal",
            background_normal="",
            background_color=(0, 0, 0, 0.7),
            color=(1, 0, 0, 1),
            size_hint=(0.30, 1)
        )
        self.mode_toggle.bind(on_release=self.on_mode_toggle)

        self.shape_toggle = ToggleButton(
            text="SHAPE: TRI",
            state="down",
            background_normal="",
            background_color=(0, 0, 0, 0.7),
            color=(1, 0, 0, 1),
            size_hint=(0.26, 1)
        )
        self.shape_toggle.bind(on_release=self.on_shape_toggle)

        self.splat_label = Label(
            text=f"SIZE: {DEFAULT_SPLAT_RADIUS:.1f}",
            color=(1, 0, 0, 1),
            size_hint=(0.18, 1)
        )
        self.splat_slider = Slider(min=1.0, max=12.0, value=DEFAULT_SPLAT_RADIUS, size_hint=(0.26, 1))
        self.splat_slider.bind(value=self.on_splat_slider)

        row1.add_widget(self.mode_toggle)
        row1.add_widget(self.shape_toggle)
        row1.add_widget(self.splat_label)
        row1.add_widget(self.splat_slider)

        row2 = BoxLayout(orientation="horizontal", size_hint=(1, 0.58), spacing=8)
        hint = Label(text="Drag rotate | SHAPE toggles TRI/CUBE | ROOM = red map", color=(1, 0, 0, 1))
        btn_close = Button(text="CLOSE", background_normal="", background_color=(0, 0, 0, 0.7), color=(1, 0, 0, 1))
        btn_close.bind(on_release=lambda *_: self.close_viewer())
        row2.add_widget(hint)
        row2.add_widget(btn_close)

        panel.add_widget(row1)
        panel.add_widget(row2)
        overlay.add_widget(panel)

        self.viewer_overlay = overlay
        self.root.add_widget(overlay)
        self.viewer_widget.set_shape_mode("TRI")

    def on_mode_toggle(self, *_):
        if not self.viewer_widget or not self.mode_toggle:
            return
        if self.mode_toggle.state == "down":
            self.mode_toggle.text = "MODE: ROOM"
            self.viewer_widget.set_mode("ROOM")
        else:
            self.mode_toggle.text = "MODE: SPLAT"
            self.viewer_widget.set_mode("SPLAT")

    def on_shape_toggle(self, *_):
        if not self.viewer_widget or not self.shape_toggle:
            return
        if self.shape_toggle.state == "down":
            self.shape_toggle.text = "SHAPE: TRI"
            self.viewer_widget.set_shape_mode("TRI")
        else:
            self.shape_toggle.text = "SHAPE: CUBE"
            self.viewer_widget.set_shape_mode("CUBE")

    def on_splat_slider(self, _, value):
        if self.splat_label:
            self.splat_label.text = f"SIZE: {float(value):.1f}"
        if self.viewer_widget:
            self.viewer_widget.set_splat_radius(float(value))

    def close_viewer(self):
        if self.viewer_widget is not None:
            self.viewer_widget.stop()
            self.viewer_widget = None
        if self.viewer_overlay is not None:
            try:
                self.root.remove_widget(self.viewer_overlay)
            except Exception:
                pass
            self.viewer_overlay = None
        self.splat_slider = None
        self.splat_label = None
        self.mode_toggle = None
        self.shape_toggle = None

    def on_stop(self):
        try:
            if self.cam:
                self.cam.release()
        except Exception:
            pass


if __name__ == "__main__":
    SplatApp().run()
