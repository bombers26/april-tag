from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_pose as pose_calib
from detect_tags import get_frames, load_config, make_detector, start_pipeline
from validate_geometry import (
    camera_matrix_from_intrinsics,
    camera_params_from_intrinsics,
    load_best_corner_permutation,
    solve_floor_pose_from_detections,
)


def inflate_polygon(corners: np.ndarray, scale: float = 1.8) -> np.ndarray:
    center = np.mean(corners, axis=0, keepdims=True)
    return center + (corners - center) * scale


def mask_tags(shape: tuple[int, int], detections: list[Any]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        poly = inflate_polygon(np.asarray(det.corners, dtype=np.float64), scale=2.2)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
    return mask


def order_polygon_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 points, got shape={pts.shape}")

    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]

    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)

    cross = np.cross(pts[1] - pts[0], pts[2] - pts[1])
    if cross < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float64)
    return pts


def approximate_quadrilateral(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0.0:
        return None

    candidates = [hull, contour]
    for source in candidates:
        for frac in (0.008, 0.012, 0.016, 0.022, 0.030, 0.040, 0.055):
            approx = cv2.approxPolyDP(source, frac * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float64)
                return order_polygon_points(quad)

    rect = cv2.minAreaRect(contour)
    quad = cv2.boxPoints(rect).astype(np.float64)
    return order_polygon_points(quad)


def wall_plane_from_tag(
    wall_detection: Any,
    T_WC: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_C = np.asarray(wall_detection.pose_t, dtype=np.float64).reshape(3)
    n_C = np.asarray(wall_detection.pose_R, dtype=np.float64)[:, 2]
    R_WC = T_WC[:3, :3]
    p_W = (p_C.reshape(1, 3) @ R_WC.T + T_WC[:3, 3])[0]
    n_W = R_WC @ n_C
    n_W = n_W / np.linalg.norm(n_W)
    return p_W, n_W


def load_world_from_camera_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    matrix = summary.get("last_T_world_from_camera")
    if matrix is None:
        return None
    return summary


def world_transform_from_summary(summary: dict[str, Any]) -> np.ndarray:
    return np.asarray(summary["last_T_world_from_camera"], dtype=np.float64)


def intersect_pixels_with_plane(
    pixel_uv: np.ndarray,
    intr: Any,
    T_WC: np.ndarray,
    plane_point_W: np.ndarray,
    plane_normal_W: np.ndarray,
) -> np.ndarray:
    if len(pixel_uv) == 0:
        return np.empty((0, 3), dtype=np.float64)

    uv = np.asarray(pixel_uv, dtype=np.float64)
    rays_C = np.column_stack(
        [
            (uv[:, 0] - float(intr.ppx)) / float(intr.fx),
            (uv[:, 1] - float(intr.ppy)) / float(intr.fy),
            np.ones(len(uv), dtype=np.float64),
        ]
    )
    R_WC = T_WC[:3, :3]
    origin_W = T_WC[:3, 3]
    rays_W = rays_C @ R_WC.T
    denom = rays_W @ plane_normal_W
    valid = np.abs(denom) > 1e-8
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    rays_W = rays_W[valid]
    denom = denom[valid]
    scale = ((plane_point_W - origin_W) @ plane_normal_W) / denom
    valid_scale = scale > 0.0
    if not np.any(valid_scale):
        return np.empty((0, 3), dtype=np.float64)

    return origin_W.reshape(1, 3) + rays_W[valid_scale] * scale[valid_scale].reshape(-1, 1)


def quadrilateral_pixels(contour: np.ndarray) -> np.ndarray:
    quad = approximate_quadrilateral(contour)
    if quad is None:
        raise RuntimeError("Could not approximate a quadrilateral from contour")
    return quad


def quadrilateral_world_stats(
    contour: np.ndarray,
    intr: Any,
    T_WC: np.ndarray,
    plane_point_W: np.ndarray,
    plane_normal_W: np.ndarray,
) -> dict[str, Any]:
    quad_px = quadrilateral_pixels(contour)
    quad_W = intersect_pixels_with_plane(quad_px, intr, T_WC, plane_point_W, plane_normal_W)
    if len(quad_W) != 4:
        return {"valid": False}

    z = quad_W[:, 2]
    top_idx = np.argsort(z)[-2:]
    bottom_idx = np.argsort(z)[:2]
    top_edge_px = quad_px[top_idx]
    bottom_edge_px = quad_px[bottom_idx]
    return {
        "valid": True,
        "top_z_m": float(np.mean(z[top_idx])),
        "top_z_min_m": float(np.min(z[top_idx])),
        "top_z_max_m": float(np.max(z[top_idx])),
        "bottom_z_m": float(np.mean(z[bottom_idx])),
        "bottom_z_min_m": float(np.min(z[bottom_idx])),
        "bottom_z_max_m": float(np.max(z[bottom_idx])),
        "height_m": float(np.mean(z[top_idx]) - np.mean(z[bottom_idx])),
        "center_W": np.mean(quad_W, axis=0).tolist(),
        "quad_px": quad_px.tolist(),
        "quad_world": quad_W.tolist(),
        "top_edge_px": top_edge_px.tolist(),
        "bottom_edge_px": bottom_edge_px.tolist(),
    }


def contour_world_stats(
    contour: np.ndarray,
    intr: Any,
    T_WC: np.ndarray,
    plane_point_W: np.ndarray,
    plane_normal_W: np.ndarray,
) -> dict[str, Any]:
    return quadrilateral_world_stats(contour, intr, T_WC, plane_point_W, plane_normal_W)


def candidate_contours(color_bgr: np.ndarray, detections: list[Any]) -> list[np.ndarray]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

    # White paper on a gray wall: low saturation and brighter than local background.
    mask_color = cv2.inRange(hsv, (0, 0, 105), (179, 95, 255))
    tag_mask = mask_tags(gray.shape, detections)
    mask_color[tag_mask > 0] = 0

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 350 or area > 25000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 18 or h < 18:
            continue
        aspect = w / max(float(h), 1.0)
        if aspect < 0.35 or aspect > 3.2:
            continue

        # Reject long ceiling lights and border strips.
        rect_area = float(w * h)
        fill = area / max(rect_area, 1.0)
        if fill < 0.25:
            continue

        filtered.append(contour)
    return filtered


def edge_contours_above_wall_tag(
    color_bgr: np.ndarray,
    detections: list[Any],
    wall_center_px: np.ndarray,
    left_px: int = 120,
    right_px: int = 180,
    up_px: int = 260,
    down_px: int = 80,
) -> list[np.ndarray]:
    h_img, w_img = color_bgr.shape[:2]
    cx, cy = wall_center_px.astype(int)
    x1 = max(0, cx - left_px)
    x2 = min(w_img, cx + right_px)
    y1 = max(0, cy - up_px)
    y2 = min(h_img, cy - down_px)
    if x2 <= x1 or y2 <= y1:
        return []

    roi = color_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < 400 or area > 18000:
            continue
        if w < 45 or h < 35:
            continue
        aspect = w / max(float(h), 1.0)
        if aspect < 0.6 or aspect > 2.2:
            continue

        contour_full = contour.copy()
        contour_full[:, 0, 0] += x1
        contour_full[:, 0, 1] += y1

        # Reject accidental tag contours after offsetting to full-image coordinates.
        tag_mask = mask_tags(color_bgr.shape[:2], detections)
        test_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(test_mask, [contour_full], -1, 255, thickness=cv2.FILLED)
        overlap = np.count_nonzero((test_mask > 0) & (tag_mask > 0))
        if overlap / max(float(np.count_nonzero(test_mask)), 1.0) > 0.2:
            continue

        filtered.append(contour_full)
    return filtered


def choose_picture_candidate(
    contours: list[np.ndarray],
    color_intr: Any,
    T_WC: np.ndarray,
    plane_point_W: np.ndarray,
    plane_normal_W: np.ndarray,
    wall_center_px: np.ndarray | None,
    expected_top_height_m: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    scored: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for contour in contours:
        stats = contour_world_stats(contour, color_intr, T_WC, plane_point_W, plane_normal_W)
        if not stats.get("valid"):
            continue
        top_z = float(stats["top_z_m"])
        if top_z < 0.8 or top_z > 2.4:
            continue

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        center_px = np.array([x + w / 2.0, y + h / 2.0])

        score = 0.0
        score += min(area / 1500.0, 4.0)
        if wall_center_px is not None:
            dist = float(np.linalg.norm(center_px - wall_center_px))
            score -= dist / 180.0
        if expected_top_height_m is not None:
            score -= abs(top_z - expected_top_height_m) * 90.0
        # Prefer candidates above the floor tags and away from image borders.
        if x < 20 or y < 20 or x + w > 1260 or y + h > 700:
            score -= 4.0

        stats.update(
            {
                "area_px": float(area),
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "center_px": center_px.tolist(),
                "score": float(score),
            }
        )
        scored.append((score, contour, stats))

    if not scored:
        raise RuntimeError("No wall picture candidate was detected.")

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[0][2]


def draw_measurement(
    color_bgr: np.ndarray,
    detections: list[Any],
    picture_contour: np.ndarray,
    stats: dict[str, Any],
) -> np.ndarray:
    out = color_bgr.copy()
    for det in detections:
        tag_id = int(det.tag_id)
        color = (0, 255, 0) if tag_id in {1, 2} else (255, 0, 255)
        cv2.polylines(out, [np.asarray(det.corners, dtype=np.int32)], True, color, 3)
        center = tuple(np.asarray(det.center, dtype=np.int32))
        cv2.putText(out, f"id={tag_id}", (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    quad = np.asarray(stats.get("quad_px", quadrilateral_pixels(picture_contour)), dtype=np.int32)
    top_edge = np.asarray(stats.get("top_edge_px", []), dtype=np.int32)
    cv2.polylines(out, [quad], True, (0, 0, 255), 3)
    quad_world = np.asarray(stats.get("quad_world", []), dtype=np.float64)
    for i, p in enumerate(quad):
        cv2.circle(out, tuple(p), 5, (255, 255, 0), -1)
        if len(quad_world) == 4:
            cv2.putText(
                out,
                f"{quad_world[i, 2]:.3f}",
                (int(p[0]) + 5, int(p[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
    if len(top_edge) == 2:
        cv2.line(out, tuple(top_edge[0]), tuple(top_edge[1]), (255, 0, 0), 4)
    x, y, w, h = stats["bbox_xywh"]
    label = f"top={stats['top_z_m']:.3f}m"
    cv2.putText(out, label, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the top-edge height of a white wall picture using AprilTag calibration.")
    parser.add_argument("--config", type=Path, default=Path("configs/d455.yaml"))
    parser.add_argument("--layout", type=Path, default=Path("configs/tag_layout.yaml"))
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--wall-tag-id", type=int, default=0)
    parser.add_argument("--expected-top-height-m", type=float, default=None)
    parser.add_argument("--pose-summary", type=Path, default=Path("output/calibration/pose_summary.json"))
    parser.add_argument("--save-dir", type=Path, default=Path("output/picture_height"))
    parser.add_argument("--corner-summary", type=Path, default=Path("output/calibration/pose_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    layout = pose_calib.load_layout(args.layout)
    corner_name = load_best_corner_permutation(args.corner_summary)
    corner_perm = pose_calib.PERMUTATIONS[corner_name]
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    detector = make_detector(cfg.apriltag.family)
    pipeline, align, depth_scale = start_pipeline(cfg.camera)
    pose_summary = load_world_from_camera_summary(args.pose_summary)

    rows: list[dict[str, Any]] = []
    latest_image: np.ndarray | None = None
    try:
        for _ in range(cfg.capture.warmup_frames):
            get_frames(pipeline, align)

        for frame_index in range(args.frames):
            color_img, _depth_img, color_intr, _depth_intr = get_frames(pipeline, align)
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            camera_params = camera_params_from_intrinsics(color_intr)
            detections = list(
                detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=layout.tag_size_m,
                )
            )
            K = camera_matrix_from_intrinsics(color_intr)
            try:
                T_WC, _rvec, mean_err, max_err, camera_height_m, used_ids = solve_floor_pose_from_detections(
                    detections,
                    layout,
                    corner_perm,
                    K,
                )
            except RuntimeError as exc:
                if pose_summary is None:
                    print(f"frame={frame_index:04d} skipped: {exc}", flush=True)
                    continue
                T_WC = world_transform_from_summary(pose_summary)
                mean_err = float(pose_summary.get("median_mean_reprojection_px", 0.0))
                max_err = float(pose_summary.get("median_max_reprojection_px", mean_err))
                camera_height_m = float(pose_summary.get("median_camera_height_m", T_WC[2, 3]))
                used_ids = list(pose_summary.get("floor_tag_ids", []))
            wall_det = next((det for det in detections if int(det.tag_id) == args.wall_tag_id), None)
            if wall_det is None:
                print(f"frame={frame_index:04d} skipped: wall tag {args.wall_tag_id} not detected", flush=True)
                continue

            plane_point_W, plane_normal_W = wall_plane_from_tag(wall_det, T_WC)
            contours = edge_contours_above_wall_tag(
                color_img,
                detections,
                np.asarray(wall_det.center, dtype=np.float64),
            )
            if not contours:
                contours = candidate_contours(color_img, detections)
            try:
                contour, stats = choose_picture_candidate(
                    contours,
                    color_intr,
                    T_WC,
                    plane_point_W,
                    plane_normal_W,
                    np.asarray(wall_det.center, dtype=np.float64),
                    args.expected_top_height_m,
                )
            except RuntimeError as exc:
                print(f"frame={frame_index:04d} skipped: {exc}", flush=True)
                continue

            row = {
                "frame_index": frame_index,
                "detected_ids": ",".join(str(int(det.tag_id)) for det in detections),
                "used_floor_ids": ",".join(str(x) for x in used_ids),
                "mean_reproj_px": mean_err,
                "max_reproj_px": max_err,
                "camera_height_m": camera_height_m,
                "picture_top_z_m": float(stats["top_z_m"]),
                "picture_bottom_z_m": float(stats["bottom_z_m"]),
                "picture_height_m": float(stats["height_m"]),
                "candidate_area_px": float(stats["area_px"]),
                "candidate_score": float(stats["score"]),
                "candidate_bbox_xywh": json.dumps(stats["bbox_xywh"]),
            }
            if args.expected_top_height_m is not None:
                row["top_error_m"] = float(stats["top_z_m"]) - args.expected_top_height_m
            rows.append(row)
            latest_image = draw_measurement(color_img, detections, contour, stats)
            print(
                f"frame={frame_index:04d} top={stats['top_z_m']:.3f}m "
                f"bottom={stats['bottom_z_m']:.3f}m reproj={mean_err:.3f}px",
                flush=True,
            )
    finally:
        pipeline.stop()

    if not rows:
        raise SystemExit("No valid picture height measurements were collected.")

    with (save_dir / "picture_height_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if latest_image is not None:
        cv2.imwrite(str(save_dir / "latest_picture_height_annotated.png"), latest_image)

    top_values = np.asarray([float(row["picture_top_z_m"]) for row in rows], dtype=np.float64)
    bottom_values = np.asarray([float(row["picture_bottom_z_m"]) for row in rows], dtype=np.float64)
    reproj_values = np.asarray([float(row["mean_reproj_px"]) for row in rows], dtype=np.float64)
    summary = {
        "frames_used": len(rows),
        "median_picture_top_z_m": float(np.median(top_values)),
        "mean_picture_top_z_m": float(np.mean(top_values)),
        "std_picture_top_z_m": float(np.std(top_values)),
        "median_picture_bottom_z_m": float(np.median(bottom_values)),
        "median_picture_height_m": float(np.median(top_values - bottom_values)),
        "median_reprojection_px": float(np.median(reproj_values)),
        "expected_top_height_m": args.expected_top_height_m,
        "median_top_error_m": (
            float(np.median(top_values) - args.expected_top_height_m)
            if args.expected_top_height_m is not None
            else None
        ),
        "method": "Configured floor tags define world Z=0; configured wall tag defines wall plane; picture contour pixels are intersected with that wall plane.",
        "depth_scale_m_per_unit": depth_scale,
    }
    (save_dir / "picture_height_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nPicture height summary")
    print(f"  frames_used: {len(rows)}")
    print(f"  median_picture_top_z_m: {np.median(top_values):.3f}")
    print(f"  std_picture_top_z_m: {np.std(top_values):.4f}")
    print(f"  median_picture_bottom_z_m: {np.median(bottom_values):.3f}")
    print(f"  median_picture_height_m: {np.median(top_values - bottom_values):.3f}")
    if args.expected_top_height_m is not None:
        print(f"  median_top_error_m: {np.median(top_values) - args.expected_top_height_m:.3f}")
    print(f"  saved_dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
