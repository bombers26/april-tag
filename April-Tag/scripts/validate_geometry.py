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


def load_best_corner_permutation(path: Path, fallback: str = "3210") -> str:
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("best_corner_permutation", fallback))


def camera_params_from_intrinsics(intr: Any) -> tuple[float, float, float, float]:
    return float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)


def camera_matrix_from_intrinsics(intr: Any) -> np.ndarray:
    return np.array(
        [
            [float(intr.fx), 0.0, float(intr.ppx)],
            [0.0, float(intr.fy), float(intr.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def solve_floor_pose_from_detections(
    detections: list[Any],
    layout: pose_calib.Layout,
    corner_permutation: list[int],
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, list[int]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_ids: list[int] = []
    seen_floor_ids: set[int] = set()

    for det in detections:
        tag_id = int(det.tag_id)
        tag = layout.floor_tags.get(tag_id)
        if tag is None:
            continue
        if tag_id in seen_floor_ids:
            raise RuntimeError(
                f"Duplicate configured floor tag id {tag_id}; "
                "cover the duplicate tag before solving camera pose."
            )
        seen_floor_ids.add(tag_id)
        object_points.append(
            pose_calib.world_corners_for_tag(tag, layout.tag_size_m, corner_permutation)
        )
        image_points.append(np.asarray(det.corners, dtype=np.float64))
        used_ids.append(tag_id)

    if len(set(used_ids)) < 2:
        raise RuntimeError(f"Need both floor tags; detected floor ids={sorted(set(used_ids))}")

    obj = np.vstack(object_points)
    img = np.vstack(image_points)
    rvec, tvec, mean_err, max_err, camera_height_m = pose_calib.solve_pose(obj, img, K)
    T_WC = pose_calib.invert_T(pose_calib.rt_to_T(rvec, tvec))
    return T_WC, rvec, mean_err, max_err, camera_height_m, sorted(set(used_ids))


def depth_pixels_to_world(
    depth_img: np.ndarray,
    pixel_uv: np.ndarray,
    depth_scale: float,
    intr: Any,
    T_WC: np.ndarray,
) -> np.ndarray:
    u = pixel_uv[:, 0].astype(np.float64)
    v = pixel_uv[:, 1].astype(np.float64)
    z = depth_img[v.astype(np.int32), u.astype(np.int32)].astype(np.float64) * depth_scale
    valid = z > 0.0
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    u = u[valid]
    v = v[valid]
    z = z[valid]
    x = (u - float(intr.ppx)) * z / float(intr.fx)
    y = (v - float(intr.ppy)) * z / float(intr.fy)
    pts_C = np.column_stack([x, y, z])
    return pts_C @ T_WC[:3, :3].T + T_WC[:3, 3]


def polygon_pixels(corners: np.ndarray, image_shape: tuple[int, int], stride: int = 2) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.asarray(corners, dtype=np.int32)
    cv2.fillConvexPoly(mask, poly, 255)
    ys, xs = np.where(mask > 0)
    if stride > 1:
        keep = (xs % stride == 0) & (ys % stride == 0)
        xs = xs[keep]
        ys = ys[keep]
    return np.column_stack([xs, ys])


def floor_tag_z_values(
    depth_img: np.ndarray,
    depth_scale: float,
    intr: Any,
    T_WC: np.ndarray,
    detections: list[Any],
    floor_ids: set[int],
) -> np.ndarray:
    all_points: list[np.ndarray] = []
    for det in detections:
        if int(det.tag_id) not in floor_ids:
            continue
        pixels = polygon_pixels(np.asarray(det.corners), depth_img.shape, stride=2)
        if len(pixels) == 0:
            continue
        pts_W = depth_pixels_to_world(depth_img, pixels, depth_scale, intr, T_WC)
        if len(pts_W) > 0:
            all_points.append(pts_W)

    if not all_points:
        return np.empty((0,), dtype=np.float64)
    return np.vstack(all_points)[:, 2]


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p.reshape(1, 3) @ T[:3, :3].T + T[:3, 3]


def draw_validation(
    color_img: np.ndarray,
    detections: list[Any],
    floor_ids: set[int],
    wall_tag_id: int,
) -> np.ndarray:
    out = color_img.copy()
    for det in detections:
        tag_id = int(det.tag_id)
        color = (0, 255, 0)
        if tag_id == wall_tag_id:
            color = (255, 0, 255)
        elif tag_id not in floor_ids:
            color = (0, 255, 255)
        corners = np.asarray(det.corners, dtype=np.int32)
        cv2.polylines(out, [corners], True, color, 3)
        center = tuple(np.asarray(det.center, dtype=np.int32))
        cv2.circle(out, center, 4, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"id={tag_id}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def summarize(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "std": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate floor-tag world transform with depth points and a wall tag height check.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/d455.yaml"))
    parser.add_argument("--layout", type=Path, default=Path("configs/tag_layout.yaml"))
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--wall-tag-id", type=int, default=0)
    parser.add_argument("--wall-center-height-m", type=float, default=1.6315)
    parser.add_argument("--save-dir", type=Path, default=Path("output/geometry_validation"))
    parser.add_argument(
        "--corner-summary",
        type=Path,
        default=Path("output/calibration/pose_summary.json"),
        help="Use its best_corner_permutation if available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    layout = pose_calib.load_layout(args.layout)
    floor_ids = set(layout.floor_tags.keys())
    corner_name = load_best_corner_permutation(args.corner_summary)
    corner_perm = pose_calib.PERMUTATIONS[corner_name]

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    detector = make_detector(cfg.apriltag.family)
    pipeline, align, depth_scale = start_pipeline(cfg.camera)

    rows: list[dict[str, Any]] = []
    latest_annotated: np.ndarray | None = None

    try:
        for _ in range(cfg.capture.warmup_frames):
            get_frames(pipeline, align)

        for frame_index in range(args.frames):
            color_img, depth_img, color_intr, depth_intr = get_frames(pipeline, align)
            K = camera_matrix_from_intrinsics(color_intr)
            camera_params = camera_params_from_intrinsics(color_intr)
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            detections = list(
                detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=layout.tag_size_m,
                )
            )

            try:
                T_WC, _rvec, mean_err, max_err, camera_height_m, used_ids = (
                    solve_floor_pose_from_detections(detections, layout, corner_perm, K)
                )
            except RuntimeError as exc:
                print(f"frame={frame_index:04d} skipped: {exc}", flush=True)
                continue

            floor_z = floor_tag_z_values(
                depth_img,
                depth_scale,
                depth_intr,
                T_WC,
                detections,
                floor_ids,
            )

            wall_center_z = None
            for det in detections:
                if int(det.tag_id) == args.wall_tag_id and hasattr(det, "pose_t"):
                    wall_center_world = transform_point(T_WC, np.asarray(det.pose_t).reshape(3))[0]
                    wall_center_z = float(wall_center_world[2])
                    break

            row = {
                "frame_index": frame_index,
                "detected_ids": ",".join(str(int(det.tag_id)) for det in detections),
                "used_floor_ids": ",".join(str(x) for x in used_ids),
                "mean_reproj_px": mean_err,
                "max_reproj_px": max_err,
                "camera_height_m": camera_height_m,
                "floor_z_count": int(floor_z.size),
                "floor_z_median_m": float(np.median(floor_z)) if floor_z.size else None,
                "floor_z_p05_m": float(np.percentile(floor_z, 5)) if floor_z.size else None,
                "floor_z_p95_m": float(np.percentile(floor_z, 95)) if floor_z.size else None,
                "wall_center_z_m": wall_center_z,
                "wall_center_error_m": (
                    wall_center_z - args.wall_center_height_m if wall_center_z is not None else None
                ),
            }
            rows.append(row)
            latest_annotated = draw_validation(color_img, detections, floor_ids, args.wall_tag_id)
            print(
                "frame={:04d} ids={} reproj={:.3f}px cam_h={:.3f}m floor_z={} wall_z={}".format(
                    frame_index,
                    row["detected_ids"],
                    mean_err,
                    camera_height_m,
                    None if row["floor_z_median_m"] is None else f"{row['floor_z_median_m']:.3f}m",
                    None if wall_center_z is None else f"{wall_center_z:.3f}m",
                ),
                flush=True,
            )
    finally:
        pipeline.stop()

    if not rows:
        raise SystemExit("No valid frames were collected.")

    with (save_dir / "geometry_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if latest_annotated is not None:
        cv2.imwrite(str(save_dir / "latest_validation_annotated.png"), latest_annotated)

    camera_heights = np.asarray([float(row["camera_height_m"]) for row in rows], dtype=np.float64)
    reproj = np.asarray([float(row["mean_reproj_px"]) for row in rows], dtype=np.float64)
    floor_values = np.asarray(
        [float(row["floor_z_median_m"]) for row in rows if row["floor_z_median_m"] is not None],
        dtype=np.float64,
    )
    wall_errors = np.asarray(
        [float(row["wall_center_error_m"]) for row in rows if row["wall_center_error_m"] is not None],
        dtype=np.float64,
    )
    wall_z = np.asarray(
        [float(row["wall_center_z_m"]) for row in rows if row["wall_center_z_m"] is not None],
        dtype=np.float64,
    )

    summary = {
        "frames_used": len(rows),
        "corner_permutation": corner_name,
        "floor_tag_ids": sorted(floor_ids),
        "wall_tag_id": args.wall_tag_id,
        "expected_wall_center_height_m": args.wall_center_height_m,
        "mean_reprojection_px": summarize(reproj),
        "camera_height_m": summarize(camera_heights),
        "floor_tag_region_z_m": summarize(floor_values),
        "wall_center_z_m": summarize(wall_z),
        "wall_center_error_m": summarize(wall_errors),
        "depth_scale_m_per_unit": depth_scale,
    }
    (save_dir / "geometry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nGeometry validation summary")
    print(f"  frames_used: {len(rows)}")
    print(f"  median_reprojection_px: {np.median(reproj):.3f}")
    print(f"  median_camera_height_m: {np.median(camera_heights):.3f}")
    if floor_values.size:
        print(f"  median_floor_tag_region_z_m: {np.median(floor_values):.3f}")
    if wall_z.size:
        print(f"  median_wall_center_z_m: {np.median(wall_z):.3f}")
        print(f"  median_wall_center_error_m: {np.median(wall_errors):.3f}")
    print(f"  saved_dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
