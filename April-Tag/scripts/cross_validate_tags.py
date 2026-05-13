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

from detect_tags import get_frames, load_config, make_detector, start_pipeline
from validate_geometry import (
    camera_matrix_from_intrinsics,
    camera_params_from_intrinsics,
    load_best_corner_permutation,
    solve_floor_pose_from_detections,
    summarize,
    transform_point,
)

import calibrate_pose as pose_calib


def draw_cross_validation(
    color_img: np.ndarray,
    detections: list[Any],
    floor_ids: set[int],
    wall_ids: set[int],
    wall_centers: dict[int, np.ndarray],
) -> np.ndarray:
    out = color_img.copy()
    centers_px: dict[int, tuple[int, int]] = {}

    for det in detections:
        tag_id = int(det.tag_id)
        corners = np.asarray(det.corners, dtype=np.int32)
        center = tuple(np.asarray(det.center, dtype=np.int32))
        centers_px[tag_id] = (int(center[0]), int(center[1]))

        color = (0, 255, 255)
        role = "extra"
        if tag_id in floor_ids:
            color = (0, 255, 0)
            role = "floor"
        elif tag_id in wall_ids:
            color = (255, 0, 255)
            role = "wall"

        cv2.polylines(out, [corners], True, color, 3)
        cv2.circle(out, center, 5, (0, 0, 255), -1)

        label = f"id={tag_id} {role}"
        if tag_id in wall_centers:
            label += f" z={wall_centers[tag_id][2]:.3f}m"
        cv2.putText(
            out,
            label,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    wall_ids_sorted = sorted(wall_ids)
    if len(wall_ids_sorted) == 2:
        a, b = wall_ids_sorted
        if a in centers_px and b in centers_px:
            cv2.line(out, centers_px[a], centers_px[b], (255, 255, 0), 2, cv2.LINE_AA)

    return out


def wall_centers_in_world(
    detections: list[Any],
    wall_ids: set[int],
    T_WC: np.ndarray,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for det in detections:
        tag_id = int(det.tag_id)
        if tag_id not in wall_ids or not hasattr(det, "pose_t"):
            continue
        out[tag_id] = transform_point(T_WC, np.asarray(det.pose_t, dtype=np.float64).reshape(3))[0]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validate four AprilTags: solve world pose from floor tags and validate "
            "wall tag center heights plus wall pair center distance."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/d455.yaml"))
    parser.add_argument("--layout", type=Path, default=Path("configs/tag_layout.yaml"))
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--wall-tag-ids", type=int, nargs=2, default=[0, 3])
    parser.add_argument("--wall-center-height-m", type=float, default=1.6315)
    parser.add_argument("--wall-center-distance-m", type=float, default=0.661)
    parser.add_argument("--save-dir", type=Path, default=Path("output/cross_validation"))
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
    wall_ids = {int(x) for x in args.wall_tag_ids}
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
            color_img, _depth_img, color_intr, _depth_intr = get_frames(pipeline, align)
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

            wall_centers = wall_centers_in_world(detections, wall_ids, T_WC)
            detected_ids = sorted(int(det.tag_id) for det in detections)
            row: dict[str, Any] = {
                "frame_index": frame_index,
                "detected_ids": ",".join(str(x) for x in detected_ids),
                "used_floor_ids": ",".join(str(x) for x in used_ids),
                "mean_reproj_px": mean_err,
                "max_reproj_px": max_err,
                "camera_height_m": camera_height_m,
                "wall_pair_distance_m": None,
                "wall_pair_distance_error_m": None,
            }

            for wall_id in sorted(wall_ids):
                center = wall_centers.get(wall_id)
                if center is None:
                    row[f"wall_{wall_id}_center_x_m"] = None
                    row[f"wall_{wall_id}_center_y_m"] = None
                    row[f"wall_{wall_id}_center_z_m"] = None
                    row[f"wall_{wall_id}_height_error_m"] = None
                    continue
                row[f"wall_{wall_id}_center_x_m"] = float(center[0])
                row[f"wall_{wall_id}_center_y_m"] = float(center[1])
                row[f"wall_{wall_id}_center_z_m"] = float(center[2])
                row[f"wall_{wall_id}_height_error_m"] = float(
                    center[2] - args.wall_center_height_m
                )

            if wall_ids.issubset(wall_centers.keys()):
                a, b = sorted(wall_ids)
                pair_distance = float(np.linalg.norm(wall_centers[a] - wall_centers[b]))
                row["wall_pair_distance_m"] = pair_distance
                row["wall_pair_distance_error_m"] = pair_distance - args.wall_center_distance_m

            rows.append(row)
            latest_annotated = draw_cross_validation(
                color_img,
                detections,
                floor_ids,
                wall_ids,
                wall_centers,
            )

            wall_summary = " ".join(
                f"id{wall_id}_z={wall_centers[wall_id][2]:.3f}m"
                for wall_id in sorted(wall_centers)
            )
            pair_summary = (
                "pair=None"
                if row["wall_pair_distance_m"] is None
                else f"pair={row['wall_pair_distance_m']:.3f}m"
            )
            print(
                "frame={:04d} ids={} reproj={:.3f}px cam_h={:.3f}m {} {}".format(
                    frame_index,
                    detected_ids,
                    mean_err,
                    camera_height_m,
                    wall_summary,
                    pair_summary,
                ),
                flush=True,
            )
    finally:
        pipeline.stop()

    if not rows:
        raise SystemExit("No valid frames were collected.")

    with (save_dir / "cross_validation_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if latest_annotated is not None:
        cv2.imwrite(str(save_dir / "latest_cross_validation_annotated.png"), latest_annotated)

    camera_heights = np.asarray([float(row["camera_height_m"]) for row in rows], dtype=np.float64)
    reproj = np.asarray([float(row["mean_reproj_px"]) for row in rows], dtype=np.float64)
    pair_distances = np.asarray(
        [float(row["wall_pair_distance_m"]) for row in rows if row["wall_pair_distance_m"] is not None],
        dtype=np.float64,
    )
    pair_errors = np.asarray(
        [
            float(row["wall_pair_distance_error_m"])
            for row in rows
            if row["wall_pair_distance_error_m"] is not None
        ],
        dtype=np.float64,
    )

    wall_summaries: dict[str, Any] = {}
    for wall_id in sorted(wall_ids):
        heights = np.asarray(
            [
                float(row[f"wall_{wall_id}_center_z_m"])
                for row in rows
                if row[f"wall_{wall_id}_center_z_m"] is not None
            ],
            dtype=np.float64,
        )
        errors = np.asarray(
            [
                float(row[f"wall_{wall_id}_height_error_m"])
                for row in rows
                if row[f"wall_{wall_id}_height_error_m"] is not None
            ],
            dtype=np.float64,
        )
        wall_summaries[str(wall_id)] = {
            "center_z_m": summarize(heights),
            "height_error_m": summarize(errors),
        }

    summary = {
        "frames_used": len(rows),
        "corner_permutation": corner_name,
        "floor_tag_ids": sorted(floor_ids),
        "wall_tag_ids": sorted(wall_ids),
        "expected_wall_center_height_m": args.wall_center_height_m,
        "expected_wall_pair_center_distance_m": args.wall_center_distance_m,
        "mean_reprojection_px": summarize(reproj),
        "camera_height_m": summarize(camera_heights),
        "wall_tags": wall_summaries,
        "wall_pair_distance_m": summarize(pair_distances),
        "wall_pair_distance_error_m": summarize(pair_errors),
        "depth_scale_m_per_unit": depth_scale,
    }
    (save_dir / "cross_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nCross validation summary")
    print(f"  frames_used: {len(rows)}")
    print(f"  detected_wall_ids: {sorted(wall_ids)}")
    print(f"  median_reprojection_px: {np.median(reproj):.3f}")
    print(f"  median_camera_height_m: {np.median(camera_heights):.3f}")
    for wall_id in sorted(wall_ids):
        wall_z = wall_summaries[str(wall_id)]["center_z_m"]["median"]
        wall_error = wall_summaries[str(wall_id)]["height_error_m"]["median"]
        if wall_z is not None:
            print(f"  wall_id{wall_id}_median_center_z_m: {wall_z:.3f}")
            print(f"  wall_id{wall_id}_median_height_error_m: {wall_error:.3f}")
    if pair_distances.size:
        print(f"  median_wall_pair_distance_m: {np.median(pair_distances):.3f}")
        print(f"  median_wall_pair_distance_error_m: {np.median(pair_errors):.3f}")
    print(f"  saved_dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
