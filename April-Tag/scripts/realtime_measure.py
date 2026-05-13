from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_pose as pose_calib
from detect_tags import get_frames, load_config, make_detector, start_pipeline
from measure_picture_height import (
    candidate_contours,
    contour_world_stats,
    edge_contours_above_wall_tag,
    load_world_from_camera_summary,
    quadrilateral_pixels,
    wall_plane_from_tag,
    world_transform_from_summary,
)
from validate_geometry import (
    camera_matrix_from_intrinsics,
    camera_params_from_intrinsics,
    load_best_corner_permutation,
    solve_floor_pose_from_detections,
)


def put_label(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
) -> None:
    x, y = xy
    cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_tags(
    image: np.ndarray,
    detections: list[Any],
    floor_ids: set[int],
    wall_tag_id: int,
) -> None:
    for det in detections:
        tag_id = int(det.tag_id)
        if tag_id in floor_ids:
            color = (0, 255, 0)
        elif tag_id == wall_tag_id:
            color = (255, 0, 255)
        else:
            color = (0, 255, 255)
        corners = np.asarray(det.corners, dtype=np.int32)
        cv2.polylines(image, [corners], True, color, 3)
        center = tuple(np.asarray(det.center, dtype=np.int32))
        cv2.circle(image, center, 4, (0, 0, 255), -1)
        put_label(image, f"id={tag_id}", (center[0] + 8, center[1] - 8), color, 0.75)


def detect_wall_targets(
    color_img: np.ndarray,
    detections: list[Any],
    wall_center_px: np.ndarray,
) -> list[np.ndarray]:
    contours = edge_contours_above_wall_tag(
        color_img,
        detections,
        wall_center_px,
    )
    if contours:
        return contours
    return candidate_contours(color_img, detections)


def draw_target_measurements(
    image: np.ndarray,
    contours: list[np.ndarray],
    color_intr: Any,
    T_WC: np.ndarray,
    plane_point_W: np.ndarray,
    plane_normal_W: np.ndarray,
    max_targets: int,
    expected_top_height_m: float | None = None,
) -> list[dict[str, Any]]:
    measured: list[dict[str, Any]] = []
    for contour in contours:
        stats = contour_world_stats(contour, color_intr, T_WC, plane_point_W, plane_normal_W)
        if not stats.get("valid"):
            continue
        top_z = float(stats["top_z_m"])
        bottom_z = float(stats["bottom_z_m"])
        if top_z < 0.6 or top_z > 2.4:
            continue

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < 300:
            continue
        measured.append(
            {
                "contour": contour,
                "top_z_m": top_z,
                "bottom_z_m": bottom_z,
                "height_m": float(top_z - bottom_z),
                "area_px": float(area),
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "quad_px": stats.get("quad_px"),
                "quad_world": stats.get("quad_world"),
                "top_edge_px": stats.get("top_edge_px"),
            }
        )

    if expected_top_height_m is not None:
        measured.sort(
            key=lambda row: (
                abs(row["top_z_m"] - expected_top_height_m),
                -row["area_px"],
                -row["top_z_m"],
            )
        )
    else:
        measured.sort(key=lambda row: (row["area_px"], row["top_z_m"]), reverse=True)
    measured = measured[:max_targets]

    for idx, row in enumerate(measured, start=1):
        contour = row["contour"]
        x, y, w, h = row["bbox_xywh"]
        color = (0, 0, 255)
        quad = np.asarray(row.get("quad_px", quadrilateral_pixels(contour)), dtype=np.int32)
        quad_world = np.asarray(row.get("quad_world", []), dtype=np.float64)
        top_edge = np.asarray(row.get("top_edge_px", []), dtype=np.int32)
        cv2.polylines(image, [quad], True, color, 3)
        for i, p in enumerate(quad):
            cv2.circle(image, tuple(p), 5, (255, 255, 0), -1)
            if len(quad_world) == 4:
                put_label(image, f"{quad_world[i, 2]:.3f}", (int(p[0]) + 5, int(p[1]) + 5), (255, 255, 0), 0.45)
        if len(top_edge) == 2:
            cv2.line(image, tuple(top_edge[0]), tuple(top_edge[1]), (255, 0, 0), 4)
        put_label(
            image,
            f"target{idx} top={row['top_z_m']:.3f}m",
            (x, max(28, y - 10)),
            color,
            0.75,
        )
    return measured


def depth_vis(depth_img: np.ndarray, max_distance_m: float = 3.0) -> np.ndarray:
    scaled = np.clip(depth_img.astype(np.float32) / (max_distance_m * 1000.0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime AprilTag-based wall target height measurement.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/d455.yaml"))
    parser.add_argument("--layout", type=Path, default=Path("configs/tag_layout.yaml"))
    parser.add_argument("--corner-summary", type=Path, default=Path("output/calibration/pose_summary.json"))
    parser.add_argument("--wall-tag-id", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until q is pressed.")
    parser.add_argument("--no-window", action="store_true", help="Do not open OpenCV UI; useful for smoke tests.")
    parser.add_argument("--show-depth", action="store_true", help="Show a depth preview next to the color image.")
    parser.add_argument("--expected-top-height-m", type=float, default=None, help="Optional expected top-edge height for target selection.")
    parser.add_argument("--save-dir", type=Path, default=Path("output/realtime_measure"))
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
    pose_summary = load_world_from_camera_summary(args.corner_summary)

    detector = make_detector(cfg.apriltag.family)
    pipeline, align, depth_scale = start_pipeline(cfg.camera)

    rows: list[dict[str, Any]] = []
    latest_view: np.ndarray | None = None
    frame_index = 0
    last_time = time.time()
    fps = 0.0

    try:
        for _ in range(cfg.capture.warmup_frames):
            get_frames(pipeline, align)

        while True:
            if args.frames and frame_index >= args.frames:
                break

            color_img, depth_img, color_intr, _depth_intr = get_frames(pipeline, align)
            now = time.time()
            dt = max(now - last_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            last_time = now

            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            detections = list(
                detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=camera_params_from_intrinsics(color_intr),
                    tag_size=layout.tag_size_m,
                )
            )

            view = color_img.copy()
            draw_tags(view, detections, floor_ids, args.wall_tag_id)

            status = "OK"
            measured_targets: list[dict[str, Any]] = []
            mean_err = None
            camera_height = None
            try:
                K = camera_matrix_from_intrinsics(color_intr)
                try:
                    T_WC, _rvec, mean_err, _max_err, camera_height, _used_ids = solve_floor_pose_from_detections(
                        detections,
                        layout,
                        corner_perm,
                        K,
                    )
                except RuntimeError as exc:
                    if pose_summary is None:
                        raise
                    status = f"pose fallback: {exc}"
                    T_WC = world_transform_from_summary(pose_summary)
                    mean_err = float(pose_summary.get("median_mean_reprojection_px", 0.0))
                    camera_height = float(pose_summary.get("median_camera_height_m", T_WC[2, 3]))
                wall_det = next((det for det in detections if int(det.tag_id) == args.wall_tag_id), None)
                if wall_det is None:
                    status = f"wall tag {args.wall_tag_id} missing"
                else:
                    plane_point_W, plane_normal_W = wall_plane_from_tag(wall_det, T_WC)
                    contours = detect_wall_targets(
                        color_img,
                        detections,
                        np.asarray(wall_det.center, dtype=np.float64),
                    )
                    measured_targets = draw_target_measurements(
                        view,
                        contours,
                        color_intr,
                        T_WC,
                        plane_point_W,
                        plane_normal_W,
                        args.max_targets,
                        args.expected_top_height_m,
                    )
                    if not measured_targets:
                        status = "no target"
            except Exception as exc:  # noqa: BLE001 - realtime UI should stay alive and show error.
                status = str(exc)

            put_label(view, f"status: {status}", (20, 34), (0, 255, 0) if status == "OK" else (0, 0, 255), 0.75)
            put_label(view, f"fps={fps:.1f}", (20, 64), (255, 255, 0), 0.7)
            if mean_err is not None:
                put_label(view, f"reproj={mean_err:.3f}px", (20, 94), (255, 255, 0), 0.7)
            if camera_height is not None:
                put_label(view, f"cam_h={camera_height:.3f}m", (20, 124), (255, 255, 0), 0.7)

            if args.show_depth:
                dvis = depth_vis(depth_img)
                dvis = cv2.resize(dvis, (view.shape[1], view.shape[0]))
                view = np.hstack([view, dvis])

            latest_view = view

            for idx, target in enumerate(measured_targets, start=1):
                rows.append(
                    {
                        "frame_index": frame_index,
                        "target_index": idx,
                        "top_z_m": target["top_z_m"],
                        "bottom_z_m": target["bottom_z_m"],
                        "height_m": target["height_m"],
                        "bbox_xywh": json.dumps(target["bbox_xywh"]),
                        "mean_reproj_px": mean_err,
                        "camera_height_m": camera_height,
                    }
                )

            if not args.no_window:
                cv2.imshow("Realtime height measurement", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and latest_view is not None:
                    cv2.imwrite(str(save_dir / "manual_snapshot.png"), latest_view)

            if frame_index % 15 == 0:
                tops = [f"{row['top_z_m']:.3f}" for row in measured_targets]
                print(f"frame={frame_index:04d} status={status} targets_top_m={tops}", flush=True)
            frame_index += 1
    finally:
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()

    if latest_view is not None:
        cv2.imwrite(str(save_dir / "latest_realtime_view.png"), latest_view)

    if rows:
        with (save_dir / "realtime_measurements.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        top_values = np.asarray([float(row["top_z_m"]) for row in rows], dtype=np.float64)
        summary = {
            "rows": len(rows),
            "median_top_z_m": float(np.median(top_values)),
            "mean_top_z_m": float(np.mean(top_values)),
            "std_top_z_m": float(np.std(top_values)),
            "depth_scale_m_per_unit": depth_scale,
            "method": "Realtime wall target measurement using configured floor tags and the configured wall tag.",
        }
    else:
        summary = {
            "rows": 0,
            "depth_scale_m_per_unit": depth_scale,
            "method": "No target measurements were produced.",
        }

    (save_dir / "realtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved_dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
