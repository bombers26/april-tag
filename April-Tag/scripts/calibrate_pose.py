from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class TagPose:
    tag_id: int
    center_m: np.ndarray
    yaw_rad: float


@dataclass(frozen=True)
class Layout:
    tag_size_m: float
    floor_tags: dict[int, TagPose]


PERMUTATIONS: dict[str, list[int]] = {
    "0123": [0, 1, 2, 3],
    "1230": [1, 2, 3, 0],
    "2301": [2, 3, 0, 1],
    "3012": [3, 0, 1, 2],
    "0321": [0, 3, 2, 1],
    "3210": [3, 2, 1, 0],
    "2103": [2, 1, 0, 3],
    "1032": [1, 0, 3, 2],
}


def load_layout(path: Path) -> Layout:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    tag_size_m = float(raw["tag_size_m"])
    floor_tags: dict[int, TagPose] = {}
    for tag_id_raw, tag_raw in raw["floor_tags"].items():
        tag_id = int(tag_id_raw)
        floor_tags[tag_id] = TagPose(
            tag_id=tag_id,
            center_m=np.asarray(tag_raw["center_m"], dtype=np.float64),
            yaw_rad=math.radians(float(tag_raw.get("yaw_deg", 0.0))),
        )

    return Layout(tag_size_m=tag_size_m, floor_tags=floor_tags)


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def camera_matrix_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    intr = metadata["color_intrinsics"]
    return np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["ppx"])],
            [0.0, float(intr["fy"]), float(intr["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_detection_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_rows_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame_index = int(row["frame_index"])
        by_frame.setdefault(frame_index, []).append(row)
    return by_frame


def tag_base_corners(size_m: float) -> np.ndarray:
    half = size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def rotation_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def world_corners_for_tag(tag: TagPose, size_m: float, permutation: list[int]) -> np.ndarray:
    base = tag_base_corners(size_m)[permutation]
    return base @ rotation_z(tag.yaw_rad).T + tag.center_m


def image_corners_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [float(row["corner_0_x"]), float(row["corner_0_y"])],
            [float(row["corner_1_x"]), float(row["corner_1_y"])],
            [float(row["corner_2_x"]), float(row["corner_2_y"])],
            [float(row["corner_3_x"]), float(row["corner_3_y"])],
        ],
        dtype=np.float64,
    )


def build_points(
    frame_rows: list[dict[str, Any]],
    layout: Layout,
    permutation: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_ids: list[int] = []
    seen_floor_ids: set[int] = set()

    for row in frame_rows:
        tag_id = int(row["tag_id"])
        tag = layout.floor_tags.get(tag_id)
        if tag is None:
            continue
        if tag_id in seen_floor_ids:
            raise ValueError(
                f"Duplicate configured floor tag id {tag_id} in one frame; "
                "cover duplicate tags or use unique physical tag ids before calibration."
            )
        seen_floor_ids.add(tag_id)
        object_points.append(world_corners_for_tag(tag, layout.tag_size_m, permutation))
        image_points.append(image_corners_from_row(row))
        used_ids.append(tag_id)

    if not object_points:
        return np.empty((0, 3)), np.empty((0, 2)), []

    return np.vstack(object_points), np.vstack(image_points), used_ids


def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> tuple[float, float]:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    return float(np.mean(errors)), float(np.max(errors))


def solve_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    if len(object_points) < 8:
        raise ValueError("Need at least two visible tags, i.e. 8 corners")

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        None,
        rvec,
        tvec,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP iterative refine failed")

    mean_err, max_err = reprojection_error(object_points, image_points, rvec, tvec, K)
    camera_height_m = float(invert_T(rt_to_T(rvec, tvec))[2, 3])
    return rvec, tvec, mean_err, max_err, camera_height_m


def evaluate_permutations(
    by_frame: dict[int, list[dict[str, Any]]],
    layout: Layout,
    K: np.ndarray,
) -> tuple[str, list[dict[str, Any]]]:
    candidates: list[tuple[str, float, float, float, int, int]] = []
    detailed_by_name: dict[str, list[dict[str, Any]]] = {}

    for name, permutation in PERMUTATIONS.items():
        frame_results: list[dict[str, Any]] = []
        for frame_index, frame_rows in sorted(by_frame.items()):
            try:
                object_points, image_points, used_ids = build_points(frame_rows, layout, permutation)
            except ValueError:
                continue
            if len(set(used_ids)) < 2:
                continue
            try:
                rvec, tvec, mean_err, max_err, camera_height_m = solve_pose(
                    object_points,
                    image_points,
                    K,
                )
            except (RuntimeError, ValueError, cv2.error):
                continue

            frame_results.append(
                {
                    "frame_index": frame_index,
                    "mean_reproj_px": mean_err,
                    "max_reproj_px": max_err,
                    "camera_height_m": camera_height_m,
                    "used_ids": ",".join(str(x) for x in sorted(set(used_ids))),
                    "rvec": rvec.reshape(3).tolist(),
                    "tvec": tvec.reshape(3).tolist(),
                }
            )

        detailed_by_name[name] = frame_results
        if frame_results:
            mean_errors = np.asarray([x["mean_reproj_px"] for x in frame_results], dtype=np.float64)
            max_errors = np.asarray([x["max_reproj_px"] for x in frame_results], dtype=np.float64)
            heights = np.asarray([x["camera_height_m"] for x in frame_results], dtype=np.float64)
            positive_height_frames = int(np.sum(heights > 0.0))
            candidates.append(
                (
                    name,
                    float(np.median(mean_errors)),
                    float(np.median(max_errors)),
                    float(np.median(heights)),
                    positive_height_frames,
                    len(frame_results),
                )
            )

    if not candidates:
        raise RuntimeError("No valid PnP result for any corner permutation")

    candidates.sort(
        key=lambda x: (
            -(x[4] / max(x[5], 1)),
            x[1],
            x[2],
        )
    )
    best_name = candidates[0][0]
    return best_name, detailed_by_name[best_name]


def save_pose_outputs(
    save_dir: Path,
    best_permutation: str,
    frame_results: list[dict[str, Any]],
    layout: Layout,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "pose_frames.csv"
    csv_rows = []
    for row in frame_results:
        csv_rows.append(
            {
                "frame_index": row["frame_index"],
                "mean_reproj_px": row["mean_reproj_px"],
                "max_reproj_px": row["max_reproj_px"],
                "camera_height_m": row["camera_height_m"],
                "used_ids": row["used_ids"],
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    mean_errors = np.asarray([x["mean_reproj_px"] for x in frame_results], dtype=np.float64)
    max_errors = np.asarray([x["max_reproj_px"] for x in frame_results], dtype=np.float64)
    heights = np.asarray([x["camera_height_m"] for x in frame_results], dtype=np.float64)

    last = frame_results[-1]
    T_CW = rt_to_T(
        np.asarray(last["rvec"], dtype=np.float64).reshape(3, 1),
        np.asarray(last["tvec"], dtype=np.float64).reshape(3, 1),
    )
    T_WC = invert_T(T_CW)

    summary = {
        "best_corner_permutation": best_permutation,
        "frames_used": len(frame_results),
        "tag_size_m": layout.tag_size_m,
        "floor_tag_ids": sorted(layout.floor_tags),
        "median_mean_reprojection_px": float(np.median(mean_errors)),
        "median_max_reprojection_px": float(np.median(max_errors)),
        "median_camera_height_m": float(np.median(heights)),
        "std_camera_height_m": float(np.std(heights)),
        "last_T_camera_from_world": T_CW.tolist(),
        "last_T_world_from_camera": T_WC.tolist(),
        "note": (
            "World frame is defined by the two floor tags. Z=0 is the floor plane. "
            "This first-pass layout assumes the two floor tags are parallel and aligned with the measured outer-edge span."
        ),
    }
    (save_dir / "pose_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate camera pose from saved AprilTag detections.")
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("output/detections"),
        help="Directory containing detections.csv and metadata.json from detect_tags.py.",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=Path("configs/tag_layout.yaml"),
        help="Path to tag layout YAML.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("output/calibration"),
        help="Output directory for pose results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = load_layout(args.layout)
    metadata = load_metadata(args.detections_dir / "metadata.json")
    detections = load_detection_rows(args.detections_dir / "detections.csv")
    by_frame = group_rows_by_frame(detections)
    K = camera_matrix_from_metadata(metadata)

    best_perm, frame_results = evaluate_permutations(by_frame, layout, K)
    save_pose_outputs(args.save_dir, best_perm, frame_results, layout)

    mean_errors = np.asarray([x["mean_reproj_px"] for x in frame_results], dtype=np.float64)
    max_errors = np.asarray([x["max_reproj_px"] for x in frame_results], dtype=np.float64)
    heights = np.asarray([x["camera_height_m"] for x in frame_results], dtype=np.float64)

    print("Pose calibration summary")
    print(f"  best_corner_permutation: {best_perm}")
    print(f"  frames_used: {len(frame_results)}")
    print(f"  median_mean_reprojection_px: {float(np.median(mean_errors)):.3f}")
    print(f"  median_max_reprojection_px: {float(np.median(max_errors)):.3f}")
    print(f"  median_camera_height_m: {float(np.median(heights)):.3f}")
    print(f"  std_camera_height_m: {float(np.std(heights)):.4f}")
    print(f"  saved_dir: {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
