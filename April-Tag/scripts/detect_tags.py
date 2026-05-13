from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyrealsense2. Install it with:\n"
        "  python -m pip install pyrealsense2"
    ) from exc

try:
    from pupil_apriltags import Detector
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pupil-apriltags. Install it with:\n"
        "  python -m pip install pupil-apriltags"
    ) from exc


@dataclass(frozen=True)
class CameraConfig:
    depth_width: int
    depth_height: int
    color_width: int
    color_height: int
    fps: int
    align_depth_to_color: bool


@dataclass(frozen=True)
class AprilTagConfig:
    family: str
    expected_ids: list[int]
    tag_size_m: float


@dataclass(frozen=True)
class CaptureConfig:
    warmup_frames: int
    frames: int
    save_dir: Path


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    apriltag: AprilTagConfig
    capture: CaptureConfig


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    camera_raw = raw["camera"]
    apriltag_raw = raw["apriltag"]
    capture_raw = raw["capture"]

    return AppConfig(
        camera=CameraConfig(
            depth_width=int(camera_raw["depth_width"]),
            depth_height=int(camera_raw["depth_height"]),
            color_width=int(camera_raw["color_width"]),
            color_height=int(camera_raw["color_height"]),
            fps=int(camera_raw["fps"]),
            align_depth_to_color=bool(camera_raw["align_depth_to_color"]),
        ),
        apriltag=AprilTagConfig(
            family=str(apriltag_raw["family"]),
            expected_ids=[int(x) for x in apriltag_raw.get("expected_ids", [])],
            tag_size_m=float(apriltag_raw["tag_size_m"]),
        ),
        capture=CaptureConfig(
            warmup_frames=int(capture_raw["warmup_frames"]),
            frames=int(capture_raw["frames"]),
            save_dir=Path(capture_raw["save_dir"]),
        ),
    )


def start_pipeline(camera: CameraConfig) -> tuple[rs.pipeline, rs.align | None, float]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.depth,
        camera.depth_width,
        camera.depth_height,
        rs.format.z16,
        camera.fps,
    )
    config.enable_stream(
        rs.stream.color,
        camera.color_width,
        camera.color_height,
        rs.format.bgr8,
        camera.fps,
    )

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    align = rs.align(rs.stream.color) if camera.align_depth_to_color else None
    return pipeline, align, depth_scale


def get_frames(
    pipeline: rs.pipeline,
    align: rs.align | None,
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    frames = pipeline.wait_for_frames()
    if align is not None:
        frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not read synchronized depth/color frames")

    depth_img = np.asanyarray(depth_frame.get_data())
    color_img = np.asanyarray(color_frame.get_data())
    depth_intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
    return color_img, depth_img, color_intr, depth_intr


def intrinsics_to_dict(intr: Any) -> dict[str, Any]:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def make_detector(family: str) -> Detector:
    return Detector(
        families=family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def detect_tags(detector: Detector, color_bgr: np.ndarray) -> list[Any]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    return list(detector.detect(gray, estimate_tag_pose=False))


def draw_detections(color_bgr: np.ndarray, detections: list[Any]) -> np.ndarray:
    out = color_bgr.copy()
    for det in detections:
        corners = np.asarray(det.corners, dtype=np.int32)
        cv2.polylines(out, [corners], isClosed=True, color=(0, 255, 0), thickness=3)

        center = tuple(np.asarray(det.center, dtype=np.int32))
        cv2.circle(out, center, 4, (0, 0, 255), -1)

        label = f"id={int(det.tag_id)} margin={float(det.decision_margin):.1f}"
        label_pos = (int(det.center[0]) + 8, int(det.center[1]) - 8)
        cv2.putText(
            out,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def depth_to_vis(depth_img: np.ndarray) -> np.ndarray:
    valid = depth_img[depth_img > 0]
    if valid.size == 0:
        return np.zeros((*depth_img.shape, 3), dtype=np.uint8)

    upper = float(np.percentile(valid, 95))
    upper = max(upper, 1.0)
    scaled = np.clip(depth_img.astype(np.float32) * 255.0 / upper, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def detection_rows(frame_index: int, detections: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det in detections:
        corners = np.asarray(det.corners, dtype=np.float64)
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "tag_id": int(det.tag_id),
            "hamming": int(det.hamming),
            "decision_margin": float(det.decision_margin),
            "center_x": float(det.center[0]),
            "center_y": float(det.center[1]),
        }
        for i, (x, y) in enumerate(corners):
            row[f"corner_{i}_x"] = float(x)
            row[f"corner_{i}_y"] = float(y)
        rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_detection_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        tag_id = int(row["tag_id"])
        counts[tag_id] = counts.get(tag_id, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture D455 RGB-D frames and detect AprilTags in the color stream.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/d455.yaml"),
        help="Path to the D455 YAML config.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override number of frames to capture.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated color frames in an OpenCV window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    frames_to_capture = int(args.frames or cfg.capture.frames)
    save_dir = cfg.capture.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    detector = make_detector(cfg.apriltag.family)
    pipeline, align, depth_scale = start_pipeline(cfg.camera)

    all_rows: list[dict[str, Any]] = []
    latest_color: np.ndarray | None = None
    latest_depth: np.ndarray | None = None
    latest_annotated: np.ndarray | None = None
    color_intr: Any | None = None
    depth_intr: Any | None = None

    try:
        for _ in range(cfg.capture.warmup_frames):
            get_frames(pipeline, align)

        start = time.time()
        for frame_index in range(frames_to_capture):
            color_img, depth_img, color_intr, depth_intr = get_frames(pipeline, align)
            detections = detect_tags(detector, color_img)
            annotated = draw_detections(color_img, detections)

            rows = detection_rows(frame_index, detections)
            all_rows.extend(rows)

            detected_ids = [int(det.tag_id) for det in detections]
            print(
                f"frame={frame_index:04d} detected={detected_ids}",
                flush=True,
            )

            latest_color = color_img
            latest_depth = depth_img
            latest_annotated = annotated

            if args.show:
                cv2.imshow("D455 AprilTag detections", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        elapsed = max(time.time() - start, 1e-6)
        counts = summarize_detection_counts(all_rows)
        expected = set(cfg.apriltag.expected_ids)
        seen = set(counts)
        missing = sorted(expected - seen)

        if latest_color is not None:
            cv2.imwrite(str(save_dir / "latest_color.png"), latest_color)
        if latest_depth is not None:
            cv2.imwrite(str(save_dir / "latest_depth_vis.png"), depth_to_vis(latest_depth))
        if latest_annotated is not None:
            cv2.imwrite(str(save_dir / "latest_annotated.png"), latest_annotated)

        save_csv(save_dir / "detections.csv", all_rows)

        metadata = {
            "config": {
                "camera": asdict(cfg.camera),
                "apriltag": asdict(cfg.apriltag),
                "capture": {
                    "warmup_frames": cfg.capture.warmup_frames,
                    "frames": frames_to_capture,
                    "save_dir": str(save_dir),
                },
            },
            "depth_scale_m_per_unit": depth_scale,
            "color_intrinsics": intrinsics_to_dict(color_intr) if color_intr is not None else None,
            "depth_intrinsics": intrinsics_to_dict(depth_intr) if depth_intr is not None else None,
            "captured_frames": frames_to_capture,
            "elapsed_s": elapsed,
            "approx_fps": frames_to_capture / elapsed,
            "detections_per_tag": counts,
            "missing_expected_ids": missing,
        }
        (save_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nSummary")
        print(f"  saved_dir: {save_dir}")
        print(f"  detections_per_tag: {counts}")
        print(f"  missing_expected_ids: {missing}")
        print(f"  approx_fps: {frames_to_capture / elapsed:.2f}")
        print(f"  depth_scale_m_per_unit: {depth_scale}")

        return 0 if not missing else 2
    finally:
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
