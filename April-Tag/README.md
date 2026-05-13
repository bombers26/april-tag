# AprilTag D455 Pose and Distance Measurement

本仓库整理自 `E:\siemens\camera` 中 AprilTag 相关代码，目标是保留一套独立、可上传 GitHub 的最小工程结构，用于：

- 使用 Intel RealSense D455 采集 RGB-D 数据。
- 在 color 图像中识别 AprilTag。
- 用地面 AprilTag 建立世界坐标系。
- 将相机坐标、深度点、墙面 tag 平面转换到世界坐标系。
- 在已定义的 tag 世界系下估计相机高度、墙面目标高度，以及近似的相机到探测器/图像平面距离。

注意：这里的距离是几何相机距离。若要测 X 光系统中的严格 SID/SDD，需要额外知道 X 光源点相对 D455 或 AprilTag 世界坐标系的位置。

## Repository Structure

```text
April-Tag/
  configs/
    d455.yaml              # D455 color/depth stream and AprilTag runtime config
    tag_layout.yaml        # AprilTag size, floor tag world coordinates, wall tag measurements
  scripts/
    detect_tags.py         # Capture RGB-D frames and detect AprilTags
    calibrate_pose.py      # Estimate camera pose from saved tag detections
    validate_geometry.py   # Convert depth pixels to world coordinates and validate geometry
    cross_validate_tags.py # Validate wall tag heights/distances from floor-tag world pose
    measure_picture_height.py # Intersect image rays with wall-tag plane
    realtime_measure.py    # Realtime wall target measurement using AprilTag geometry
  requirements.txt
  .gitignore
  README.md
```

## Coordinate System

`configs/tag_layout.yaml` defines the world frame:

- Origin: midpoint between the two configured floor tag centers.
- `X`: from floor tag `id0` center to floor tag `id2` center.
- `Y`: floor-plane direction toward the standing / measurement area.
- `Z`: vertical up.
- Floor plane: `Z = 0`.

The camera frame follows the OpenCV / RealSense convention:

- `x`: image right.
- `y`: image down.
- `z`: camera forward.

`calibrate_pose.py` estimates:

\[
T_{C \leftarrow W}
\]

then inverts it to obtain:

\[
T_{W \leftarrow C}
\]

This transform is saved in:

```text
output/calibration/pose_summary.json
```

## Install

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

If `pyrealsense2` cannot find the D455, first confirm the camera is connected and visible in RealSense Viewer.

## Basic Workflow

### 1. Detect AprilTags

```powershell
python scripts\detect_tags.py --frames 60
```

Expected outputs:

```text
output/detections/detections.csv
output/detections/metadata.json
output/detections/latest_annotated.png
```

`detections.csv` stores tag IDs, centers, and four image corners. `metadata.json` stores camera intrinsics.

### 2. Estimate Camera Pose

```powershell
python scripts\calibrate_pose.py
```

Expected outputs:

```text
output/calibration/pose_frames.csv
output/calibration/pose_summary.json
```

Important fields in `pose_summary.json`:

- `median_mean_reprojection_px`: PnP reprojection error.
- `median_camera_height_m`: camera optical center height above floor.
- `last_T_camera_from_world`: \(T_{C \leftarrow W}\).
- `last_T_world_from_camera`: \(T_{W \leftarrow C}\).

### 3. Validate Geometry

```powershell
python scripts\validate_geometry.py --frames 90 --wall-tag-id 3 --wall-center-height-m 1.631
```

This checks whether depth points on the floor have world `Z` near `0`, and whether a known wall tag center height matches the configured manual measurement.

### 4. Cross-Validate Wall Tags

```powershell
python scripts\cross_validate_tags.py --frames 90 --wall-tag-ids 0 3 --wall-center-height-m 1.6315 --wall-center-distance-m 0.661
```

This solves the world pose from floor tags, then converts wall tag centers from camera coordinates to world coordinates.

### 5. Wall Plane / Target Measurement

```powershell
python scripts\measure_picture_height.py --frames 60 --wall-tag-id 3 --expected-top-height-m 1.65
```

The wall tag provides a wall plane. Image rays from detected target corners are intersected with that plane, then converted into world coordinates.

## How AprilTag Is Used

The code does not use AprilTag only as a visual marker label. It uses the physical tag size and configured tag layout as a metric calibration target.

Core logic:

1. Detect each visible tag's four image corners.
2. Look up the corresponding four known world corners from `tag_layout.yaml`.
3. Use `cv2.solvePnP()` to estimate the camera pose.
4. Convert depth pixels or tag-relative pose estimates into world coordinates.
5. Measure distances or heights in the world frame.

The critical functions are:

- `scripts/detect_tags.py`
  - `make_detector()`
  - `detect_tags()`
  - `get_frames()`
- `scripts/calibrate_pose.py`
  - `world_corners_for_tag()`
  - `solve_pose()`
  - `rt_to_T()`
  - `invert_T()`
- `scripts/validate_geometry.py`
  - `solve_floor_pose_from_detections()`
  - `depth_pixels_to_world()`
  - `transform_point()`
- `scripts/measure_picture_height.py`
  - `wall_plane_from_tag()`
  - `intersect_pixels_with_plane()`

## Measuring SID / SDD-Like Distance

If your target is a D455-based geometric proxy for SID/SDD, define the terms explicitly:

- Camera optical center in world coordinates:

\[
O_W = T_{W \leftarrow C}[0:3, 3]
\]

- Detector or image plane from a wall-mounted AprilTag:

\[
\Pi: (X_W - P_W) \cdot n_W = 0
\]

where `P_W` is a point on the wall tag plane and `n_W` is the wall plane normal in world coordinates.

The perpendicular camera-to-plane distance is:

\[
d = |(O_W - P_W) \cdot n_W|
\]

This can be computed from:

- `T_WC` from `pose_summary.json`.
- `P_W, n_W` from `wall_plane_from_tag()`.

For true X-ray SID:

\[
SID = \|S_W - D_W\|
\]

where \(S_W\) is the X-ray source point and \(D_W\) is the detector/image plane reference point. AprilTag can locate the detector plane, but it cannot infer \(S_W\) unless the source has been separately calibrated into the same world frame.

## Current Configuration Notes

Current `configs/tag_layout.yaml` assumes:

- tag family: `tag36h11`
- black tag edge size: `0.148 m`
- floor tags: `id0` and `id2`
- floor tag center distance: `1.008 m`
- wall tag `id3` center height reference: about `1.6294-1.631 m`

If the camera or tags move, rerun detection and calibration before trusting any distance measurement.

## Suggested Git Upload

After reviewing the folder:

```powershell
cd E:\siemens\camera\April-Tag
git init
git branch -M main
git add .
git commit -m "Initial AprilTag D455 pose measurement code"
git remote add origin https://github.com/bombers26/April-Tag.git
git push -u origin main
```

If the remote repository already contains files, pull or clone it first, then copy this folder content into the cloned repository to avoid overwriting remote history.
