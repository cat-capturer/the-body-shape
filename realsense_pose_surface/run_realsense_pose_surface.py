#!/usr/bin/env python3
"""RealSense D455 human keypoints rendered as spherical surfaces."""

from __future__ import annotations

import argparse
import dataclasses
import math
import time

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


@dataclasses.dataclass
class Landmark3D:
    point: np.ndarray
    visibility: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render D455 human keypoints as spheres and capsules.")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-visibility", type=float, default=0.55)
    parser.add_argument("--sphere-radius", type=float, default=0.045, help="Sphere radius in meters.")
    parser.add_argument("--capsule-radius", type=float, default=0.022, help="Bone radius in meters.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd pixel window for median depth.")
    parser.add_argument("--smooth", type=float, default=0.68, help="EMA smoothing factor from 0 to 0.95.")
    parser.add_argument("--max-depth", type=float, default=4.5)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


class RealSensePoseSource:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.started = False

    def start(self) -> None:
        if len(rs.context().query_devices()) == 0:
            raise RuntimeError("No RealSense device was detected. Connect the D455 and check udev permissions.")
        profile = self.pipeline.start(self.config)
        self.started = True
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def next_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        return color, depth_frame, intrinsics


class PoseEstimator:
    def __init__(self, model_complexity: int) -> None:
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self.pose.close()

    def detect(self, bgr_image: np.ndarray):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.pose.process(rgb)


class SurfaceRenderer:
    def __init__(self, sphere_radius: float, capsule_radius: float) -> None:
        self.sphere_radius = sphere_radius
        self.capsule_radius = capsule_radius
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window("RealSense D455 Pose Surface", width=1280, height=760)
        self.geometries: list[o3d.geometry.Geometry] = []
        render = self.visualizer.get_render_option()
        render.background_color = np.asarray([0.02, 0.025, 0.025])
        render.mesh_show_back_face = True
        self.visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35))

    def close(self) -> None:
        self.visualizer.destroy_window()

    def update(self, landmarks: dict[int, Landmark3D]) -> bool:
        for geometry in self.geometries:
            self.visualizer.remove_geometry(geometry, reset_bounding_box=False)
        self.geometries.clear()

        for index, landmark in landmarks.items():
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=self.sphere_radius * joint_scale(index), resolution=16
            )
            sphere.compute_vertex_normals()
            sphere.translate(landmark.point)
            sphere.paint_uniform_color(joint_color(index))
            self.visualizer.add_geometry(sphere, reset_bounding_box=False)
            self.geometries.append(sphere)

        for start, end in POSE_CONNECTIONS:
            if start not in landmarks or end not in landmarks:
                continue
            capsule = create_cylinder_between(landmarks[start].point, landmarks[end].point, self.capsule_radius)
            if capsule is None:
                continue
            capsule.paint_uniform_color([1.0, 0.48, 0.34])
            capsule.compute_vertex_normals()
            self.visualizer.add_geometry(capsule, reset_bounding_box=False)
            self.geometries.append(capsule)

        return self.visualizer.poll_events()


def deproject_landmarks(results, depth_frame, intrinsics, width: int, height: int, depth_units: float, args):
    if not results.pose_landmarks:
        return {}
    depth_image = np.asanyarray(depth_frame.get_data())
    output: dict[int, Landmark3D] = {}
    for index, landmark in enumerate(results.pose_landmarks.landmark):
        if landmark.visibility < args.min_visibility:
            continue
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        depth_m = median_depth(depth_image, x, y, args.depth_window, depth_units)
        if depth_m <= 0.0 or depth_m > args.max_depth:
            continue
        point = np.asarray(rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m), dtype=np.float64)
        output[index] = Landmark3D(point=point, visibility=float(landmark.visibility))
    return output


def median_depth(depth_image: np.ndarray, x: int, y: int, window: int, units: float) -> float:
    radius = max(1, window // 2)
    y0 = max(0, y - radius)
    y1 = min(depth_image.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(depth_image.shape[1], x + radius + 1)
    patch = depth_image[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * float(units)


def smooth_landmarks(previous: dict[int, Landmark3D], current: dict[int, Landmark3D], smooth: float):
    if not previous:
        return current
    alpha = min(max(smooth, 0.0), 0.95)
    output: dict[int, Landmark3D] = {}
    for index, landmark in current.items():
        if index in previous:
            point = previous[index].point * alpha + landmark.point * (1.0 - alpha)
        else:
            point = landmark.point
        output[index] = Landmark3D(point=point, visibility=landmark.visibility)
    return output


def draw_preview(image: np.ndarray, results, fps: float, mirror: bool) -> np.ndarray:
    preview = image.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            preview,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    cv2.putText(preview, f"{fps:4.1f} FPS", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (115, 241, 200), 2)
    if mirror:
        preview = cv2.flip(preview, 1)
    return preview


def create_cylinder_between(start: np.ndarray, end: np.ndarray, radius: float):
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-4:
        return None
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=16)
    midpoint = (start + end) * 0.5
    cylinder.translate(midpoint)
    z_axis = np.asarray([0.0, 0.0, 1.0])
    direction_unit = direction / length
    axis = np.cross(z_axis, direction_unit)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-8:
        axis = axis / axis_norm
        angle = math.acos(float(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0)))
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(rotation, center=midpoint)
    elif np.dot(z_axis, direction_unit) < 0:
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([math.pi, 0.0, 0.0]))
        cylinder.rotate(rotation, center=midpoint)
    return cylinder


def joint_color(index: int) -> list[float]:
    if index in {11, 12, 23, 24}:
        return [0.45, 0.94, 0.78]
    if index in {15, 16, 27, 28}:
        return [0.52, 0.68, 1.0]
    if index == 0:
        return [1.0, 0.80, 0.42]
    return [0.30, 0.86, 0.72]


def joint_scale(index: int) -> float:
    if index == 0:
        return 1.25
    if index in {11, 12, 23, 24}:
        return 1.35
    if index in {13, 14, 25, 26}:
        return 1.08
    return 0.86


def main() -> int:
    args = parse_args()
    source = RealSensePoseSource(args.width, args.height, args.fps)
    estimator = None
    renderer = None
    smoothed: dict[int, Landmark3D] = {}
    last_time = time.monotonic()
    fps = 0.0
    try:
        source.start()
        estimator = PoseEstimator(args.model_complexity)
        if not args.no_render:
            renderer = SurfaceRenderer(args.sphere_radius, args.capsule_radius)
        print("RealSense D455 stream started. Press q in the preview window to quit.")
        while True:
            frame_bundle = source.next_frames()
            if frame_bundle is None:
                continue
            color, depth_frame, intrinsics = frame_bundle
            results = estimator.detect(color)
            landmarks = deproject_landmarks(
                results, depth_frame, intrinsics, args.width, args.height, source.depth_scale, args
            )
            smoothed = smooth_landmarks(smoothed, landmarks, args.smooth)
            now = time.monotonic()
            delta = max(now - last_time, 1e-6)
            last_time = now
            fps = fps * 0.88 + (1.0 / delta) * 0.12
            if renderer is not None and not renderer.update(smoothed):
                break
            if not args.no_preview:
                cv2.imshow("D455 RGB Pose", draw_preview(color, results, fps, args.mirror))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        source.stop()
        if estimator is not None:
            estimator.close()
        if renderer is not None:
            renderer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
