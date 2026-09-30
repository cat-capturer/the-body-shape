#!/usr/bin/env python3
"""RealSense D455 human keypoints rendered as spherical surfaces."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import time

import numpy as np

if __package__:
    from .pose_geometry import Landmark3D, bone_transform, median_depth, normalized_pixel, smooth_landmarks
else:
    from pose_geometry import Landmark3D, bone_transform, median_depth, normalized_pixel, smooth_landmarks


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render RealSense RGB-D human keypoints as spheres and bones.")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", help="Select a RealSense camera by serial number.")
    parser.add_argument("--depth-width", type=int, help="Depth width; defaults to --width.")
    parser.add_argument("--depth-height", type=int, help="Depth height; defaults to --height.")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-visibility", type=float, default=0.55)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--sphere-radius", type=float, default=0.045, help="Sphere radius in meters.")
    parser.add_argument("--capsule-radius", type=float, default=0.022, help="Bone radius in meters.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd pixel window for median depth.")
    parser.add_argument("--smooth", type=float, default=0.68, help="EMA smoothing factor from 0 to 0.95.")
    parser.add_argument("--max-depth", type=float, default=4.5)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--depth-tolerance", type=float, default=None,
                        help="Optional maximum depth difference from a valid center pixel, in meters.")
    parser.add_argument("--reset-distance", type=float, default=0.75,
                        help="Reset smoothing on jumps larger than this many meters.")
    parser.add_argument("--reset-after", type=float, default=0.5,
                        help="Reset smoothing after a frame gap of this many seconds.")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)
    for name in ("width", "height", "fps", "depth_width", "depth_height"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.depth_window < 1 or args.depth_window % 2 == 0:
        parser.error("--depth-window must be a positive odd integer")
    for name in ("min_visibility", "detection_confidence", "tracking_confidence", "smooth"):
        value = getattr(args, name)
        maximum = 0.95 if name == "smooth" else 1.0
        if not math.isfinite(value) or not 0 <= value <= maximum:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and {maximum}")
    for name in ("sphere_radius", "capsule_radius", "min_depth", "max_depth",
                 "depth_tolerance", "reset_distance", "reset_after"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.min_depth >= args.max_depth:
        parser.error("--min-depth must be smaller than --max-depth")
    return args


class RealSensePoseSource:
    def __init__(self, width: int, height: int, fps: int, serial=None,
                 depth_width=None, depth_height=None) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, depth_width or width, depth_height or height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.started = False

    def start(self) -> None:
        if len(self.rs.context().query_devices()) == 0:
            raise RuntimeError("No RealSense device was detected. Check the connection and udev permissions.")
        try:
            profile = self.pipeline.start(self.config)
        except RuntimeError as exc:
            raise RuntimeError("Cannot start RealSense streams. Check --serial and the color/depth "
                               "resolutions and FPS in realsense-viewer.") from exc
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
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        return color, depth_frame, intrinsics, color_frame.get_timestamp() / 1000.0


class PoseEstimator:
    def __init__(self, model_complexity: int, detection_confidence=0.5, tracking_confidence=0.5) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError("This implementation requires the legacy MediaPipe Pose API. "
                               "Install the versions in requirements.txt (MediaPipe 0.10.21).")
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def close(self) -> None:
        self.pose.close()

    def detect(self, bgr_image: np.ndarray):
        import cv2

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.pose.process(rgb)


class SurfaceRenderer:
    def __init__(self, sphere_radius: float, capsule_radius: float) -> None:
        import open3d as o3d
        import mediapipe as mp

        self.o3d = o3d
        self.connections = tuple(mp.solutions.pose.POSE_CONNECTIONS)
        self.sphere_radius = sphere_radius
        self.capsule_radius = capsule_radius
        self.visualizer = o3d.visualization.Visualizer()
        if not self.visualizer.create_window("RealSense Pose Surface", width=1280, height=760):
            self.visualizer.destroy_window()
            raise RuntimeError("Cannot open the Open3D window. Use --no-render on a headless machine.")
        self.meshes = {}
        self.active = set()
        self.fitted = False
        try:
            render = self.visualizer.get_render_option()
            render.background_color = np.asarray([0.02, 0.025, 0.025])
            render.mesh_show_back_face = True
            self.visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35))
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.visualizer.destroy_window()

    def _update_mesh(self, key, transform, color):
        o3d = self.o3d
        if key not in self.meshes:
            if key[0] == "joint":
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=16)
            else:
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=1.0, resolution=16)
            mesh.paint_uniform_color(color)
            self.meshes[key] = mesh, np.asarray(mesh.vertices).copy()
        mesh, template = self.meshes[key]
        # Always transform the original vertices, avoiding cumulative scale/rotation drift.
        mesh.vertices = o3d.utility.Vector3dVector(template @ transform[:3, :3].T + transform[:3, 3])
        mesh.compute_vertex_normals()
        if key in self.active:
            self.visualizer.update_geometry(mesh)
        else:
            self.visualizer.add_geometry(mesh, reset_bounding_box=False)

    def update(self, landmarks: dict[int, Landmark3D]) -> bool:
        visible = set()
        for index, landmark in landmarks.items():
            key = ("joint", index)
            transform = np.eye(4)
            transform[:3, :3] *= self.sphere_radius * joint_scale(index)
            transform[:3, 3] = landmark.point
            self._update_mesh(key, transform, joint_color(index))
            visible.add(key)
        for start, end in self.connections:
            if start not in landmarks or end not in landmarks:
                continue
            transform = bone_transform(landmarks[start].point, landmarks[end].point, self.capsule_radius)
            if transform is None:
                continue
            key = ("bone", start, end)
            self._update_mesh(key, transform, [1.0, 0.48, 0.34])
            visible.add(key)
        for key in self.active - visible:
            self.visualizer.remove_geometry(self.meshes[key][0], reset_bounding_box=False)
        self.active = visible
        if visible and not self.fitted:
            self.visualizer.reset_view_point(True)
            self.fitted = True
        alive = self.visualizer.poll_events()
        if alive:
            self.visualizer.update_renderer()
        return alive


def deproject_landmarks(results, depth_frame, intrinsics, width: int, height: int, depth_units: float, args):
    import pyrealsense2 as rs

    if not results.pose_landmarks:
        return {}
    depth_image = np.asanyarray(depth_frame.get_data())
    if depth_image.shape != (height, width):
        raise ValueError("Aligned depth and color dimensions must match")
    output: dict[int, Landmark3D] = {}
    for index, landmark in enumerate(results.pose_landmarks.landmark):
        if not math.isfinite(landmark.visibility) or landmark.visibility < args.min_visibility:
            continue
        pixel = normalized_pixel(landmark.x, landmark.y, width, height)
        if pixel is None:
            continue
        x, y = pixel
        depth_m = median_depth(depth_image, x, y, args.depth_window, depth_units,
                               args.min_depth, args.max_depth, args.depth_tolerance)
        if depth_m <= 0.0:
            continue
        point = np.asarray(rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m), dtype=np.float64)
        if np.all(np.isfinite(point)):
            output[index] = Landmark3D(point=point, visibility=float(landmark.visibility))
    return output


def draw_preview(image: np.ndarray, results, fps: float, mirror: bool) -> np.ndarray:
    import cv2
    import mediapipe as mp

    preview = image.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            preview,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    if mirror:
        preview = cv2.flip(preview, 1)
    cv2.putText(preview, f"{fps:4.1f} FPS", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (115, 241, 200), 2)
    return preview


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
    import cv2

    source = RealSensePoseSource(args.width, args.height, args.fps, args.serial,
                                args.depth_width, args.depth_height)
    renderer = None
    smoothed: dict[int, Landmark3D] = {}
    last_time = time.monotonic()
    last_timestamp = None
    fps = 0.0
    with ExitStack() as cleanup:
        if not args.no_preview:
            cleanup.callback(cv2.destroyAllWindows)
        cleanup.callback(source.stop)
        source.start()
        estimator = PoseEstimator(args.model_complexity, args.detection_confidence, args.tracking_confidence)
        cleanup.callback(estimator.close)
        if not args.no_render:
            renderer = SurfaceRenderer(args.sphere_radius, args.capsule_radius)
            cleanup.callback(renderer.close)
        print("RealSense stream started. Press Ctrl+C, q in preview, or close the 3D window to quit.")
        while True:
            frame_bundle = source.next_frames()
            if frame_bundle is None:
                continue
            color, depth_frame, intrinsics, timestamp = frame_bundle
            results = estimator.detect(color)
            landmarks = deproject_landmarks(
                results, depth_frame, intrinsics, color.shape[1], color.shape[0], source.depth_scale, args
            )
            dt = timestamp - last_timestamp if last_timestamp is not None else 1/30
            last_timestamp = timestamp
            smoothed = smooth_landmarks(smoothed, landmarks, args.smooth, dt,
                                        args.reset_distance, args.reset_after)
            now = time.monotonic()
            delta = max(now - last_time, 1e-6)
            last_time = now
            fps = fps * 0.88 + (1.0 / delta) * 0.12
            if renderer is not None and not renderer.update(smoothed):
                break
            if not args.no_preview:
                cv2.imshow("RealSense RGB Pose", draw_preview(color, results, fps, args.mirror))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
