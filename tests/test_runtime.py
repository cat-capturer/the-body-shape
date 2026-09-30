"""Hardware boundary tests with SDK/window doubles; no real camera or GUI required."""
import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from realsense_pose_surface import run_realsense_pose_surface as app
from realsense_pose_surface.pose_geometry import Landmark3D


class RendererTests(unittest.TestCase):
    def make_renderer(self):
        def mesh(**kwargs):
            return SimpleNamespace(vertices=np.array([[0., 0., -.5], [0., 0., .5], [1., 0., 0.]]),
                                   paint_uniform_color=Mock(), compute_vertex_normals=Mock())
        factory = SimpleNamespace(create_sphere=Mock(side_effect=mesh),
                                  create_cylinder=Mock(side_effect=mesh),
                                  create_coordinate_frame=Mock(side_effect=mesh))
        visualizer = Mock()
        visualizer.create_window.return_value = True
        visualizer.poll_events.return_value = True
        o3d = SimpleNamespace(geometry=SimpleNamespace(TriangleMesh=factory),
                              utility=SimpleNamespace(Vector3dVector=np.asarray),
                              visualization=SimpleNamespace(Visualizer=lambda: visualizer))
        mp = SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(POSE_CONNECTIONS=[(0, 1)])))
        with patch.dict(sys.modules, {'open3d': o3d, 'mediapipe': mp}):
            renderer = app.SurfaceRenderer(.045, .022)
        return renderer, visualizer, factory

    def test_visible_meshes_are_reused_without_accumulated_transforms(self):
        renderer, vis, factory = self.make_renderer()
        landmarks = {0: Landmark3D(np.array([1., 0., 2.]), .9),
                     1: Landmark3D(np.array([1., 1., 2.]), .9)}
        renderer.update(landmarks)
        mesh = renderer.meshes[('joint', 0)][0]
        original = mesh.vertices.copy()
        for _ in range(10):
            renderer.update(landmarks)
        np.testing.assert_equal(mesh.vertices, original)
        self.assertEqual(factory.create_sphere.call_count, 2)
        self.assertEqual(factory.create_cylinder.call_count, 1)
        self.assertEqual(vis.add_geometry.call_count, 4)  # axes + 2 joints + bone
        self.assertEqual(vis.remove_geometry.call_count, 0)
        self.assertEqual(vis.update_geometry.call_count, 30)
        self.assertEqual(vis.reset_view_point.call_count, 1)
        self.assertEqual(vis.update_renderer.call_count, 11)

    def test_lost_points_and_bones_hide_then_reuse_cache(self):
        renderer, vis, factory = self.make_renderer()
        landmarks = {0: Landmark3D(np.array([0., 0., 1.]), .9),
                     1: Landmark3D(np.array([0., 0., 2.]), .9)}
        renderer.update(landmarks)
        renderer.update({0: landmarks[0]})
        self.assertEqual(renderer.active, {('joint', 0)})
        self.assertEqual(vis.remove_geometry.call_count, 2)
        renderer.update(landmarks)
        self.assertEqual(factory.create_sphere.call_count, 2)
        self.assertEqual(factory.create_cylinder.call_count, 1)
        vis.poll_events.return_value = False
        self.assertFalse(renderer.update({}))


class SourceTests(unittest.TestCase):
    def test_camera_selection_separate_depth_profile_and_aligned_intrinsics(self):
        rs = Mock()
        rs.stream.depth, rs.stream.color = 'depth', 'color'
        rs.format.z16, rs.format.bgr8 = 'z16', 'bgr8'
        rs.context.return_value.query_devices.return_value = ['device']
        rs.pipeline.return_value.start.return_value.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = .002
        aligned = rs.align.return_value.process.return_value
        aligned.get_color_frame.return_value.get_data.return_value = np.zeros((480, 640, 3))
        aligned.get_color_frame.return_value.get_timestamp.return_value = 1500
        intrinsics = object()
        aligned.get_depth_frame.return_value.profile.as_video_stream_profile.return_value.intrinsics = intrinsics
        with patch.dict(sys.modules, {'pyrealsense2': rs}):
            source = app.RealSensePoseSource(640, 480, 30, 'camera', 848, 480)
            source.start()
            bundle = source.next_frames()
            source.stop()
            source.stop()
        rs.config.return_value.enable_device.assert_called_once_with('camera')
        rs.config.return_value.enable_stream.assert_any_call('depth', 848, 480, 'z16', 30)
        self.assertEqual(source.depth_scale, .002)
        self.assertIs(bundle[2], intrinsics)
        self.assertEqual(bundle[3], 1.5)
        rs.pipeline.return_value.stop.assert_called_once()


class LifecycleTests(unittest.TestCase):
    def test_processing_error_releases_all_started_resources(self):
        source, estimator, renderer, cv2 = Mock(), Mock(), Mock(), Mock()
        source.next_frames.side_effect = RuntimeError('frame failure')
        with patch.object(app, 'parse_args', return_value=app.parse_args([])), \
                patch.object(app, 'RealSensePoseSource', return_value=source), \
                patch.object(app, 'PoseEstimator', return_value=estimator), \
                patch.object(app, 'SurfaceRenderer', return_value=renderer), \
                patch.dict(sys.modules, {'cv2': cv2}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'frame failure'):
                app.main()
        source.stop.assert_called_once()
        estimator.close.assert_called_once()
        renderer.close.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()

    def test_headless_interrupt_does_not_import_renderer_or_call_window_apis(self):
        source, estimator, cv2 = Mock(), Mock(), Mock()
        source.next_frames.side_effect = KeyboardInterrupt
        with patch.object(app, 'parse_args', return_value=app.parse_args(['--no-render', '--no-preview'])), \
                patch.object(app, 'RealSensePoseSource', return_value=source), \
                patch.object(app, 'PoseEstimator', return_value=estimator), \
                patch.dict(sys.modules, {'cv2': cv2, 'open3d': None}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                app.main()
        source.stop.assert_called_once()
        estimator.close.assert_called_once()
        cv2.destroyAllWindows.assert_not_called()
        cv2.imshow.assert_not_called()


if __name__ == '__main__':
    unittest.main()
