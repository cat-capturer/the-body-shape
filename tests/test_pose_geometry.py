import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from realsense_pose_surface.pose_geometry import (
    Landmark3D, bone_transform, median_depth, normalized_pixel, smooth_landmarks,
)
from realsense_pose_surface.run_realsense_pose_surface import deproject_landmarks, parse_args


def point(x, visibility=0.9):
    return Landmark3D(np.array([x, 0., 1.]), visibility)


class DepthTests(unittest.TestCase):
    def test_window_one_reads_only_center(self):
        depth = np.full((3, 3), 4000)
        depth[1, 1] = 1000
        self.assertEqual(median_depth(depth, 1, 1, 1, 0.001), 1.)

    def test_invalid_and_out_of_range_values_removed_before_median(self):
        depth = np.array([[0, np.nan, np.inf], [-1, 1000, 1100], [9000, 50, 1200]])
        self.assertAlmostEqual(median_depth(depth, 1, 1, 3, .001, .2, 4.5), 1.1)

    def test_depth_boundary_gate_uses_center(self):
        depth = np.full((5, 5), 3000)
        depth[2, 2] = 1000
        depth[2, 1] = 1040
        self.assertAlmostEqual(median_depth(depth, 2, 2, 5, .001, .2, 4.5, .15), 1.02)

    def test_missing_center_falls_back_and_units_are_not_assumed(self):
        depth = np.array([[100, 110], [120, 0]])
        self.assertAlmostEqual(median_depth(depth, 1, 1, 3, .01, .2, 4.5, .15), 1.1)

    def test_empty_patch_and_out_of_bounds(self):
        for x, y in ((0, 0), (-1, 0), (2, 0), (0, 2)):
            self.assertEqual(median_depth(np.zeros((2, 2)), x, y, 5, .001), 0.)

    def test_window_and_units_validation(self):
        for window in (0, 2, -1):
            with self.assertRaises(ValueError):
                median_depth(np.ones((3, 3)), 1, 1, window, .001)
        for units in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                median_depth(np.ones((3, 3)), 1, 1, 1, units)

    def test_normalized_edge_and_invalid_coordinates(self):
        self.assertEqual(normalized_pixel(.9999, .9999, 640, 480), (639, 479))
        self.assertEqual(normalized_pixel(0, 0, 640, 480), (0, 0))
        for x, y in ((-.01, 0), (1, 0), (0, 1), (np.nan, 0), (0, np.inf)):
            self.assertIsNone(normalized_pixel(x, y, 640, 480))

    def test_projection_uses_valid_pixels_and_rejects_bad_visibility(self):
        args = parse_args(['--depth-window', '1'])
        results = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=[
            SimpleNamespace(x=.999, y=.999, visibility=.9),
            SimpleNamespace(x=.5, y=.5, visibility=np.nan),
            SimpleNamespace(x=-.1, y=.5, visibility=.9),
            SimpleNamespace(x=.5, y=.5, visibility=.1),
        ]))
        depth = SimpleNamespace(get_data=lambda: np.full((2, 4), 1000))
        calls = []
        def deproject(intrinsics, pixel, meters):
            calls.append((pixel, meters))
            return [pixel[0], pixel[1], meters]
        with patch.dict(sys.modules, {'pyrealsense2': SimpleNamespace(rs2_deproject_pixel_to_point=deproject)}):
            output = deproject_landmarks(results, depth, object(), 4, 2, .001, args)
            self.assertEqual(list(output), [0])
            self.assertEqual(calls, [([3, 1], 1.)])
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                deproject_landmarks(results, depth, object(), 8, 4, .001, args)


class SmoothingTests(unittest.TestCase):
    def test_thirty_hz_preserves_original_weight(self):
        output = smooth_landmarks({0: point(0)}, {0: point(.5, .8)}, .68)
        self.assertAlmostEqual(output[0].point[0], .16)
        self.assertEqual(output[0].visibility, .8)

    def test_equal_elapsed_time_at_different_frame_rates(self):
        at_30 = at_60 = {0: point(0)}
        target = {0: point(.5)}
        for _ in range(3):
            at_30 = smooth_landmarks(at_30, target, .68, 1/30)
        for _ in range(6):
            at_60 = smooth_landmarks(at_60, target, .68, 1/60)
        np.testing.assert_allclose(at_30[0].point, at_60[0].point)

    def test_missing_points_disappear_and_reacquisition_is_fresh(self):
        self.assertEqual(smooth_landmarks({0: point(0)}, {}, .68), {})
        np.testing.assert_equal(smooth_landmarks({}, {0: point(.5)}, .68)[0].point, point(.5).point)

    def test_jump_and_timestamp_gap_reset(self):
        np.testing.assert_equal(smooth_landmarks({0: point(0)}, {0: point(2)}, .68)[0].point, point(2).point)
        for dt in (1., 0., -1., np.nan, np.inf):
            output = smooth_landmarks({0: point(0)}, {0: point(.5)}, .68, dt)
            np.testing.assert_equal(output[0].point, point(.5).point)


class BoneTests(unittest.TestCase):
    def test_endpoints_and_radius_for_parallel_antiparallel_and_oblique_segments(self):
        start = np.array([1., 2., 3.])
        for delta in ([0, 0, 2], [0, 0, -2], [2, -3, 1], [1, 0, 0]):
            end = start + delta
            matrix = bone_transform(start, end, .02)
            np.testing.assert_allclose((matrix @ [0, 0, -.5, 1])[:3], start)
            np.testing.assert_allclose((matrix @ [0, 0, .5, 1])[:3], end)
            self.assertAlmostEqual(np.linalg.norm(matrix[:3, 0]), .02)
            self.assertAlmostEqual(np.dot(matrix[:3, 0], matrix[:3, 2]), 0.)

    def test_degenerate_bone(self):
        for end in ([0, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]):
            self.assertIsNone(bone_transform(np.zeros(3), np.array(end), .02))


class ArgumentTests(unittest.TestCase):
    def test_invalid_arguments_fail_before_hardware_imports(self):
        for arguments in (['--depth-window', '2'], ['--fps', '0'], ['--smooth', 'nan'],
                          ['--sphere-radius', '-1'], ['--min-depth', '5'],
                          ['--tracking-confidence', '1.1'], ['--depth-width', '0'],
                          ['--depth-tolerance', 'inf']):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_help_works_without_camera_pose_or_render_dependencies(self):
        with patch.dict(sys.modules, {'open3d': None, 'mediapipe': None, 'pyrealsense2': None}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args(['--help'])
            self.assertEqual(error.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
