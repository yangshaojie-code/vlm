import unittest
from types import SimpleNamespace

import numpy as np

from ros_sensor_utils import SensorCache, SensorDataError, TransformStore, decode_depth, decode_image


def image_message(array, encoding, stamp=1.0, frame="camera"):
    array = np.asarray(array)
    if encoding == "16UC1":
        payload = array.astype("<u2").tobytes()
        step = array.shape[1] * 2
    elif encoding == "32FC1":
        payload = array.astype("<f4").tobytes()
        step = array.shape[1] * 4
    else:
        payload = array.astype(np.uint8).tobytes()
        step = array.shape[1] * array.shape[2] if array.ndim == 3 else array.shape[1]
    sec = int(stamp)
    header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=int((stamp - sec) * 1e9)), frame_id=frame)
    return SimpleNamespace(height=array.shape[0], width=array.shape[1], step=step, data=payload, encoding=encoding, is_bigendian=False, header=header)


class SensorUtilityTests(unittest.TestCase):
    def test_bgr_is_normalized_to_rgb_and_depth_is_meters(self):
        rgb = decode_image(image_message(np.array([[[3, 2, 1]]]), "bgr8"))
        np.testing.assert_array_equal(rgb, [[[1, 2, 3]]])
        depth = decode_depth(image_message(np.array([[1000]], dtype=np.uint16), "16UC1"))
        self.assertAlmostEqual(float(depth[0, 0]), 1.0)

    def test_transform_store_finds_multihop_path(self):
        store = TransformStore()
        first = np.eye(4); first[0, 3] = 1.0
        second = np.eye(4); second[1, 3] = 2.0
        store.set_transform("base_link", "camera", first)
        store.set_transform("odom", "base_link", second)
        np.testing.assert_allclose(store.lookup("odom", "camera")[:3, 3], [1.0, 2.0, 0.0])

    def test_snapshot_rejects_stale_pair(self):
        cache = SensorCache()
        rgb = image_message(np.zeros((2, 2, 3)), "rgb8", 1.0)
        depth = image_message(np.ones((2, 2), dtype=np.float32), "32FC1", 1.5)
        info = SimpleNamespace(width=2, height=2, k=[1, 0, 1, 0, 1, 1, 0, 0, 1], header=rgb.header)
        cache.update_rgb(rgb); cache.update_depth(depth); cache.update_camera_info(info)
        with self.assertRaises(TimeoutError):
            cache.wait_snapshot(timeout=0.0, max_skew=0.1)

    def test_snapshot_rejects_old_or_mismatched_frames(self):
        cache = SensorCache()
        rgb = image_message(np.zeros((2, 2, 3)), "rgb8", 1.0, "rgb_frame")
        depth = image_message(np.ones((2, 2), dtype=np.float32), "32FC1", 1.0, "depth_frame")
        info = SimpleNamespace(width=2, height=2, k=[1, 0, 1, 0, 1, 1, 0, 0, 1], header=rgb.header)
        cache.update_rgb(rgb); cache.update_depth(depth); cache.update_camera_info(info)
        with self.assertRaisesRegex(TimeoutError, "frame mismatch"):
            cache.wait_snapshot(timeout=0.0)

        depth.header.frame_id = "rgb_frame"
        cache.update_depth(depth)
        cache.rgb_received_at -= 2.0
        with self.assertRaisesRegex(TimeoutError, "stale"):
            cache.wait_snapshot(timeout=0.0)

    def test_joint_state_rejects_invalid_feedback_without_overwriting_cache(self):
        cache = SensorCache()
        valid = SimpleNamespace(name=["joint"], position=[0.25])
        cache.update_joint_state(valid)
        with self.assertRaises(SensorDataError):
            cache.update_joint_state(SimpleNamespace(name=["joint"], position=[np.nan]))
        np.testing.assert_allclose(cache.joint_vector(["joint"]), [0.25])


if __name__ == "__main__":
    unittest.main()
