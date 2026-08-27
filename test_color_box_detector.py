import unittest

import numpy as np

from color_box_detector import detect_colored_boxes


class ColorBoxDetectorTests(unittest.TestCase):
    def test_detects_all_synthetic_box_colors(self):
        detections = detect_colored_boxes("test_images/grounding_cases/clear_three_boxes.png")
        colors = {item.color for item in detections}
        self.assertTrue({"pink", "yellow", "brown"}.issubset(colors))

    def test_color_filter(self):
        detections = detect_colored_boxes("test_images/grounding_cases/clear_three_boxes.png", "褐色")
        self.assertTrue(detections)
        self.assertTrue(all(item.color == "brown" for item in detections))

    def test_rejects_scene_scale_brown_region(self):
        image = np.full((480, 640, 3), (160, 124, 83), dtype=np.uint8)
        self.assertEqual(detect_colored_boxes(image, "brown", min_area=60), [])

    def test_detects_compact_white_packaging_box(self):
        image = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
        image[200:280, 310:400] = (240, 240, 238)
        detections = detect_colored_boxes(image, "white", min_area=60)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].color, "white")
        self.assertEqual(detections[0].bbox, (310, 200, 400, 280))


if __name__ == "__main__":
    unittest.main()
