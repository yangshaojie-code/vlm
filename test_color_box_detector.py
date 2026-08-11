import unittest

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


if __name__ == "__main__":
    unittest.main()

