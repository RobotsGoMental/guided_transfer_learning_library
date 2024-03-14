import unittest
import torch

from guided_transfer_learning.AdjustGuidanceMatrix import (
    AdjustGuidanceMatrix,
    Focus,
    Temp,
)


class TestAdjustGuidanceMatrix(unittest.TestCase):
    def setUp(self):
        self.guide = {
            "weight": torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "bias": torch.Tensor([0.1]),
        }
        self.guide_type = Focus.ZERO_ENFORCED_AND_NORMALIZED
        self.temperature = Temp.ROOM
        self.adjust_matrix = AdjustGuidanceMatrix(
            self.guide, self.guide_type, self.temperature
        )

    def test_apply_temperature(self):
        self.adjust_matrix.slope = 0.5
        self.adjust_matrix.intercept = 0.1
        self.adjust_matrix.apply_temperature()
        self.assertTrue(
            torch.equal(
                self.adjust_matrix.guidance_matrix["weight"],
                torch.Tensor([[0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9]]),
            )
        )

    def test_apply_guide_type(self):
        self.adjust_matrix.apply_focus()

    def test_string_arguments(self):
        self.guide_type = "zero_enforced_and_normalized"
        self.temperature = "room"
        self.adjust_matrix = AdjustGuidanceMatrix(
            self.guide, self.guide_type, self.temperature
        )
        self.adjust_matrix.apply_focus()

    def test_focus_types(self):
        self.guide_type = Focus.ZERO_ENFORCED
        self.adjust_matrix = AdjustGuidanceMatrix(
            self.guide, self.guide_type, self.temperature
        )
        self.adjust_matrix.apply_focus()
        self.guide_type = Focus.NORMALIZED
        self.adjust_matrix = AdjustGuidanceMatrix(
            self.guide, self.guide_type, self.temperature
        )
        self.adjust_matrix.apply_focus()
        self.guide_type = Focus.RAW
        self.adjust_matrix = AdjustGuidanceMatrix(
            self.guide, self.guide_type, self.temperature
        )
        self.adjust_matrix.apply_focus()


if __name__ == "__main__":
    unittest.main()
