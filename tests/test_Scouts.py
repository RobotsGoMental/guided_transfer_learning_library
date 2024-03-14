import unittest
import torch
from guided_transfer_learning import Scouts
from pathlib import Path


class TestScouts(unittest.TestCase):
    path_to_model = Path("gtl/Scouts.py")

    def setUp(self):
        self.scouts = Scouts(
            base_model=torch.nn.Linear(20, 1),
            path=self.path_to_model,
            should_save_guidance=False,
            should_save_scouts=False,
        )
        self.test_model = torch.nn.Linear(20, 1)

    def test_length(self):
        self.assertEqual(len(self.scouts), 0)
        self.scouts.add_scout(self.test_model)
        self.assertEqual(len(self.scouts), 1)

    def test_add_scout(self):
        expected_scout_count = self.scouts.__dict__["_Scouts__scout_count"] + 1
        self.scouts.add_scout(self.test_model)
        actual_scout_count = self.scouts.__dict__["_Scouts__scout_count"]
        self.assertEqual(expected_scout_count, actual_scout_count)

    def test_create_raw_guidance(self):
        self.scouts.add_scout(self.test_model)
        guidance = self.scouts.create_raw_guidance()
        self.assertIsInstance(guidance, dict)
        for value in guidance.values():
            self.assertIsInstance(value, torch.Tensor)

    def test_calculate_update_value(self):
        test_difference = torch.randn(5)
        square = self.scouts._Scouts__calculate_update_value(test_difference)
        self.assertTrue(torch.equal(square, torch.square(test_difference)))

    def test_absolute_difference(self):
        scout = Scouts(
            base_model=torch.nn.Linear(20, 1),
            path=self.path_to_model,
            should_save_guidance=False,
            should_save_scouts=False,
            use_squared_differences=False,
        )
        scout.add_scout(self.test_model)
        guidance = scout.create_raw_guidance()
        for value in guidance.values():
            self.assertTrue(torch.equal(value, torch.abs(value)))


if __name__ == "__main__":
    unittest.main()
