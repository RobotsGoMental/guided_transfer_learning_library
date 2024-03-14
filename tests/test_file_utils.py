import unittest
from guided_transfer_learning.file_utils import save_base, save_guidance_matrix
from pathlib import Path
import torch
import os
import tempfile


class TestSaveBase(unittest.TestCase):
    def setUp(self):
        # Create a dummy model
        self.model = torch.nn.Linear(2, 3)

        # Define a path and name for testing
        self.path = "/tmp/models"
        self.name = "test_model"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def test_save_base(self):
        # Call the function with the model, path, and name
        result = save_base(self.model, self.path, self.name)

        # assert the return is a Path object
        self.assertIsInstance(result, Path)

        # assert the correct file path is returned
        expected_path = Path(self.path, self.name)
        self.assertEqual(result, expected_path)

        # assert the file exists after save
        self.assertTrue(os.path.exists(result))

        # Load the state_dict to verify
        state_dict = torch.load(result)

        # assert the loaded state_dict is equal to the original model's state_dict
        self.assertTrue(torch.equal(state_dict["weight"], self.model.state_dict()["weight"]))

    def test_save_guidance_matrix(self):
        # Create a guidance_matrix and path using tempfile
        guidance_matrix = torch.nn.Linear(5, 3)
        temp_dir = tempfile.TemporaryDirectory()

        # Call save_guidance_matrix function
        save_path = save_guidance_matrix(guidance_matrix, path=temp_dir.name, name="test_matrix.pt")

        # Ensure the correct path is returned
        self.assertEqual(save_path, Path(temp_dir.name) / "test_matrix.pt")

        # Ensure the file was correctly written on the disk
        self.assertTrue(os.path.exists(save_path))

        # Load the saved matrix and verify
        saved_matrix = torch.load(Path(temp_dir.name) / "test_matrix.pt")
        self.assertTrue(torch.equal(saved_matrix.weight, guidance_matrix.weight))

        temp_dir.cleanup()

    def tearDown(self):
        # Cleanup the created file
        expected_path = Path(self.path, self.name)
        if os.path.exists(expected_path):
            os.remove(expected_path)


if __name__ == "__main__":
    unittest.main()
