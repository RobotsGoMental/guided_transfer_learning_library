import unittest
from typing import List

import torch

from guided_transfer_learning.gtl import (
    _create_scout_data,
    expand_scout_indexes,
    get_guidance_values,
    list_params,
    GuidanceMatrix,
    get_param_values,
)


class TestCreateScoutData(unittest.TestCase):
    def setUp(self):
        self.scout_indexes = [[1, 2], [2, 3], [4, 5]]
        self.features_or_labels = torch.tensor([k for k in range(1, 10)])

    def test_create_scout_data_type(self):
        result = _create_scout_data(self.scout_indexes, self.features_or_labels)
        self.assertEqual(type(result), type([]))
        self.assertEqual(type(result[0]), torch.Tensor)

    def test_create_scout_data_content(self):
        result = _create_scout_data(self.scout_indexes, self.features_or_labels)
        for idx, result_data in enumerate(result):
            self.assertTrue(
                torch.equal(
                    result_data, self.features_or_labels[self.scout_indexes[idx]]
                )
            )

    def test_create_scout_data_length(self):
        result = _create_scout_data(self.scout_indexes, self.features_or_labels)
        self.assertEqual(len(result), len(self.scout_indexes))
        for idx, result_data in enumerate(result):
            self.assertEqual(len(result_data), len(self.scout_indexes[idx]))

    def test_create_scout_data_invalid_values(self):
        with self.assertRaises(TypeError):
            _create_scout_data(self.scout_indexes, "Invalid Tensor")


class TestGetGuideValues(unittest.TestCase):
    def test_get_guide_values(self):
        guidance_matrix: GuidanceMatrix = {
            "g1": torch.Tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ),
            "g2": torch.Tensor([[13, 14, 15], [17, 18, 16], [21, 22, 23]]),
            "g3": torch.Tensor([[25], [29]]),
        }
        result = get_guidance_values(guidance_matrix)
        self.assertEqual(len(result), len(guidance_matrix))

        self.assertEqual(result["g1"].shape, torch.Size([4, 2]))
        self.assertEqual(result["g2"].shape, torch.Size([3, 2]))
        self.assertEqual(result["g3"].shape, torch.Size([2, 1]))

        self.assertTrue(torch.equal(result["g1"], guidance_matrix["g1"][:, :2]))
        self.assertTrue(torch.equal(result["g2"], guidance_matrix["g2"][:, :2]))
        self.assertTrue(torch.equal(result["g3"], guidance_matrix["g3"]))

    def test_get_guide_values_change_first_n(self):
        guidance_matrix: GuidanceMatrix = {
            "g1": torch.Tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ),
            "g2": torch.Tensor([[13, 14, 15], [17, 18, 16], [21, 22, 23]]),
            "g3": torch.Tensor([[25], [29]]),
        }

        result = get_guidance_values(guidance_matrix, first_n=1)
        self.assertEqual(len(result), len(guidance_matrix))

        self.assertEqual(result["g1"].shape, torch.Size([4, 1]))
        self.assertEqual(result["g2"].shape, torch.Size([3, 1]))
        self.assertEqual(result["g3"].shape, torch.Size([2, 1]))

        self.assertTrue(torch.equal(result["g1"], guidance_matrix["g1"][:, :1]))
        self.assertTrue(torch.equal(result["g2"], guidance_matrix["g2"][:, :1]))
        self.assertTrue(torch.equal(result["g3"], guidance_matrix["g3"]))

    def test_get_guide_values_with_empty_matrix(self):
        guidance_matrix = {}
        result = get_guidance_values(guidance_matrix)
        self.assertEqual(result, {})

    def test_get_guide_values_with_empty_lists(self):
        guidance_matrix: GuidanceMatrix = {
            "g1": torch.Tensor([]),
            "g2": torch.Tensor([]),
            "g3": torch.Tensor([]),
        }

        result = get_guidance_values(guidance_matrix)
        self.assertTrue(result["g1"].shape, torch.Size([0, 0]))
        self.assertTrue(result["g2"].shape, torch.Size([0, 0]))
        self.assertTrue(result["g3"].shape, torch.Size([0, 0]))


class TestExpandScoutIndexes(unittest.TestCase):
    def test_index_single_integer(self):
        test_list = [[1]]
        expected_result = [[1]]
        actual_result = expand_scout_indexes(test_list)
        self.assertEqual(actual_result, expected_result)

    def test_single_ints(self):
        input_data = [[2, 3, 4]]
        output_data = expand_scout_indexes(input_data)
        expected_data = [[2, 3, 4]]
        self.assertEqual(output_data, expected_data)

    def test_index_range(self):
        test_list = [[[1], [5]]]
        expected_result = [[1, 2, 3, 4, 5]]
        actual_result = expand_scout_indexes(test_list)
        self.assertEqual(actual_result, expected_result)

    def test_multiple_scout_indexes(self):
        test_list = [[1], [[1], [5]]]
        expected_result = [[1], [1, 2, 3, 4, 5]]
        actual_result = expand_scout_indexes(test_list)
        self.assertEqual(actual_result, expected_result)

    def test_empty_scout_indexes(self):
        test_list = [[]]
        expected_result = [[]]
        actual_result = expand_scout_indexes(test_list)
        self.assertEqual(actual_result, expected_result)

    def test_full_example(self):
        test_list = [
            [1, 2, 4],
            [[7], [9]],
            [3, 5, 6],
            [[5], [20], 23, 24, [28], [30]],
            [40, 45, 46],
        ]
        expected_result = [
            [1, 2, 4],
            [7, 8, 9],
            [3, 5, 6],
            [
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                23,
                24,
                28,
                29,
                30,
            ],
            [40, 45, 46],
        ]
        actual_result = expand_scout_indexes(test_list)
        self.assertEqual(actual_result, expected_result)

    def test_fail_type(self):
        with self.assertRaises(TypeError):
            expand_scout_indexes(1)

        with self.assertRaises(TypeError):
            expand_scout_indexes([1])

        with self.assertRaises(TypeError):
            expand_scout_indexes([1, 2, 3, 4, 5])


class TestListParams(unittest.TestCase):
    def test_list_params_simple_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 2), torch.nn.ReLU()
        )  # simple test model

        result = list_params(model)

        self.assertIsInstance(result, List)

        for param_name, param_size in result:
            self.assertIsInstance(param_name, str)
            self.assertIsInstance(param_size, torch.Size)

            # the parameter shapes should match the defined model structure
            if "weight" in param_name:
                self.assertEqual(param_size, torch.Size([2, 3]))
            elif "bias" in param_name:
                self.assertEqual(param_size, torch.Size([2]))


class TestGetParamValues(unittest.TestCase):
    def setUp(self):
        # Create a simple sequential model for testing
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )

    def assert_param_values(self, first_n, expected):
        actual = get_param_values(self.model, first_n)
        self.assertEqual(
            len(actual), len(expected), "Length mismatch in parameter values"
        )
        for actual_value, expected_value in zip(actual, expected):
            self.assertTrue(
                torch.allclose(actual_value[:first_n], expected_value[:first_n]),
                "Parameter values mismatch",
            )

    def test_get_param_values_5(self):
        # Expect the first 5 elements of each parameter tensor in the model
        expected = [param for name, param in self.model.named_parameters()]
        self.assert_param_values(5, expected)

    def test_get_param_values_1(self):
        # Expect only the first element of each parameter tensor in the model
        expected = [param for name, param in self.model.named_parameters()]
        self.assert_param_values(1, expected)


if __name__ == "__main__":
    unittest.main()
