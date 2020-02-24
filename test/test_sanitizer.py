import unittest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
import torch_testing
from blurnn import ModelSanitizer

class TestModelSanitizer(unittest.TestCase):
    def test_properties(self):
        base_model = torch.nn.Module()
        base_model.linear1 = nn.Linear(2, 2)
        base_model.linear2 = nn.Linear(3, 3)
        sanitizer = ModelSanitizer(eps1=0.1, eps3=0.1, epochs=5, release_proportion=0.5)
        sanitizer.base_model = base_model
        sanitizer.sanitize_init()

        self.assertEqual(sanitizer.eps1, 0.1)
        self.assertEqual(sanitizer.eps3, 0.1)
        self.assertEqual(sanitizer.model_size, 18)
        self.assertEqual(sanitizer.release_size, 9)
        self.assertEqual(sanitizer.gamma, 0.5)
        self.assertEqual(sanitizer.sensitivity, 1)
        self.assertEqual(sanitizer.eps2, ((2 * 9 * 1) ** (2/3)) * 0.1)

        sanitizer.eps2 = 0.1
        self.assertEqual(sanitizer.eps2, 0.1)

    def test_pick_one_round(self):
        with patch('torch.randn_like') as randn_like:
            do_nothing = lambda: None
            sanitizer = ModelSanitizer(1, 1, 1, 1, 1, 1, 1, 1)
            sanitizer.sanitize_init = Mock(side_effect=do_nothing)
            sanitizer.model_size = 5
            sanitizer.release_size = 1
            sanitizer.epochs = 1
            sanitizer.gamma = 10000
            sanitizer.tau = 2
            sanitizer._noise1_distributions = Mock()
            sanitizer._noise2_distributions = Mock()
            sanitizer._noise3_distributions = Mock()
            sanitizer._noise1_distributions.sample.return_value = torch.tensor([0., 0., 0., 0., 0.])
            sanitizer._noise2_distributions.sample.return_value = torch.tensor([0., 0., 0., 0., 0.])
            sanitizer._noise3_distributions.sample.return_value = torch.tensor([0., 0., 0., 0., 0.])
            randn_like.return_value = torch.tensor([0., 0., 0., 0., 0.])
            sanitizer._base_params = [torch.zeros(5)]

            dest_model = torch.nn.Module()
            dest_model.layer1 = torch.nn.Parameter(torch.tensor([1., 1., 3., 4., 5.]))

            sanitizer.sanitize(dest_model)

            result_params = [*dest_model.parameters()][0]
            torch_testing.assert_equal(result_params, torch.tensor([0., 0., 3., 4., 5.]))

    def test_pick_multiple(self):
        with patch('torch.randn_like') as randn_like:
            do_nothing = lambda: None
            sanitizer = ModelSanitizer(1, 1, 1, 1, 1, 1, 1, 1)
            sanitizer.sanitize_init = Mock(side_effect=do_nothing)
            sanitizer.model_size = 5
            sanitizer.release_size = 2.5
            sanitizer.epochs = 1
            sanitizer.gamma = 10000
            sanitizer.tau = 1.5
            sanitizer._noise1_distributions = Mock()
            sanitizer._noise2_distributions = Mock()
            sanitizer._noise3_distributions = Mock()
            sanitizer._noise1_distributions.sample.side_effect = [
                torch.tensor([1., 0., 0., 0., 0.]),
                torch.tensor([1., 1., 0., 0., 0.]),
                torch.tensor([1., 1., 1., 0., 0.]),
            ]
            sanitizer._noise2_distributions.sample.return_value = torch.tensor([0., 0., 0., 0., 0.])
            sanitizer._noise3_distributions.sample.return_value = torch.tensor([0., 0., 0., 0., 0.])
            randn_like.return_value = torch.tensor([0., 0., 0., 0., 0.,])
            sanitizer._base_params = [
                torch.zeros(5)
            ]

            dest_model = torch.nn.Module()
            dest_model.layer1 = torch.nn.Parameter(torch.tensor([1., 1., 1., 1., 1.,]))

            sanitizer.sanitize(dest_model)

            result_params = [*dest_model.parameters()][0]
            torch_testing.assert_equal(result_params, torch.tensor([1., 1., 1., 0, 0]))

if __name__ == '__main__':
    unittest.main()
