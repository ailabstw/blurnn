from ._util import combine_iterators
import functools
import copy
import torch
import random

class ModelSanitizer:
    def __init__(self, eps1, eps3, epochs, release_proportion=0.4, eps2=None, sensitivity=1, gamma=5e-1, tau=1e-1):
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.sensitivity = sensitivity
        self.gamma = gamma
        self.tau = tau
        self.release_proportion = release_proportion
        self.epochs = epochs
        self.model_size = None
        self.release_size = None
        self._base_params = None
        self._noise1_distributions = None
        self._noise2_distributions = None
        self._noise3_distributions = None

    @property
    def base_model(self):
        pass

    @base_model.setter
    def base_model(self, base_model):
        self._base_params = [*map(
            lambda param: copy.deepcopy(param).to(device=param.device),
            base_model.parameters()
        )] # Type: list Parameter

    def sanitize_init(self):
        self.model_size = sum(map(lambda layer: layer.data.numel(), self._base_params))
        self.release_size = self.model_size * self.release_proportion
        self.eps2 = self.eps2 if self.eps2 else ((2 * self.release_size * self.sensitivity) ** (2/3)) * self.eps1

        device = self._base_params[0].data.device
        tensor_zero = torch.tensor(0., device=device)
        self._noise1_distributions = torch.distributions.Laplace(tensor_zero, 2 * self.release_proportion * self.sensitivity / self.eps1)
        self._noise2_distributions = torch.distributions.Laplace(tensor_zero, self.sensitivity / self.eps2)
        self._noise3_distributions = torch.distributions.Laplace(tensor_zero, self.release_proportion * self.sensitivity / self.eps3)

    def sanitize(self, dest_model):
        self.sanitize_init()

        delta_weights = torch.cat([*map(
            lambda _: _[1].add(-1, _[0]).div_(self.epochs).reshape(-1),
            combine_iterators(self._base_params, dest_model.parameters())
        )])
        noise1_generate, noise2_generate, noise3_generate = map(
            lambda distribution:
                lambda:
                    distribution.sample(sample_shape=delta_weights.size()),
            [self._noise1_distributions, self._noise2_distributions, self._noise3_distributions]
        )

        release_weights = torch.zeros(delta_weights.size(), device=delta_weights.device)
        join_weights_size = 0
        while join_weights_size < self.release_size:
            noisy_taus = self.tau + noise2_generate()
            noisy_weights = delta_weights.clamp(max=self.gamma).abs() + noise1_generate()

            candidate_weights = (delta_weights + noise3_generate()).clamp(max=self.gamma, min=-self.gamma)
            candidate_weights[noisy_weights < noisy_taus] = 0
            join_weights = torch.where(release_weights > 0, release_weights, candidate_weights)
            join_weights_size = join_weights.nonzero().numel()

        candidate_weights[release_weights != 0] = 0
        try:
            candidate_weights[
                torch.randn_like(candidate_weights) > ((join_weights_size - self.release_size)/candidate_weights.nonzero().numel())
            ] = 0
        except ZeroDivisionError:
            pass

        release_weights = list((release_weights + candidate_weights).split(
            tuple([*map(
                lambda layer: layer.numel(),
                self._base_params)
        ])))
        for i, release_weight in enumerate(release_weights):
            release_weights[i] = release_weight.reshape(self._base_params[i].size())

        # reconstruct model
        for base_param, delta_weight_head, dest_param in combine_iterators(self._base_params, release_weights, dest_model.parameters()):
            dest_param.data = base_param.data.add_(self.epochs, delta_weight_head)