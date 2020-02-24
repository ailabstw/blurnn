import functools
import itertools
import torch

def safe_apply(appliable, func, **param):
    return functools.partial(func, **param) if appliable(**param) else (lambda _: _)

def grad_clip(grad, norm_bound):
    if grad.norm() <= norm_bound:
        return grad
    return grad.div_(grad.norm()).mul_(norm_bound)

def grad_add_noise(grad, noise_scale):
    return grad.add_(noise_scale, torch.randn(list(grad.size()), device=grad.device))

clip_generator = functools.partial(safe_apply,
    appliable=(lambda norm_bound: norm_bound != 0),
    func=grad_clip,
)

add_noise_generator = functools.partial(safe_apply,
    appliable=(lambda noise_scale: noise_scale != 0),
    func=grad_add_noise,
)

def combine_iterators(*iterators):
    end = False
    # Type Casting
    iterators = [*map(iter, iterators)]
    while not end:
        end = True
        nexts = ()
        for iterator in iterators:
            try:
                nexts = nexts + tuple([iterator.__next__()])
                end = False
            except StopIteration:
                nexts = nexts + tuple([None])
        if not end:
            yield nexts