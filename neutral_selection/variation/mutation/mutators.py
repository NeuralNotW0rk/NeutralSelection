import copy
import importlib
import random
import sys
from typing import Any, Callable


def gaussian_noise_mutator(std: float, mean: float = 0.0) -> Callable[[Any], Any]:
    """
    Returns a mutator function that adds Gaussian noise to numeric values.
    Supports floats/ints, numpy arrays, and torch Tensors dynamically.
    """
    if not isinstance(std, (int, float)):
        raise TypeError("std must be a float or int.")
    if not isinstance(mean, (int, float)):
        raise TypeError("mean must be a float or int.")

    def mutate_fn(val: Any) -> Any:
        # Check for torch Tensor
        if hasattr(val, "device") and hasattr(val, "dtype") and hasattr(val, "clone"):
            try:
                torch = sys.modules.get("torch") or importlib.import_module("torch")
                return val + (torch.randn_like(val) * std + mean)
            except ModuleNotFoundError:
                pass

        # Check for numpy array
        elif hasattr(val, "shape") and hasattr(val, "copy"):
            try:
                np = sys.modules.get("numpy") or importlib.import_module("numpy")
                return val + np.random.normal(mean, std, size=val.shape)
            except ModuleNotFoundError:
                pass

        # Fallback to python floats/ints
        elif isinstance(val, (int, float)):
            return val + random.gauss(mean, std)

        return val

    return mutate_fn


def bit_flip_mutator(prob: float = 0.5) -> Callable[[bool], bool]:
    """
    Returns a mutator function that flips a boolean value with a given probability.
    """
    if not isinstance(prob, (int, float)):
        raise TypeError("prob must be a float or int.")
    if not (0.0 <= prob <= 1.0):
        raise ValueError("prob must be between 0.0 and 1.0 inclusive.")

    def mutate_fn(val: bool) -> bool:
        if not isinstance(val, bool):
            raise TypeError("bit_flip_mutator only supports boolean values.")
        return not val if random.random() < prob else val

    return mutate_fn


def attribute_mutator(mutators: dict[str, Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """
    Returns a mutator function that applies specified mutator functions to
    particular attributes of an object dynamically.
    """
    if not isinstance(mutators, dict):
        raise TypeError("mutators must be a dictionary.")

    def mutate_fn(obj: Any) -> Any:
        new_obj = copy.deepcopy(obj)
        for attr_name, mutator in mutators.items():
            if hasattr(new_obj, attr_name):
                val = getattr(new_obj, attr_name)
                setattr(new_obj, attr_name, mutator(val))
        return new_obj

    return mutate_fn
