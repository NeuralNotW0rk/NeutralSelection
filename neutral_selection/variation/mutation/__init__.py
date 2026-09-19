from .base import (
    MutationStrategy,
    mutate,
)
from .real_mutation import (
    GaussianMutation,
    UniformRealMutation,
    PolynomialMutation,
    CauchyMutation,
)
from .binary_mutation import (
    BitFlipMutation,
    BoundaryMutation,
)
from .sequence_mutation import (
    InversionMutation,
    SwapMutation,
    ScrambleMutation,
    InsertionMutation,
    TranspositionMutation,
    DuplicationMutation,
    DeletionMutation,
)
from .uniform_mutation import UniformMutation
from .mutators import (
    gaussian_noise_mutator,
    bit_flip_mutator,
    attribute_mutator,
)

# Alias HierarchicalMutation to InversionMutation (sequence mutations natively support multi-scale hierarchies)
HierarchicalMutation = InversionMutation

__all__ = [
    "MutationStrategy",
    "mutate",
    "GaussianMutation",
    "UniformRealMutation",
    "PolynomialMutation",
    "CauchyMutation",
    "BitFlipMutation",
    "BoundaryMutation",
    "InversionMutation",
    "SwapMutation",
    "ScrambleMutation",
    "InsertionMutation",
    "TranspositionMutation",
    "DuplicationMutation",
    "DeletionMutation",
    "UniformMutation",
    "HierarchicalMutation",
    "gaussian_noise_mutator",
    "bit_flip_mutator",
    "attribute_mutator",
]
