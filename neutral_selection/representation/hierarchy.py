from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import importlib
import sys
from typing import Any, Callable, Optional, Sequence, Union
from neutral_selection.representation.genome import Genome, Segment


@dataclass(frozen=True)
class NodeDef:
    """Metadata schema representing a node in the hierarchical genome tree."""
    node_type: type
    metadata: dict[str, Any]
    num_leaves: int
    children_defs: tuple["NodeDef", ...]


@dataclass(frozen=True)
class TreeDef:
    """Complete structural schema of a hierarchical genome tree."""
    root_def: NodeDef
    total_leaves: int


def _is_duck_tensor(obj: Any) -> bool:
    """Checks if an object is a tensor-like array (e.g., PyTorch Tensor or NumPy ndarray)."""
    return (
        hasattr(obj, "shape")
        and hasattr(obj, "reshape")
        and not isinstance(obj, (Genome, list, tuple))
    )


def _torch_module() -> Any:
    """Returns the torch module if it has been imported, without importing it."""
    return sys.modules.get("torch")


def is_tensor_strand(leaves: Any) -> bool:
    """Checks if a leaf sequence is a 1D torch tensor strand (from flatten_strand) rather than a list."""
    torch = _torch_module()
    return torch is not None and isinstance(leaves, torch.Tensor)


class _TensorRun:
    """Placeholder for a whole torch tensor's flattened elements, used while building a strand."""

    __slots__ = ("flat",)

    def __init__(self, flat: Any) -> None:
        self.flat = flat


def flatten_hierarchy(
    obj: Any,
    max_depth: Optional[int] = None,
    atomic_types: tuple[type, ...] = (),
    current_depth: int = 0,
) -> tuple[list[Any], TreeDef]:
    """
    Flattens a composite hierarchical structure (Genomes, Segments, Dataclasses, Tensors, Protocol objects)
    into a 1D continuous sequence of leaf elements alongside a structural TreeDef.

    Parameters:
        obj: The root hierarchical object to flatten.
        max_depth: Maximum recursion depth (None for full recursion to leaves, 0 for root leaf).
        atomic_types: Types explicitly treated as atomic/indivisible leaves.
        current_depth: Current recursion depth in the tree.

    Returns:
        A tuple of (leaf_items, treedef).
    """
    return _flatten(obj, max_depth, atomic_types, current_depth, tensor_runs=False)


def flatten_strand(
    obj: Any,
    max_depth: Optional[int] = None,
    atomic_types: tuple[type, ...] = (),
) -> tuple[Any, TreeDef]:
    """
    Flattens like flatten_hierarchy, but when every leaf comes from torch tensors sharing one dtype and
    device, returns the leaves as a single contiguous 1D tensor (a strand) instead of a list of Python
    scalars. Otherwise returns the same leaf list as flatten_hierarchy. unflatten_hierarchy accepts both.

    Parameters:
        obj: The root hierarchical object to flatten.
        max_depth: Maximum recursion depth (None for full recursion to leaves, 0 for root leaf).
        atomic_types: Types explicitly treated as atomic/indivisible leaves.

    Returns:
        A tuple of (strand_or_leaf_items, treedef).
    """
    items, treedef = _flatten(obj, max_depth, atomic_types, 0, tensor_runs=True)
    runs = [item.flat for item in items if isinstance(item, _TensorRun)]
    if runs and len(runs) == len(items):
        first = runs[0]
        if all(r.dtype == first.dtype and r.device == first.device for r in runs):
            torch = _torch_module()
            return (torch.cat(runs) if len(runs) > 1 else first.clone()), treedef

    leaves: list[Any] = []
    for item in items:
        if isinstance(item, _TensorRun):
            leaves.extend(item.flat.cpu().tolist())
        else:
            leaves.append(item)
    return leaves, treedef


def float_strand(strand: Any, op_name: str) -> Any:
    """
    Returns a tensor strand converted for value arithmetic: floating dtypes are promoted to at least
    float32, integer dtypes to float64. unflatten_hierarchy casts results back to each tensor's dtype.
    """
    torch = _torch_module()
    if strand.dtype == torch.bool or strand.is_complex():
        raise TypeError(f"{op_name} requires numeric genome elements, got {strand.dtype} tensors.")
    if strand.is_floating_point():
        return strand.to(torch.promote_types(strand.dtype, torch.float32))
    return strand.to(torch.float64)


def align_strands(leaves_a: Any, leaves_b: Any) -> tuple[Any, Any]:
    """
    Returns two leaf sequences in a common form: both tensor strands when they share dtype and device,
    otherwise both lists.
    """
    if is_tensor_strand(leaves_a) and is_tensor_strand(leaves_b):
        if leaves_a.dtype == leaves_b.dtype and leaves_a.device == leaves_b.device:
            return leaves_a, leaves_b
    as_list = lambda leaves: leaves.cpu().tolist() if is_tensor_strand(leaves) else leaves
    return as_list(leaves_a), as_list(leaves_b)


def concat_strands(parts: Sequence[Any]) -> Any:
    """Concatenates slices of one leaf sequence, keeping tensor strands as tensors."""
    if parts and is_tensor_strand(parts[0]):
        return _torch_module().cat(list(parts))
    items: list[Any] = []
    for part in parts:
        items.extend(part)
    return items


def _flatten(
    obj: Any,
    max_depth: Optional[int],
    atomic_types: tuple[type, ...],
    current_depth: int,
    tensor_runs: bool,
) -> tuple[list[Any], TreeDef]:
    """
    Implements flatten_hierarchy. With tensor_runs, each expanded torch tensor contributes a single
    _TensorRun item holding its flattened elements, while its TreeDef node still counts every element.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative or None")

    # 1. Check if explicitly marked atomic or max depth reached
    if (atomic_types and isinstance(obj, atomic_types)) or (
        max_depth is not None and current_depth >= max_depth
    ):
        node_def = NodeDef(
            node_type=type(obj),
            metadata={"is_leaf": True},
            num_leaves=1,
            children_defs=(),
        )
        return [obj], TreeDef(root_def=node_def, total_leaves=1)

    # 2. Check custom protocol: __hierarchical_flatten__
    if hasattr(obj, "__hierarchical_flatten__") and callable(obj.__hierarchical_flatten__):
        leaves, custom_meta = obj.__hierarchical_flatten__(
            max_depth=max_depth, current_depth=current_depth
        )
        if not isinstance(leaves, list):
            leaves = list(leaves)
        node_def = NodeDef(
            node_type=type(obj),
            metadata={"is_custom": True, "custom_meta": custom_meta},
            num_leaves=len(leaves),
            children_defs=(),
        )
        return leaves, TreeDef(root_def=node_def, total_leaves=len(leaves))

    # 3. Check Dataclass instances (e.g. PerturbationGene)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        children_defs = []
        all_leaves = []
        field_metadata: dict[str, Any] = {}
        field_is_child: dict[str, bool] = {}

        for field in dataclasses.fields(obj):
            val = getattr(obj, field.name)
            if (
                _is_duck_tensor(val)
                or isinstance(val, (Genome, list, tuple))
                or (dataclasses.is_dataclass(val) and not isinstance(val, type))
            ):
                child_leaves, child_tree = _flatten(
                    val,
                    max_depth=max_depth,
                    atomic_types=atomic_types,
                    current_depth=current_depth + 1,
                    tensor_runs=tensor_runs,
                )
                all_leaves.extend(child_leaves)
                children_defs.append(child_tree.root_def)
                field_is_child[field.name] = True
            else:
                field_metadata[field.name] = val
                field_is_child[field.name] = False

        # Counted from children, since a tensor run stands in for many leaves
        num_leaves = sum(child.num_leaves for child in children_defs)
        node_def = NodeDef(
            node_type=type(obj),
            metadata={
                "is_dataclass": True,
                "field_metadata": field_metadata,
                "field_is_child": field_is_child,
            },
            num_leaves=num_leaves,
            children_defs=tuple(children_defs),
        )
        return all_leaves, TreeDef(root_def=node_def, total_leaves=num_leaves)

    # 3. Check duck-typed Tensor / ndarray
    if _is_duck_tensor(obj):
        # Convert tensor to flat 1D list of elements (or a single run of them when building a strand)
        if tensor_runs and is_tensor_strand(obj):
            flat = obj.reshape(-1)
            flat_items = [_TensorRun(flat.detach() if flat.requires_grad else flat)]
            num_leaves = obj.numel()
        else:
            if hasattr(obj, "detach"):
                flat_items = obj.detach().cpu().reshape(-1).tolist()
            elif hasattr(obj, "tolist"):
                flat_items = obj.reshape(-1).tolist()
            else:
                flat_items = list(obj)
            num_leaves = len(flat_items)

        node_def = NodeDef(
            node_type=type(obj),
            metadata={
                "is_tensor": True,
                "shape": tuple(obj.shape),
                "dtype": getattr(obj, "dtype", None),
                "device": getattr(obj, "device", None),
            },
            num_leaves=num_leaves,
            children_defs=(),
        )
        return flat_items, TreeDef(root_def=node_def, total_leaves=num_leaves)

    # 4. Check Genome / Segment
    if isinstance(obj, Genome):
        children_defs = []
        all_leaves = []
        for child in obj:
            child_leaves, child_tree = _flatten(
                child,
                max_depth=max_depth,
                atomic_types=atomic_types,
                current_depth=current_depth + 1,
                tensor_runs=tensor_runs,
            )
            all_leaves.extend(child_leaves)
            children_defs.append(child_tree.root_def)

        num_leaves = sum(child.num_leaves for child in children_defs)
        meta: dict[str, Any] = {"is_genome": True}
        if isinstance(obj, Segment):
            meta["is_segment"] = True
            meta["key"] = obj.key

        node_def = NodeDef(
            node_type=type(obj),
            metadata=meta,
            num_leaves=num_leaves,
            children_defs=tuple(children_defs),
        )
        return all_leaves, TreeDef(root_def=node_def, total_leaves=num_leaves)

    # 5. Check standard sequence (list, tuple)
    if isinstance(obj, (list, tuple)):
        children_defs = []
        all_leaves = []
        for child in obj:
            child_leaves, child_tree = _flatten(
                child,
                max_depth=max_depth,
                atomic_types=atomic_types,
                current_depth=current_depth + 1,
                tensor_runs=tensor_runs,
            )
            all_leaves.extend(child_leaves)
            children_defs.append(child_tree.root_def)

        num_leaves = sum(child.num_leaves for child in children_defs)
        node_def = NodeDef(
            node_type=type(obj),
            metadata={"is_sequence": True},
            num_leaves=num_leaves,
            children_defs=tuple(children_defs),
        )
        return all_leaves, TreeDef(root_def=node_def, total_leaves=num_leaves)

    # 6. Fallback: atomic leaf
    node_def = NodeDef(
        node_type=type(obj),
        metadata={"is_leaf": True},
        num_leaves=1,
        children_defs=(),
    )
    return [obj], TreeDef(root_def=node_def, total_leaves=1)


def unflatten_hierarchy(leaves: Sequence[Any], treedef: TreeDef) -> Any:
    """
    Reconstructs the original hierarchical composite structure from a sequence of leaf elements
    and the structural TreeDef schema.

    Parameters:
        leaves: Sequence of leaf elements (or a 1D tensor strand from flatten_strand) matching the
            total leaf count of treedef.
        treedef: Structural TreeDef schema.

    Returns:
        Reconstructed hierarchical structure with original types, segments, shapes, and metadata.
    """
    is_flat_sequence = (
        treedef.root_def.metadata.get("is_genome")
        or treedef.root_def.metadata.get("is_sequence")
    ) and all(child.metadata.get("is_leaf") for child in treedef.root_def.children_defs)

    if len(leaves) != treedef.total_leaves and not is_flat_sequence:
        raise ValueError(
            f"Leaf count mismatch for unflattening: got {len(leaves)} leaves, expected {treedef.total_leaves}"
        )

    def _reconstruct_node(node_def: NodeDef, leaf_slice: Sequence[Any]) -> Any:
        meta = node_def.metadata

        # 1. Atomic Leaf
        if meta.get("is_leaf"):
            if len(leaf_slice) != 1:
                raise ValueError(f"Expected 1 leaf for leaf node, got {len(leaf_slice)}")
            return leaf_slice[0]

        # 2. Custom Protocol: __hierarchical_unflatten__
        if meta.get("is_custom"):
            custom_unflatten = getattr(node_def.node_type, "__hierarchical_unflatten__", None)
            if custom_unflatten is None or not callable(custom_unflatten):
                raise TypeError(f"Type {node_def.node_type} does not implement __hierarchical_unflatten__")
            return custom_unflatten(list(leaf_slice), meta["custom_meta"])

        # 3. Dataclass reconstruction
        if meta.get("is_dataclass"):
            field_metadata = meta["field_metadata"]
            field_is_child = meta["field_is_child"]
            kwargs = dict(field_metadata)
            offset = 0
            child_idx = 0
            for field_name, is_child in field_is_child.items():
                if is_child:
                    child_def = node_def.children_defs[child_idx]
                    child_slice = leaf_slice[offset : offset + child_def.num_leaves]
                    offset += child_def.num_leaves
                    child_idx += 1
                    kwargs[field_name] = _reconstruct_node(child_def, child_slice)
            return node_def.node_type(**kwargs)

        # 4. Duck-typed Tensor / ndarray
        if meta.get("is_tensor"):
            shape = meta["shape"]
            dtype = meta["dtype"]
            device = meta["device"]

            # Slice of a tensor strand: copy so tensors in the result never share storage with each other
            if is_tensor_strand(leaf_slice):
                return leaf_slice.reshape(shape).to(dtype=dtype, device=device, copy=True)

            # Try PyTorch Tensor if torch is present in sys.modules or class name matches
            if node_def.node_type.__name__ == "Tensor":
                try:
                    torch = sys.modules.get("torch") or importlib.import_module("torch")
                    return torch.tensor(leaf_slice, dtype=dtype, device=device).reshape(shape)
                except ModuleNotFoundError:
                    pass

            # Try NumPy ndarray
            if "ndarray" in node_def.node_type.__name__:
                try:
                    np = sys.modules.get("numpy") or importlib.import_module("numpy")
                    return np.array(leaf_slice, dtype=dtype).reshape(shape)
                except ModuleNotFoundError:
                    pass

            # Fallback: reshape if class callable
            return leaf_slice

        # 4. Genome / Segment
        if meta.get("is_genome"):
            if all(child.metadata.get("is_leaf") for child in node_def.children_defs):
                if meta.get("is_segment"):
                    key = meta["key"]
                    return node_def.node_type(key, list(leaf_slice))
                return node_def.node_type(list(leaf_slice))

            reconstructed_children = []
            offset = 0
            for child_def in node_def.children_defs:
                child_slice = leaf_slice[offset : offset + child_def.num_leaves]
                offset += child_def.num_leaves
                reconstructed_children.append(_reconstruct_node(child_def, child_slice))

            if meta.get("is_segment"):
                key = meta["key"]
                return node_def.node_type(key, reconstructed_children)
            return node_def.node_type(reconstructed_children)

        # 5. Standard Sequence (list, tuple)
        if meta.get("is_sequence"):
            if all(child.metadata.get("is_leaf") for child in node_def.children_defs):
                if issubclass(node_def.node_type, tuple):
                    return tuple(leaf_slice)
                return list(leaf_slice)

            reconstructed_children = []
            offset = 0
            for child_def in node_def.children_defs:
                child_slice = leaf_slice[offset : offset + child_def.num_leaves]
                offset += child_def.num_leaves
                reconstructed_children.append(_reconstruct_node(child_def, child_slice))

            if issubclass(node_def.node_type, tuple):
                return tuple(reconstructed_children)
            return list(reconstructed_children)

        # Fallback leaf
        return leaf_slice[0] if leaf_slice else None

    return _reconstruct_node(treedef.root_def, leaves)
