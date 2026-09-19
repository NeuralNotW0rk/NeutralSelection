from __future__ import annotations

import dataclasses
from dataclasses import dataclass
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
                child_leaves, child_tree = flatten_hierarchy(
                    val,
                    max_depth=max_depth,
                    atomic_types=atomic_types,
                    current_depth=current_depth + 1,
                )
                all_leaves.extend(child_leaves)
                children_defs.append(child_tree.root_def)
                field_is_child[field.name] = True
            else:
                field_metadata[field.name] = val
                field_is_child[field.name] = False

        node_def = NodeDef(
            node_type=type(obj),
            metadata={
                "is_dataclass": True,
                "field_metadata": field_metadata,
                "field_is_child": field_is_child,
            },
            num_leaves=len(all_leaves),
            children_defs=tuple(children_defs),
        )
        return all_leaves, TreeDef(root_def=node_def, total_leaves=len(all_leaves))

    # 3. Check duck-typed Tensor / ndarray
    if _is_duck_tensor(obj):
        # Convert tensor to flat 1D list of elements
        if hasattr(obj, "detach"):
            flat_items = obj.detach().cpu().reshape(-1).tolist()
        elif hasattr(obj, "tolist"):
            flat_items = obj.reshape(-1).tolist()
        else:
            flat_items = list(obj)

        node_def = NodeDef(
            node_type=type(obj),
            metadata={
                "is_tensor": True,
                "shape": tuple(obj.shape),
                "dtype": getattr(obj, "dtype", None),
                "device": getattr(obj, "device", None),
            },
            num_leaves=len(flat_items),
            children_defs=(),
        )
        return flat_items, TreeDef(root_def=node_def, total_leaves=len(flat_items))

    # 4. Check Genome / Segment
    if isinstance(obj, Genome):
        children_defs = []
        all_leaves = []
        for child in obj:
            child_leaves, child_tree = flatten_hierarchy(
                child,
                max_depth=max_depth,
                atomic_types=atomic_types,
                current_depth=current_depth + 1,
            )
            all_leaves.extend(child_leaves)
            children_defs.append(child_tree.root_def)

        meta: dict[str, Any] = {"is_genome": True}
        if isinstance(obj, Segment):
            meta["is_segment"] = True
            meta["key"] = obj.key

        node_def = NodeDef(
            node_type=type(obj),
            metadata=meta,
            num_leaves=len(all_leaves),
            children_defs=tuple(children_defs),
        )
        return all_leaves, TreeDef(root_def=node_def, total_leaves=len(all_leaves))

    # 5. Check standard sequence (list, tuple)
    if isinstance(obj, (list, tuple)):
        children_defs = []
        all_leaves = []
        for child in obj:
            child_leaves, child_tree = flatten_hierarchy(
                child,
                max_depth=max_depth,
                atomic_types=atomic_types,
                current_depth=current_depth + 1,
            )
            all_leaves.extend(child_leaves)
            children_defs.append(child_tree.root_def)

        node_def = NodeDef(
            node_type=type(obj),
            metadata={"is_sequence": True},
            num_leaves=len(all_leaves),
            children_defs=tuple(children_defs),
        )
        return all_leaves, TreeDef(root_def=node_def, total_leaves=len(all_leaves))

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
        leaves: Sequence of leaf elements matching the total leaf count of treedef.
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

            # Try PyTorch Tensor if torch is present in sys.modules or class name matches
            if node_def.node_type.__name__ == "Tensor":
                try:
                    import torch
                    return torch.tensor(leaf_slice, dtype=dtype, device=device).reshape(shape)
                except Exception:
                    pass

            # Try NumPy ndarray
            if "ndarray" in node_def.node_type.__name__:
                try:
                    import numpy as np
                    return np.array(leaf_slice, dtype=dtype).reshape(shape)
                except Exception:
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
