from typing import TypeVar, Callable, Generic, Any, Union, overload, KeysView, ItemsView, Iterator

T = TypeVar("T")  # Generic item payload type (e.g., float, dataclass, tensor wrapper)
K = TypeVar("K")  # Key type for segments (e.g., str, int)

CrossoverFn = Callable[[T, T, float], T]  # (parent_a, parent_b, weight) -> child_item
MutationFn = Callable[[T], T]             # (item) -> mutated_item


class ContiguousArrayGenome(Generic[T]):
    """A 1D sequence container representing a continuous array of generic items T."""

    def __init__(self, items: list[T]) -> None:
        if not isinstance(items, list):
            raise TypeError("items must be a list")
        self._items: list[T] = list(items)

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, list[T]]:
        if not isinstance(index, (int, slice)):
            raise TypeError(f"Invalid index type: {type(index).__name__}")
        return self._items[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def map(self, fn: Callable[[T], T]) -> "ContiguousArrayGenome[T]":
        """Maps a function over all elements of the genome, returning a new genome."""
        if fn is None:
            raise ValueError("fn must be provided and cannot be None.")
        if not callable(fn):
            raise TypeError("fn must be a callable.")
        return ContiguousArrayGenome([fn(item) for item in self._items])

    def zip_map(
        self,
        other: "ContiguousArrayGenome[T]",
        fn: Callable[[T, T, float], T],
        blend_factor: float
    ) -> "ContiguousArrayGenome[T]":
        """Zips this genome with other and maps a blending function over their elements."""
        if not isinstance(other, ContiguousArrayGenome):
            raise TypeError("other must be an instance of ContiguousArrayGenome")
        if fn is None:
            raise ValueError("fn must be provided and cannot be None.")
        if not callable(fn):
            raise TypeError("fn must be a callable.")
        if not isinstance(blend_factor, (int, float)):
            raise TypeError("blend_factor must be a float or int.")
        if len(self) != len(other):
            raise ValueError("Genomes must have the same length for zip_map.")

        return ContiguousArrayGenome([
            fn(self._items[i], other._items[i], float(blend_factor))
            for i in range(len(self))
        ])

    def crossover(
        self,
        other: "ContiguousArrayGenome[T]",
        cut_points: list[int]
    ) -> tuple["ContiguousArrayGenome[T]", "ContiguousArrayGenome[T]"]:
        """Structural N-point crossover slicing across sequence boundaries."""
        if not isinstance(other, ContiguousArrayGenome):
            raise TypeError("other must be an instance of ContiguousArrayGenome")
        if not isinstance(cut_points, list):
            raise TypeError("cut_points must be a list")
        if any(not isinstance(c, int) for c in cut_points):
            raise TypeError("All cut points must be integers")
        if any(c < 0 for c in cut_points):
            raise ValueError("Cut points must be non-negative integers")

        sorted_cuts = sorted(list(set(cut_points)))

        child1_items: list[T] = []
        child2_items: list[T] = []

        last_cut = 0
        use_self = True

        for cut in sorted_cuts:
            if use_self:
                child1_items.extend(self._items[last_cut:cut])
                child2_items.extend(other._items[last_cut:cut])
            else:
                child1_items.extend(other._items[last_cut:cut])
                child2_items.extend(self._items[last_cut:cut])
            last_cut = cut
            use_self = not use_self

        # Append the final segment
        if use_self:
            child1_items.extend(self._items[last_cut:])
            child2_items.extend(other._items[last_cut:])
        else:
            child1_items.extend(other._items[last_cut:])
            child2_items.extend(self._items[last_cut:])

        return (ContiguousArrayGenome(child1_items), ContiguousArrayGenome(child2_items))


class SegmentedGenome(Generic[K, T]):
    """A key-addressable container mapping segment identifiers K to individual ContiguousArrayGenome[T] instances."""

    def __init__(self, segments: dict[K, ContiguousArrayGenome[T]]) -> None:
        if not isinstance(segments, dict):
            raise TypeError("segments must be a dict mapping K to ContiguousArrayGenome[T]")
        for k, v in segments.items():
            if not isinstance(v, ContiguousArrayGenome):
                raise TypeError(f"Segment value for key '{k}' must be a ContiguousArrayGenome.")
        self._segments: dict[K, ContiguousArrayGenome[T]] = dict(segments)

    def __getitem__(self, key: K) -> ContiguousArrayGenome[T]:
        # Dict-like navigation: raises KeyError if key is not found
        return self._segments[key]

    def keys(self) -> KeysView[K]:
        return self._segments.keys()

    def items(self) -> ItemsView[K, ContiguousArrayGenome[T]]:
        return self._segments.items()

    def map(self, fn: Callable[[T], T]) -> "SegmentedGenome[K, T]":
        """Maps a function over all segment elements recursively."""
        if fn is None:
            raise ValueError("fn must be provided and cannot be None.")
        if not callable(fn):
            raise TypeError("fn must be a callable.")

        return SegmentedGenome({
            k: seg.map(fn) for k, seg in self._segments.items()
        })

    def zip_map(
        self,
        other: "SegmentedGenome[K, T]",
        fn: Callable[[T, T, float], T],
        blend_factor: float
    ) -> "SegmentedGenome[K, T]":
        """Zips this segmented genome with another and blends segment elements recursively."""
        if not isinstance(other, SegmentedGenome):
            raise TypeError("other must be an instance of SegmentedGenome")
        if fn is None:
            raise ValueError("fn must be provided and cannot be None.")
        if not callable(fn):
            raise TypeError("fn must be a callable.")
        if not isinstance(blend_factor, (int, float)):
            raise TypeError("blend_factor must be a float or int.")
        if self._segments.keys() != other._segments.keys():
            raise ValueError("SegmentedGenomes must have the same set of keys for zip_map.")

        return SegmentedGenome({
            k: self._segments[k].zip_map(other._segments[k], fn, float(blend_factor))
            for k in self._segments.keys()
        })

    def crossover_segments(
        self,
        other: "SegmentedGenome[K, T]",
        active_keys: set[K]
    ) -> "SegmentedGenome[K, T]":
        """Inter-segment crossover. Swaps entire ContiguousArrayGenome instances wholesale."""
        if not isinstance(other, SegmentedGenome):
            raise TypeError("other must be an instance of SegmentedGenome")
        if not isinstance(active_keys, set):
            raise TypeError("active_keys must be a set")
        if self._segments.keys() != other._segments.keys():
            raise ValueError("SegmentedGenomes must have the same set of keys for crossover.")

        for key in active_keys:
            if key not in self._segments:
                raise KeyError(f"Active key '{key}' not found in genome segments.")

        new_segments: dict[K, ContiguousArrayGenome[T]] = {}
        for k in self._segments.keys():
            if k in active_keys:
                new_segments[k] = other._segments[k]
            else:
                new_segments[k] = self._segments[k]

        return SegmentedGenome(new_segments)
