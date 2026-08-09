from typing import Any, Union, overload, Iterator, Callable

CrossoverFn = Callable[[Any, Any, float], Any]
MutationFn = Callable[[Any], Any]


class Genome:
    """A sequence container representing a genome or sub-genome."""

    def __init__(self, items: list) -> None:
        if not isinstance(items, list):
            raise TypeError("items must be a list")
        self._items: list = list(items)

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> list: ...

    def __getitem__(self, index: Union[int, slice]) -> Any:
        if not isinstance(index, (int, slice)):
            raise TypeError(f"Invalid index type: {type(index).__name__}")
        return self._items[index]

    def __iter__(self) -> Iterator:
        return iter(self._items)


class Segment(Genome):
    """A labeled sub-genome that groups a sequence of items under a name/key."""

    def __init__(self, key: Any, items: list) -> None:
        super().__init__(items)
        self.key: Any = key
