from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    List,
    Literal,
    Tuple,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

class Geography:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def nshape(self) -> int: ...

class GeographyType:
    __members__: ClassVar[dict] = ...  # read-only
    LINESTRING: ClassVar[GeographyType] = ...
    NONE: ClassVar[GeographyType] = ...
    POINT: ClassVar[GeographyType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class LineString(Geography):
    def __init__(self, arg0: List[Tuple[float, float]]) -> None: ...

class Point(Geography):
    def __init__(self, arg0: float, arg1: float) -> None: ...

# Numpy-like vectorized (universal) functions

_NameType = TypeVar("_NameType", bound=str)
_ScalarReturnType = TypeVar("_ScalarReturnType", bound=Any)
_ArrayReturnDType = TypeVar("_ArrayReturnDType", bound=Any)

class _VFunc_Nin1_Nout1(Generic[_NameType, _ScalarReturnType, _ArrayReturnDType]):
    @property
    def __name__(self) -> _NameType: ...
    @overload
    def __call__(self, geography: Geography) -> _ScalarReturnType: ...
    @overload
    def __call__(self, geography: npt.ArrayLike) -> npt.NDArray[_ArrayReturnDType]: ...
    @overload
    def __call__(
        self, geography: npt.NDArray[Any]
    ) -> npt.NDArray[_ArrayReturnDType]: ...

class _VFunc_Nin2_Nout1(Generic[_NameType, _ScalarReturnType, _ArrayReturnDType]):
    @property
    def __name__(self) -> _NameType: ...
    @overload
    def __call__(self, a: Geography, b: Geography) -> _ScalarReturnType: ...
    @overload
    def __call__(
        self, a: npt.ArrayLike, b: npt.ArrayLike
    ) -> npt.NDArray[_ArrayReturnDType]: ...
    @overload
    def __call__(
        self, a: Geography, b: npt.ArrayLike
    ) -> npt.NDArray[_ArrayReturnDType]: ...
    @overload
    def __call__(
        self, a: npt.ArrayLike, b: Geography
    ) -> npt.NDArray[_ArrayReturnDType]: ...

# Geography properties

get_dimensions: _VFunc_Nin1_Nout1[Literal["get_dimensions"], Geography, Any]
get_type_id: _VFunc_Nin1_Nout1[Literal["get_type_id"], int, np.int8]

# Geography utils

is_geography: _VFunc_Nin1_Nout1[Literal["is_geography"], bool, bool]
is_prepared: _VFunc_Nin1_Nout1[Literal["is_prepared"], bool, bool]
prepare: _VFunc_Nin1_Nout1[Literal["prepare"], Geography, Any]
destroy_prepared: _VFunc_Nin1_Nout1[Literal["destroy_prepared"], Geography, Any]

# predicates

intersects: _VFunc_Nin2_Nout1[Literal["intersects"], bool, bool]
equals: _VFunc_Nin2_Nout1[Literal["intersects"], bool, bool]

# temp (remove)

def create(arg0: Iterable[float], arg1: Iterable[float]) -> npt.NDArray[Any]: ...
def nshape(arg0: npt.NDArray[Any]) -> npt.NDArray[np.int32]: ...
