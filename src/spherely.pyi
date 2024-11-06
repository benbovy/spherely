import sys
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    overload,
)

try:
    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack
except ImportError:
    if TYPE_CHECKING:
        raise
    else:
        Unpack: Any = None

import numpy as np
import numpy.typing as npt

__version__: str = ...
__s2geography_version__: str = ...
EARTH_RADIUS_METERS: float = ...

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
    POLYGON: ClassVar[GeographyType] = ...
    MULTIPOLYGON: ClassVar[GeographyType] = ...
    MULTIPOINT: ClassVar[GeographyType] = ...
    MULTILINESTRING: ClassVar[GeographyType] = ...
    GEOMETRYCOLLECTION: ClassVar[GeographyType] = ...
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

# Annotated type aliases

PointGeography = Annotated[Geography, GeographyType.POINT]
LineStringGeography = Annotated[Geography, GeographyType.LINESTRING]
PolygonGeography = Annotated[Geography, GeographyType.POLYGON]
MultiPointGeography = Annotated[Geography, GeographyType.MULTIPOINT]
MultiLineStringGeography = Annotated[Geography, GeographyType.MULTILINESTRING]
MultiPolygonGeography = Annotated[Geography, GeographyType.MULTIPOLYGON]
GeometryCollection = Annotated[Geography, GeographyType.GEOMETRYCOLLECTION]

# Numpy-like vectorized (universal) functions

_NameType = TypeVar("_NameType", bound=str)
_ScalarReturnType = TypeVar("_ScalarReturnType", bound=Any)
_ArrayReturnDType = TypeVar("_ArrayReturnDType", bound=Any)

_Empty_Kwargs = TypedDict("_Empty_Kwargs", {})
_KwargsType = TypeVar("_KwargsType", bound=TypedDict, default=_Empty_Kwargs)

class _VFunc_Nin1_Nout1(
    Generic[_NameType, _ScalarReturnType, _ArrayReturnDType, _KwargsType]
):
    @property
    def __name__(self) -> _NameType: ...
    @overload
    def __call__(
        self, geography: Geography, **kwargs: Unpack[_KwargsType]
    ) -> _ScalarReturnType: ...
    @overload
    def __call__(
        self, geography: npt.ArrayLike, **kwargs: Unpack[_KwargsType]
    ) -> npt.NDArray[_ArrayReturnDType]: ...

class _VFunc_Nin2_Nout1(
    Generic[_NameType, _ScalarReturnType, _ArrayReturnDType, _KwargsType]
):
    @property
    def __name__(self) -> _NameType: ...
    @overload
    def __call__(
        self, a: Geography, b: Geography, **kwargs: Unpack[_KwargsType]
    ) -> _ScalarReturnType: ...
    @overload
    def __call__(
        self, a: npt.ArrayLike, b: npt.ArrayLike, **kwargs: Unpack[_KwargsType]
    ) -> npt.NDArray[_ArrayReturnDType]: ...
    @overload
    def __call__(
        self, a: Geography, b: npt.ArrayLike, **kwargs: Unpack[_KwargsType]
    ) -> npt.NDArray[_ArrayReturnDType]: ...
    @overload
    def __call__(
        self, a: npt.ArrayLike, b: Geography, **kwargs: Unpack[_KwargsType]
    ) -> npt.NDArray[_ArrayReturnDType]: ...

# Geography properties

get_dimensions: _VFunc_Nin1_Nout1[Literal["get_dimensions"], Geography, Any]
get_type_id: _VFunc_Nin1_Nout1[Literal["get_type_id"], int, np.int8]

# Geography creation (scalar)

def point(
    longitude: float | None = None, latitude: float | None = None
) -> Geography: ...
def multipoint(
    points: Iterable[Sequence[float]] | Iterable[PointGeography],
) -> MultiPointGeography: ...
def linestring(
    vertices: Iterable[Sequence[float]] | Iterable[PointGeography] | None = None,
) -> LineStringGeography: ...
def multilinestring(
    vertices: (
        Iterable[Iterable[Sequence[float]]]
        | Iterable[Iterable[PointGeography]]
        | Iterable[LineStringGeography]
    ),
) -> MultiLineStringGeography: ...
@overload
def polygon(
    shell: None = None,
    holes: None = None,
) -> PolygonGeography: ...
@overload
def polygon(
    shell: Iterable[Sequence[float]],
    holes: Iterable[Iterable[Sequence[float]]] | None = None,
) -> PolygonGeography: ...
@overload
def polygon(
    shell: Iterable[PointGeography],
    holes: Iterable[Iterable[PointGeography]] | None = None,
) -> PolygonGeography: ...
def multipolygon(polygons: Iterable[PolygonGeography]) -> MultiPolygonGeography: ...
def collection(geographies: Iterable[Geography]) -> GeometryCollection: ...

# Geography creation (vectorized)

@overload
def points(
    longitude: npt.ArrayLike, latitude: npt.ArrayLike
) -> npt.NDArray[np.object_]: ...
@overload
def points(longitude: float, latitude: float) -> PointGeography: ...  # type: ignore[misc]

# Geography utils

is_geography: _VFunc_Nin1_Nout1[Literal["is_geography"], bool, bool]
is_prepared: _VFunc_Nin1_Nout1[Literal["is_prepared"], bool, bool]
prepare: _VFunc_Nin1_Nout1[Literal["prepare"], Geography, Any]
destroy_prepared: _VFunc_Nin1_Nout1[Literal["destroy_prepared"], Geography, Any]

# predicates

intersects: _VFunc_Nin2_Nout1[Literal["intersects"], bool, bool]
equals: _VFunc_Nin2_Nout1[Literal["intersects"], bool, bool]
contains: _VFunc_Nin2_Nout1[Literal["contains"], bool, bool]
within: _VFunc_Nin2_Nout1[Literal["within"], bool, bool]
disjoint: _VFunc_Nin2_Nout1[Literal["disjoint"], bool, bool]

# geography accessors

centroid: _VFunc_Nin1_Nout1[Literal["centroid"], PointGeography, PointGeography]
boundary: _VFunc_Nin1_Nout1[Literal["boundary"], Geography, Geography]
convex_hull: _VFunc_Nin1_Nout1[
    Literal["convex_hull"], PolygonGeography, PolygonGeography
]

class DistanceKwargs(TypedDict):
    radius: float

distance: _VFunc_Nin2_Nout1[Literal["distance"], float, float, DistanceKwargs]

# io functions

to_wkt: _VFunc_Nin1_Nout1[Literal["to_wkt"], str, object]

def from_wkt(
    a: Iterable[str],
    oriented: bool = False,
    planar: bool = False,
    tessellate_tolerance: float = 100.0,
) -> npt.NDArray[Any]: ...

class ArrowArrayExportable(Protocol):
    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> Tuple[object, object]: ...

def from_geoarrow(
    input: ArrowArrayExportable,
    /,
    *,
    oriented: bool = False,
    planar: bool = False,
    tessellate_tolerance: float = 100.0,
    geometry_encoding: str | None = None,
) -> npt.NDArray[Any]: ...
