from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

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

# Projection class

class Projection:
    @staticmethod
    def lnglat() -> Projection: ...
    @staticmethod
    def speudo_mercator() -> Projection: ...
    @staticmethod
    def orthographic(longitude: float, latitude: float) -> Projection: ...

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

class _VFunc_Nin2optradius_Nout1(
    Generic[_NameType, _ScalarReturnType, _ArrayReturnDType]
):
    @property
    def __name__(self) -> _NameType: ...
    @overload
    def __call__(
        self, a: Geography, b: Geography, radius: float = ...
    ) -> _ScalarReturnType: ...
    @overload
    def __call__(
        self, a: npt.ArrayLike, b: npt.ArrayLike, radius: float = ...
    ) -> npt.NDArray[_ArrayReturnDType]: ...
    @overload
    def __call__(
        self, a: Geography, b: npt.ArrayLike, radius: float = ...
    ) -> npt.NDArray[_ArrayReturnDType]: ...
    @overload
    def __call__(
        self, a: npt.ArrayLike, b: Geography, radius: float = ...
    ) -> npt.NDArray[_ArrayReturnDType]: ...

class _VFunc_Nin1optradius_Nout1(
    Generic[_NameType, _ScalarReturnType, _ArrayReturnDType]
):
    @property
    def __name__(self) -> _NameType: ...
    @overload
    def __call__(self, a: Geography, radius: float = ...) -> _ScalarReturnType: ...
    @overload
    def __call__(
        self, a: npt.ArrayLike, radius: float = ...
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
is_empty: _VFunc_Nin1_Nout1[Literal["is_empty"], bool, bool]

# predicates

intersects: _VFunc_Nin2_Nout1[Literal["intersects"], bool, bool]
equals: _VFunc_Nin2_Nout1[Literal["intersects"], bool, bool]
contains: _VFunc_Nin2_Nout1[Literal["contains"], bool, bool]
within: _VFunc_Nin2_Nout1[Literal["within"], bool, bool]
disjoint: _VFunc_Nin2_Nout1[Literal["disjoint"], bool, bool]
touches: _VFunc_Nin2_Nout1[Literal["touches"], bool, bool]
covers: _VFunc_Nin2_Nout1[Literal["covers"], bool, bool]
covered_by: _VFunc_Nin2_Nout1[Literal["covered_by"], bool, bool]

# geography accessors

centroid: _VFunc_Nin1_Nout1[Literal["centroid"], PointGeography, PointGeography]
boundary: _VFunc_Nin1_Nout1[Literal["boundary"], Geography, Geography]
convex_hull: _VFunc_Nin1_Nout1[
    Literal["convex_hull"], PolygonGeography, PolygonGeography
]
distance: _VFunc_Nin2optradius_Nout1[Literal["distance"], float, float]
area: _VFunc_Nin1optradius_Nout1[Literal["area"], float, float]
length: _VFunc_Nin1optradius_Nout1[Literal["length"], float, float]
perimeter: _VFunc_Nin1optradius_Nout1[Literal["perimeter"], float, float]

# io functions

to_wkt: _VFunc_Nin1_Nout1[Literal["to_wkt"], str, object]
to_wkb: _VFunc_Nin1_Nout1[Literal["to_wkb"], bytes, object]

def from_wkt(
    a: Iterable[str],
    oriented: bool = False,
    planar: bool = False,
    tessellate_tolerance: float = 100.0,
) -> npt.NDArray[Any]: ...
def from_wkb(
    a: Iterable[bytes],
    oriented: bool = False,
    planar: bool = False,
    tessellate_tolerance: float = 100.0,
) -> npt.NDArray[Any]: ...

class ArrowSchemaExportable(Protocol):
    def __arrow_c_schema__(self) -> object: ...

class ArrowArrayExportable(Protocol):
    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> Tuple[object, object]: ...

def to_geoarrow(
    input: npt.ArrayLike,
    /,
    *,
    output_schema: ArrowSchemaExportable | None = None,
    projection: Projection = Projection.lnglat(),
    planar: bool = False,
    tessellate_tolerance: float = 100.0,
    precision: int = 6,
) -> ArrowArrayExportable: ...
def from_geoarrow(
    input: ArrowArrayExportable,
    /,
    *,
    oriented: bool = False,
    planar: bool = False,
    tessellate_tolerance: float = 100.0,
    projection: Projection = Projection.lnglat(),
    geometry_encoding: str | None = None,
) -> npt.NDArray[Any]: ...
