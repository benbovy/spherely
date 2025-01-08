.. _api:

API reference
=============

.. currentmodule:: spherely

.. _api_properties:

Geography properties
--------------------

Functions that provide access to properties of :py:class:`~spherely.Geography`
objects without side-effects (except for ``prepare`` and ``destroy_prepared``).

.. autosummary::
   :toctree: _api_generated/

   GeographyType
   is_geography
   get_dimensions
   get_type_id
   get_x
   get_y
   is_prepared
   prepare
   destroy_prepared

.. _api_creation:

Geography creation
------------------

Functions that build new :py:class:`~spherely.Geography` objects from
coordinates or existing geographies.

.. autosummary::
   :toctree: _api_generated/

   create_point
   create_multipoint
   create_linestring
   create_multilinestring
   create_polygon
   create_multipolygon
   create_collection

.. _api_io:

Input/Output
------------

Functions that convert :py:class:`~spherely.Geography` objects to/from an
external format such as `WKT <https://en.wikipedia.org/wiki/Well-known_text>`_.

.. autosummary::
   :toctree: _api_generated/

   from_wkt
   to_wkt
   from_wkb
   to_wkb
   from_geoarrow
   to_geoarrow

.. _api_measurement:

Measurement
-----------

Functions that compute measurements of one or more geographies.

.. autosummary::
   :toctree: _api_generated/

   area
   distance
   length
   perimeter

.. _api_predicates:

Predicates
----------

Functions that return ``True`` or ``False`` for some spatial relationship
between two geographies.

.. autosummary::
   :toctree: _api_generated/

   equals
   intersects
   touches
   contains
   within
   disjoint
   covers
   covered_by

.. _api_overlays:

Overlays (boolean operations)
-----------------------------

Functions that generate a new geography based on the combination of two
geographies.

.. autosummary::
   :toctree: _api_generated/

   union
   intersection
   difference
   symmetric_difference

.. _api_constructive_ops:

Constructive operations
-----------------------

Functions that generate a new geography based on input.

.. autosummary::
   :toctree: _api_generated/

   centroid
   boundary
   convex_hull
