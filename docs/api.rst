.. _api:

API reference
=============

.. currentmodule:: spherely

.. _api_properties:

Geography properties
--------------------

.. autosummary::
   :toctree: _api_generated/

   GeographyType
   is_geography
   get_dimensions
   get_type_id

.. _api_creation:

Geography creation
------------------

.. autosummary::
   :toctree: _api_generated/

   point
   linestring
   multipoint
   multilinestring
   polygon
   collection
   is_prepared
   prepare
   destroy_prepared

.. _api_io:

Input/Output
------------

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

.. autosummary::
   :toctree: _api_generated/

   area
   distance

.. _api_predicates:

Predicates
----------

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

.. _api_constructive_ops:

Constructive operations
-----------------------

.. autosummary::
   :toctree: _api_generated/

   centroid
   boundary
   convex_hull
