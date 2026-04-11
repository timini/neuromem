"""neuromem.storage — pluggable persistence backends.

The ``base`` module defines the ``StorageAdapter`` ABC. The ``sqlite``
module provides the default SQLite implementation. Future backends
(Postgres, Firebase, Qdrant, ...) land as additional modules here or
as sibling packages under ``packages/``.
"""
