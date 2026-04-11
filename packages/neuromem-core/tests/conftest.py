"""Shared pytest fixtures for neuromem-core.

This module is populated incrementally by downstream tasks:
  - T011: MockEmbeddingProvider, MockLLMProvider fixtures
  - T012: storage_adapter parametrised fixture (initially empty params)
  - T013: SQLiteAdapter added to storage_adapter params
  - T025: DictStorageAdapter added to storage_adapter params
"""
